import math
import gpytorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from pytorch_lightning import LightningModule
from factorio.models.gp_approx_multi_task import MultitaskApproxGP

from factorio.utils.data_prep import make_gpnarx_regressor
from factorio.utils.fitting import fit



class MultitaskApproxGPpl(LightningModule):
    def __init__(self,
                 obs_dim: int,
                 num_latents: int,
                 input_dim: int,
                 num_inducing=32,
                 inputs_range=None,
                 learn_inducing_locations=True,
                 subsample_size=1,
                 lr = 0.1,
                 use_predictive_mll=False) -> None:
        super().__init__()
        self.gp = MultitaskApproxGP(
            num_tasks=obs_dim,
            input_dim=input_dim,
            num_latents=num_latents,
            num_inducing=num_inducing,
            inputs_range=inputs_range,
            learn_inducing_locations=learn_inducing_locations
        )

        self.subsample_size = subsample_size
        self.obs_dim = obs_dim
        self.lr = lr
        self.use_predictive_mll = use_predictive_mll
        self.save_hyperparameters()

    def forward(self, x):
        output = self.gp(x)
        return output
    
    def predict(self, X):
        return self.gp.predict(X)

    def configure_optimizers(self):
        datalength = 100
        self.mll = VariationalELBO(self.gp.likelihood, self.gp, num_data=datalength)
        if self.use_predictive_mll:
            self.mll = PredictiveLogLikelihood(self.gp.likelihood, self.gp, num_data=datalength)
        
        # return optim.Adam(
        #     [self.parameters(), self.gp.parameters()],
        #     lr=self.lr
        # )
        self.gp.train()
        return optim.Adam(
            self.gp.parameters(),
            lr=self.lr
        )
        
    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        # Output from model
        output = self(x.squeeze(0))
        loss = -self.mll(output, y.squeeze(0))#.sum()
        self.log('train_loss', loss.detach(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')
    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 1000
    print(f'NSamp = {NSamp}')
    time_range = (0, 1.5)
    num_inducing = 16

    X_tens = torch.linspace(time_range[0], time_range[1], NSamp)

    # Shape Size([num_samples, lat_dim])
    Y_tens = torch.stack([
        torch.sin(X_tens * (2 * math.pi)) + torch.randn(X_tens.size()) * 0.2,
        torch.cos(X_tens * (2 * math.pi)) + torch.randn(X_tens.size()) * 0.2,
        torch.sin(X_tens * (2 * math.pi)) + 2 * torch.cos(X_tens * (2 * math.pi)) + torch.randn(X_tens.size()) * 0.2,
        -torch.cos(X_tens * (2 * math.pi)) + torch.randn(X_tens.size()) * 0.2,
    ], -1)

    num_tasks = Y_tens.size(1)  # must be the size in the second dimension of data
    num_latents = 2  # can be arbitrary number

    fig, axes = plt.subplots(1, num_tasks, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.scatter(X_tens, Y_tens[:, i])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Observations with Noise')
    fig.tight_layout()
    plt.show()

    regressor, y_regressor = make_gpnarx_regressor(X_tens, Y_tens, y_lag=5, u_lag=1)


    model = MultitaskApproxGPpl(
                 obs_dim=y_regressor.size(-1),
                 num_latents=num_latents,
                 input_dim=regressor.size(-1),
                 num_inducing=num_inducing,
                 inputs_range=None,
                 learn_inducing_locations=False,
                 subsample_size=1,
                 lr = 0.1,
                 use_predictive_mll=True)

    loader = DataLoader(
        TensorDataset(
            regressor,
            y_regressor
        ),
        batch_size=256,
        shuffle=True
    )
    fit(model,
        train_dataloader=loader,
        max_epochs=1000,
        patience=50,
        verbose=False)

    # define test set (optionally on GPU)
    denser = 2  # make test set 2 times denser then the training set

    model.eval()
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model.predict(regressor)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # Initialize plots
    time_aux = torch.arange(mean.size(0))
    fig, axes = plt.subplots(1, num_tasks)

    for task, ax in enumerate(axes):
        # Plot training data as black stars
        ax.plot(time_aux, y_regressor[:, task], 'k*')
        # Predictive mean as blue line
        ax.plot(time_aux, mean[:, task].numpy(), 'b')
        # Shade in confidence
        ax.fill_between(time_aux, lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Data', 'Mean', '$\pm\sigma$'])
        ax.set_title(f'Task {task + 1}')

    fig.tight_layout()
    plt.show()

    print(f'Done')
