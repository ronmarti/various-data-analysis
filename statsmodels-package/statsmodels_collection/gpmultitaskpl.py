import math
from typing import Tuple
import gpytorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.functional import Tensor
import torch.optim as optim
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from factorio.gpmodels.gpmultitask import MultitaskApproxGP
from pathlib import Path


class MultitaskApproxGPpl(LightningModule):
    def __init__(self,
                 num_tasks,
                 num_latents,
                 num_inducing=64,
                 time_range: Tuple[float, float] = None,
                 likelihood=None,
                 lr=0.1,
                 batch_size=512,
                 use_predictive_mll=False):
        super().__init__()
        if time_range is None:
            time_range = (0, 1)
        self.gp = MultitaskApproxGP(num_latents=num_latents,
                                    num_tasks=num_tasks,
                                    time_range=time_range,
                                    num_inducing=num_inducing,
                                    likelihood=likelihood)
        self.use_predictive_mll = use_predictive_mll
        self.lr = lr
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self, x):
        output = self.gp(x)
        return output
    
    def predict(self, X):
        return self.gp.predict(X)

    def configure_optimizers(self):
        datalength = len(self.train_dataloader.dataloader.dataset)
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
        loss = -self.mll(output, y.squeeze(0)).sum()
        self.log('train_loss', loss.detach(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}


def fit(module, train_dataloader, max_epochs=1000, patience=10, min_delta=1e-4, verbose=True):
    '''Runs training with earlystopping and constructs default trainer for you.'''
    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=min_delta,
        patience=patience,
        verbose=verbose,
        mode='min',
        check_on_train_epoch_end=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        save_top_k=1,
        mode='min',
    )

    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback,
                    early_stop_callback]
    )
    trainer.fit(module, train_dataloader)
    gp_state_dict = module.gp.state_dict()
    torch.save(gp_state_dict, Path(trainer.log_dir) / Path('gp_state_dict.pth'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')
    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 100
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

    model = MultitaskApproxGPpl(
        num_tasks=num_tasks,
        num_latents=num_latents,
        num_inducing=num_inducing,
        time_range=(X_tens[0], X_tens[-1]),
        batch_size=256
    )

    loader = DataLoader(
        TensorDataset(
            X_tens,
            Y_tens
        ),
        batch_size=256,
        shuffle=True
    )
    fit(model,
        train_dataloader=loader,
        max_epochs=100,
        patience=10,
        verbose=False)

    # define test set (optionally on GPU)
    denser = 2  # make test set 2 times denser then the training set

    model.eval()
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(time_range[0], 1.2*time_range[1], denser * NSamp).float()#.cuda()

        predictions = model.predict(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # Initialize plots
    fig, axes = plt.subplots(1, num_tasks)

    for task, ax in enumerate(axes):
        # Plot training data as black stars
        ax.plot(X_tens, Y_tens[:, task], 'k*')
        # Predictive mean as blue line
        ax.plot(test_x, mean[:, task].numpy(), 'b')
        # Shade in confidence
        ax.fill_between(test_x, lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Data', 'Mean', '$\pm\sigma$'])
        ax.set_title(f'Task {task + 1}')

    fig.tight_layout()
    plt.show()

    print(f'Done')
