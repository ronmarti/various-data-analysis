from enum import Enum
import math
from typing import Tuple
import gpytorch
import pyro
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.functional import Tensor
import torch.optim as optim
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from pyro.distributions import constraints
import pyro.distributions as dist

class GPKernels(Enum):
    RBF = gpytorch.kernels.RBFKernel
    MATERN = gpytorch.kernels.MaternKernel
    SPECTRAL = gpytorch.kernels.SpectralMixtureKernel
    PERIODIC = gpytorch.kernels.PeriodicKernel


class SingleApproxGP(gpytorch.models.ApproximateGP):
    def __init__(self,
                 num_inducing=64,
                 time_range: Tuple[float, float] = None,
                 variational_mean_init: Tensor = None,
                 learn_inducing_locations=False,
                 ):
        if time_range is None:
            time_range = (0, 1)
        # Define all the variational stuff
        inducing_points = torch.linspace(time_range[0], time_range[1], num_inducing)
        
        num_inducing = inducing_points.size(0)
        variational_dist = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_dist,
            learn_inducing_locations=learn_inducing_locations
        )
        if variational_mean_init is not None:
            variational_dist.initialize(variational_mean=variational_mean_init)

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GPKernels.MATERN.value(nu=2.5)
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def predict(self, x):
        return self.likelihood(self(x))


class SingleApproxGPpl(LightningModule):
    def __init__(self,
                 num_inducing=64,
                 time_range: Tuple[float, float] = None,
                 variational_mean_init: Tensor = None,
                 datalength = 100,
                 lr = 0.01,
                 learn_inducing_locations=False,
                 use_predictive_mll=False):
        super().__init__()
        if time_range is None:
            time_range = (0, 1)
        self.gp = SingleApproxGP(num_inducing=num_inducing,
                                 time_range=time_range,
                                 variational_mean_init=variational_mean_init,
                                 learn_inducing_locations=learn_inducing_locations)
        self.lr = lr
        self.use_predictive_mll = use_predictive_mll
        self.datalength = datalength
        self.save_hyperparameters()

    def forward(self, x):
        output = self.gp(x)
        return output

    def predict(self, x):
        return self.gp.predict(x)

    def configure_optimizers(self):
        self.mll = VariationalELBO(self.gp.likelihood, self.gp, num_data=self.datalength)
        if self.use_predictive_mll:
            self.mll = PredictiveLogLikelihood(self.gp.likelihood, self.gp, num_data=self.datalength)
        
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


def fit(module,
        train_dataloader,
        max_epochs=1000,
        patience=10,
        min_delta=1e-4,
        verbose=True,
        enable_logger=True,
        enable_checkpointing = True):
    '''Runs training with earlystopping and constructs default trainer for you.'''
    callbacks = [
        EarlyStopping(
            monitor='train_loss',
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode='min',
            check_on_train_epoch_end=True
        )
    ]

    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            save_top_k=1,
            mode='min',
        )
        callbacks += checkpoint_callback

    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        checkpoint_callback=enable_checkpointing,
        logger=enable_logger
    )
    trainer.fit(module, train_dataloader)
    # gp_state_dict = module.gp.state_dict()
    # torch.save(gp_state_dict, Path(trainer.log_dir) / Path('gp_state_dict.pth'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')

    num_inducing=32
    num_iter=1000
    lr=0.1
    patience = 50
    use_predictive_mll = False
    learn_inducing_locations = False
    slow_mode = False  # enables checkpointing and logging
    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 256
    datalength = NSamp
    print(f'NSamp = {NSamp}')
    time_range = (0, 2.5)


    # Generate synthetic data
    # here we generate some synthetic samples

    X_tens = torch.linspace(time_range[0], time_range[1], NSamp)

    # Shape Size([num_samples, lat_dim])
    Y_tens_raw = 2* torch.sin(X_tens * (2 * math.pi)) + torch.randn(X_tens.size()) * 0.01
    Y_tens = Y_tens_raw.round()

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(X_tens, Y_tens, label='Obs')
    ax.plot(X_tens, Y_tens_raw, 'b*', label='Latent')
    ax.legend()
    ax.set_title('Observations Quantized')
    fig.tight_layout()
    plt.show()

    variational_mean_init = Y_tens.index_select(0,torch.randperm(Y_tens.size(0)))[:num_inducing]

    model = SingleApproxGPpl(
        num_inducing=num_inducing,
        time_range=(X_tens[0], X_tens[-1]),
        variational_mean_init=variational_mean_init,
        lr=lr,
        learn_inducing_locations = learn_inducing_locations,
        use_predictive_mll = use_predictive_mll,
        datalength=datalength
    )

    loader = DataLoader(
        TensorDataset(
            X_tens,
            Y_tens
        ),
        batch_size=8256,
        shuffle=True
    )
    fit(model,
        train_dataloader=loader,
        max_epochs=num_iter,
        patience=patience,
        verbose=False,
        enable_checkpointing=slow_mode,
        enable_logger=True)
        
    # define test set (optionally on GPU)
    denser = 1.25 # make test set 2 times denser then the training set
    test_x = torch.linspace(time_range[0], 2*time_range[1], round(denser * NSamp)).float()#.cuda()

    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = model(test_x)
        lower, upper = output.confidence_region()
        fn_mean = output.mean.detach()

    # Initialize plots
    fig, ax = plt.subplots(1, 1)

    # for task, ax in enumerate(axes):
    # Plot training data as black stars
    ax.plot(X_tens, Y_tens, 'k*')
    # Predictive mean as blue line
    ax.plot(test_x, fn_mean.numpy(), 'b')
    # Shade in confidence
    ax.fill_between(test_x, lower.numpy(), upper.numpy(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.legend(['Data', 'Mean', '$\pm\sigma$'])
    # ax.set_title(f'Task {task + 1}')

    fig.tight_layout()
    plt.show()


    print(f'Done')

