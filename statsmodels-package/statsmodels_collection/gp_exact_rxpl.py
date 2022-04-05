import math
import gpytorch
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, PredictiveLogLikelihood
from pytorch_lightning import LightningModule
from pathlib import Path
from enum import Enum
from statsmodels_collection.fitting import fit


class GPKernels(Enum):
    RBF = gpytorch.kernels.RBFKernel
    MATERN = gpytorch.kernels.MaternKernel
    SPECTRAL = gpytorch.kernels.SpectralMixtureKernel


class SingletaskExactGP(ExactGP):
    def __init__(self,
                 tr_y: Tensor,
                 tr_x: Tensor = None,
                 likelihood: gpytorch.likelihoods.Likelihood = None,
                 kernel: gpytorch.kernels.Kernel = None,
                 ard: int = 1):
        num_data = tr_y.size(0)
        if tr_x is None:
            tr_x = torch.linspace(0, 1., num_data).unsqueeze(-1)

        if likelihood is None:
            likelihood = GaussianLikelihood()
        super().__init__(tr_x, tr_y, likelihood)

        if kernel is None:
            # kernel = GPKernel.MATERN.value(nu=2.5, ard_num_dims=ard)
            kernel = GPKernels.SPECTRAL.value(
                num_mixtures=2,
                ard_num_dims=ard
            )
            kernel.initialize_from_data(tr_x, tr_y)

        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(kernel, ard_num_dims=ard)
        self.covar_module = kernel


    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        return self.likelihood(self(x))


class SingletaskExactGPpl(LightningModule):
    def __init__(self,
                 tr_y: Tensor,
                 tr_x: Tensor = None,
                 likelihood=None,
                 ard: int = 1,
                 lr=0.1):
        """Module for easy exact GP training with early stopping and patience and
        logging.
        !!! You have to train on the training data - CHECK tr_y, tr_x order in argument !!!

        Args:
            tr_y (Tensor): _description_
            tr_x (Tensor, optional): _description_. Defaults to None.
            likelihood (_type_, optional): _description_. Defaults to None.
            ard (int, optional): _description_. Defaults to 1.
            lr (float, optional): _description_. Defaults to 0.1.
        """                 
        super().__init__()
        self.gp = SingletaskExactGP(tr_y=tr_y,
                                    tr_x=tr_x,
                                    likelihood=likelihood,
                                    ard=ard)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        output = self.gp(x)
        return output
    
    def predict(self, X):
        return self.gp.predict(X)

    def configure_optimizers(self):
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
 
        self.gp.train()
        self.gp.likelihood.train()
        return optim.Adam(
            self.gp.parameters(),
            lr=self.lr
        )
        
    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        # Output from model
        output = self(x)
        loss = -self.mll(output, y.squeeze(0))#.sum()
        self.log('train_loss', loss.detach(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')
    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 100
    print(f'NSamp = {NSamp}')
    time_range = (0, 6.)

    X_tens = torch.linspace(time_range[0], time_range[1], NSamp)

    # Shape Size([num_samples, lat_dim])
    Y_tens = torch.sin(X_tens * (2 * math.pi)) + torch.randn(X_tens.size()) * 0.1

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.scatter(X_tens, Y_tens)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Observations with Noise')
    fig.tight_layout()
    plt.show()

    model = SingletaskExactGPpl(
        Y_tens,
        X_tens
    )

    loader = DataLoader(
        TensorDataset(
            X_tens,
            Y_tens
        ),
        batch_size=1256,
        shuffle=False
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
        test_x = torch.linspace(time_range[0], 1.2*time_range[1], denser * NSamp).float()#.cuda()

        predictions = model.predict(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # Initialize plots
    fig, ax = plt.subplots(1, 1)

    # for task, ax in enumerate(axes):
    # Plot training data as black stars
    ax.plot(X_tens, Y_tens, 'k*')
    # Predictive mean as blue line
    ax.plot(test_x, mean.numpy(), 'b')
    # Shade in confidence
    ax.fill_between(test_x, lower.numpy(), upper.numpy(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.legend(['Data', 'Mean', '$\pm\sigma$'])
    # ax.set_title(f'Task {task + 1}')

    fig.tight_layout()
    plt.show()

    print(f'Done')
