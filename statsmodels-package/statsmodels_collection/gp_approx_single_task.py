from enum import Enum
from typing import Tuple
import gpytorch
import torch
from torch.types import _TensorOrTensors
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPKernels(Enum):
    RBF = gpytorch.kernels.RBFKernel
    MATERN = gpytorch.kernels.MaternKernel
    SPECTRAL = gpytorch.kernels.SpectralMixtureKernel
    PERIODIC = gpytorch.kernels.PeriodicKernel


class SingleApproxGP(gpytorch.models.ApproximateGP):
    def __init__(self,
                 num_inducing=64,
                 inputs_range: _TensorOrTensors = None,
                 variational_mean_init: _TensorOrTensors = None,
                 learn_inducing_locations=False,
                 ):
        """Universal class for single task Approximate Gaussian Process.

        Args:
            num_inducing (int, optional): number of inducing points
            per input dim. Defaults to 64.
            inputs_range (_TensorOrTensors, optional): ranges of inputs,
            `(in_dim, 2)` tensor of initial start and stop boundaries.
            Can be overcome through training
            with `learn_inducing_locations=True`. Defaults to None.
            variational_mean_init (_TensorOrTensors, optional): initialization
            of inducing points values. Must be of size `(num_inducing, in_dim)`.
            Defaults to None.
            learn_inducing_locations (bool, optional): whether the locations
            of inducing points can be moved through learning.
            Defaults to False.
        """                 
        if inputs_range is None:
            inputs_range = torch.tensor([[0, 1.]])

        inducing_points = torch.stack([
            torch.linspace(dim_range[0], dim_range[1], num_inducing)
            for dim_range in inputs_range
        ], dim=-1)
        
        
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')
    print(f'No training run provided, just prior prediction.')

    num_inducing=32
    learn_inducing_locations = False
    variational_mean_init = None

    x = torch.linspace(0, 1., 100).unsqueeze(-1).repeat(1, 3)

    model = SingleApproxGP(
        num_inducing=num_inducing,
        inputs_range=torch.tensor([[0, 1.],
                                [0, 1.],
                                [0, 1.]]),
        variational_mean_init=variational_mean_init,
        learn_inducing_locations = learn_inducing_locations
    )

    pred = model.predict(x)
    plt.plot(x, pred.mean.detach())
    plt.show()
    print(f'Done')
