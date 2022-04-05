from enum import Enum
from typing import Tuple
import gpytorch
import torch
from torch.types import _TensorOrTensors
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import LMCVariationalStrategy
from gpytorch.likelihoods import MultitaskGaussianLikelihood

class GPKernels(Enum):
    RBF = gpytorch.kernels.RBFKernel
    MATERN = gpytorch.kernels.MaternKernel
    SPECTRAL = gpytorch.kernels.SpectralMixtureKernel
    PERIODIC = gpytorch.kernels.PeriodicKernel


class MultitaskApproxGP(gpytorch.models.ApproximateGP):
    def __init__(self,
                 num_tasks:int,
                 input_dim: int,
                 num_latents:int = 2,
                 num_inducing=64,
                 inputs_range: _TensorOrTensors = None,
                 learn_inducing_locations=False,
                 ):
        """Universal class for multi task Approximate Gaussian Process.

        Args:
            num_tasks:
            input_dim:
            num_latents:
            num_inducing (int, optional): number of inducing points
            per input dim. Overridden if tensor of inducing points is provided.
            Defaults to 64.
            inputs_range (_TensorOrTensors, optional): ranges of inputs,
            `(input_dim, 2)` tensor of initial start and stop boundaries.
            Can be overcome through training
            with `learn_inducing_locations=True`. Defaults to None.
            learn_inducing_locations (bool, optional): whether the locations
            of inducing points can be moved through learning.
            Defaults to False.
        """                 
        if inputs_range is None:
            inputs_range = torch.tensor([[0, 1.]]).repeat(input_dim, 1)

        # size must be (num_latents, num_inducing, in_dim)
        # inducing_points = torch.stack([
        #     torch.linspace(dim_range[0], dim_range[1], num_inducing)
        #     for dim_range in inputs_range
        # ], dim=-1).repeat(num_latents, 1, 1)
        inducing_points = torch.randn(num_latents, num_inducing, input_dim)

        size_latents = torch.Size([num_latents])

        num_inducing = inducing_points.size(-2)
        variational_dist = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=size_latents
        )
        variational_strategy = LMCVariationalStrategy(
            VariationalStrategy(
                self,
                inducing_points,
                variational_dist,
                learn_inducing_locations=learn_inducing_locations
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        # Standard initializtation
        super().__init__(variational_strategy)


        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=size_latents
            )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GPKernels.MATERN.value(nu=2.5, batch_shape=size_latents),
            batch_shape=size_latents
        )

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

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

    # size must be (num_samples, in_dim)
    x = torch.linspace(0, 1., 100).unsqueeze(-1).repeat(1, 3)

    model = MultitaskApproxGP(
        num_tasks=2,
        input_dim=3,
        num_latents=4,
        num_inducing=num_inducing,
        inputs_range=torch.tensor([[0, 1.],
                                [0, 1.],
                                [0, 1.]]),
        learn_inducing_locations = learn_inducing_locations
    )

    pred = model(x)
    plt.plot(x[:, 0], pred.mean.detach())
    plt.show()
    print(f'Done')
