import math
from typing import Tuple
import gpytorch
import torch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from tqdm.std import trange


class MultitaskApproxGP(gpytorch.models.ApproximateGP):
    """Deprecated, use factorio.models.gp_approx_multi_task which supports
    multidim inputs.
    """    
    def __init__(self, num_latents, num_tasks,
                 time_range: Tuple[float, float] = None,
                 num_inducing=64,
                 likelihood=None,
                 learn_inducing_locations=True):
        if time_range is None:
            time_range = (0, 1)

        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.linspace(time_range[0], time_range[1], num_inducing).repeat(num_latents, 1).unsqueeze(-1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

        self.likelihood = likelihood
        if likelihood is None:
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        return self.likelihood(self(x))

    def fit(self, tr_x, tr_y,
            num_iter=100,
            lr=0.1,
            use_predictive_mll=False):
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.parameters()},
            # {'params': self.likelihood.parameters()},
        ], lr=lr)

        # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
        mll = VariationalELBO(self.likelihood, self, num_data=tr_y.size(0))
        if use_predictive_mll:
            mll = PredictiveLogLikelihood(self.likelihood, self, num_data=tr_y.size(0))

        # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
        # effective for VI.
        iterator = trange(num_iter)
        for i in iterator:
            # Within each iteration, we will go over each minibatch of data
            optimizer.zero_grad()
            output = self(tr_x)
            loss = -mll(output, tr_y)
            iterator.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path

    print(f'Run {__file__}')
    out_root = Path('.out')
    out_root.mkdir(parents=True, exist_ok=True)
    out_name = Path('gpmultitask_state_dict.pth')
    out_path = out_root / out_name

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

    model = MultitaskApproxGP(num_latents=num_latents,
                             num_tasks=num_tasks,
                             num_inducing=num_inducing,
                             time_range=time_range)
    model.fit(X_tens, Y_tens, num_iter=300, use_predictive_mll=True)
    gp_state_dict = model.state_dict()
    torch.save(gp_state_dict, out_path)
    print(f'Model state_dict saved to: {out_path}')

    # define test set (optionally on GPU)
    denser = 2 # make test set 2 times denser then the training set

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
