import math
from typing import Tuple
from pyro.distributions import constraints
import pyro.distributions as dist
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from tqdm import trange


class SingleApproxGP(gpytorch.models.ApproximateGP):
    def __init__(self,
                 num_inducing=64,
                 time_range: Tuple[float, float] = None,
                 name_prefix="mixture_gp"):
        self.name_prefix = name_prefix
        if time_range is None:
            time_range = (0, 1)
        # Define all the variational stuff
        inducing_points = torch.linspace(time_range[0], time_range[1], num_inducing)
        variational_dist = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points,
            variational_dist,
            learn_inducing_locations=False
        )

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp", self)

        # register the variational parameters with Pyro.
        ss_offset = pyro.param("ss_offset_q",
                               torch.tensor(0.224),
                               constraint=constraints.positive)
        noise_scale = pyro.param("process_noise_scale_q",
                                 torch.tensor(1.01),
                                 constraint=constraints.positive)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

            # Use the link function to convert GP samples into observations dists parameters
            transformed_samples = function_samples

            transforms = [dist.transforms.ExpTransform(), dist.transforms.AffineTransform(loc=ss_offset, scale=1.0)]
            transformed_dist = dist.TransformedDistribution(
                dist.Normal(transformed_samples, noise_scale),
                transforms
                )

            # Sample from observed distribution
            return pyro.sample(
                self.name_prefix + ".y",
                transformed_dist,
                obs=y
            )

    def fit(self, tr_x, tr_y, num_iter = 100, num_particles = 256):
        optimizer = pyro.optim.Adam({"lr": 0.01})
        elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True)
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, elbo)

        loader = DataLoader(
            TensorDataset(
                tr_x,
                tr_y
            ),
            batch_size=256,
            shuffle=True
        )
        
        self.train()
        iterator = trange(num_iter)
        for i in iterator:
            for x, y in loader:
                self.zero_grad()
                loss = svi.step(x, y)
                iterator.set_postfix(
                    loss=loss,
                    lengthscale=self.covar_module.base_kernel.lengthscale.item(),
                    ss_offset=pyro.param('ss_offset_q').item(),
                    process_noise_scale_q=pyro.param('process_noise_scale_q').item(),
                    )

    # Here's a quick helper function for getting smoothed percentile values from samples
    @staticmethod
    def percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95]):
        num_samples = samples.size(0)
        samples = samples.sort(dim=0)[0]

        # Get samples corresponding to percentile
        percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]

        # Smooth the samples
        kernel = torch.full((1, 1, 5), fill_value=0.2)
        percentiles_samples = [
            torch.nn.functional.conv1d(percentile_sample.view(1, 1, -1), kernel, padding=2).view(-1)
            for percentile_sample in percentile_samples
        ]

        return percentiles_samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')

    # Here we specify a 'true' latent function lambda
    lat_fn = lambda x: torch.sin(2 * math.pi * x) + torch.sin(3.3 * math.pi * x)
    obs_fn = lambda x, scale, offset: offset + x.exp() + scale*torch.randn(x.size(), dtype=torch.float)

    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 100
    print(f'NSamp = {NSamp}')
    time_range = (0, 2.5)
    process_noise_scale = 0.03
    ss_offset = 0.7

    X = torch.linspace(time_range[0], time_range[1], NSamp)
    fx = lat_fn(X)
    Y = obs_fn(fx, scale=process_noise_scale, offset=ss_offset)
    X_tens = torch.tensor(X).float()
    Y_tens = torch.tensor(Y).float()

    fig, (ax_lat, ax_sample) = plt.subplots(1, 2, figsize=(10, 3))
    ax_lat.plot(X, fx)
    ax_lat.set_xlabel('x')
    ax_lat.set_ylabel('$f(x)$')
    ax_lat.set_title('Latent function')
    ax_sample.scatter(X_tens, Y_tens)
    ax_sample.set_xlabel('x')
    ax_sample.set_ylabel('y')
    ax_sample.set_title('Observations with Noise')
    plt.show()

    model = SingleApproxGP(num_inducing=64, time_range=time_range)
    model.fit(X_tens, Y_tens, num_iter=100, num_particles=256)

    # define test set (optionally on GPU)
    denser = 2 # make test set 2 times denser then the training set
    test_x = torch.linspace(time_range[0], 2*time_range[1], denser * NSamp).float()#.cuda()

    model.eval()
    with torch.no_grad():
        output = model(test_x)

    # Get E[exp(f)] via f_i ~ GP, 1/n \sum_{i=1}^{n} exp(f_i).
    # Similarly get the 5th and 95th percentiles
    samples = output(torch.Size([1000]))
    lower, fn_mean, upper = SingleApproxGP.percentiles_from_samples(samples)

    # Draw some simulated y values
    ss_offset = pyro.param('ss_offset_q').item()
    process_noise_scale_q = pyro.param('process_noise_scale_q').item()

    y_sim = obs_fn(fn_mean, scale=process_noise_scale_q, offset=ss_offset)

    # visualize the result
    fig, (ax_func, ax_samp) = plt.subplots(1, 2, figsize=(12, 3))
    line, = ax_func.plot(test_x, fn_mean.detach().cpu().numpy(), label='GP prediction')
    ax_func.fill_between(
        test_x, lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(), color=line.get_color(), alpha=0.5
    )

    # func.plot(test_x, lat_fn(test_x), label='True latent function')
    ax_func.legend()

    # sample from p(y|D,x) = \int p(y|f) p(f|D,x) df (doubly stochastic)
    ax_samp.scatter(X_tens, Y_tens, alpha = 0.5, label='True train data', color='orange')
    ax_samp.plot(test_x, y_sim.cpu().detach().numpy(), alpha=0.5, label='Sample from the model')
    # samp.fill_between(
    #     test_x, y_sim_lower.detach().cpu().numpy(),
    #     y_sim_upper.detach().cpu().numpy(), color=line.get_color(), alpha=0.5
    # )
    ax_samp.legend()
    plt.show()

    print(f'Done')
