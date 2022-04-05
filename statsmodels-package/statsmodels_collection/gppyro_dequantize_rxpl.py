from enum import Enum
import math
from typing import Tuple
import gpytorch
import pyro
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import pyro.distributions as dist
from factorio.utils.fitting import fit

class GPKernels(Enum):
    RBF = gpytorch.kernels.RBFKernel
    MATERN = gpytorch.kernels.MaternKernel
    SPECTRAL = gpytorch.kernels.SpectralMixtureKernel
    PERIODIC = gpytorch.kernels.PeriodicKernel


class QuantizedApproxGP(gpytorch.models.ApproximateGP):
    def __init__(self,
                 num_inducing=64,
                 time_range: Tuple[float, float] = None,
                 variational_mean_init: torch.Tensor = None,
                 name_prefix="mixture_gp"):
        self.name_prefix = name_prefix
        if time_range is None:
            time_range = (0, 1)
        # Define all the variational stuff
        inducing_points = torch.linspace(time_range[0], time_range[1], num_inducing)
        
        num_inducing = inducing_points.size(0)
        variational_dist = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points,
            variational_dist,
            learn_inducing_locations=False
        )
        if variational_mean_init is not None:
            variational_dist.initialize(variational_mean=variational_mean_init)

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean()

        # kernel = GPKernels.SPECTRAL.value(
        #         num_mixtures=3
        #     )
        # self.covar_module = kernel
        # kernel.initialize(mixture_means=torch.randn(3, 1, 1).exp())

        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     GPKernels.MATERN.value(nu=2.5,
        #                            lengthscale_constraint=gpytorch.constraints.GreaterThan(0.1)),
        #     outputscale_constraint=gpytorch.constraints.Interval(0.01, 0.05))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GPKernels.MATERN.value(nu=2.5)
        )
        # self.covar_module = gpytorch.kernels.ScaleKernel(GPKernels.RBF.value())

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

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

            # Sample from observed distribution
            return pyro.sample(
                self.name_prefix + ".y",
                dist.Normal(loc=function_samples, scale=0.0625),
                obs=y
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


class QuantizedApproxGPpl(LightningModule):
    def __init__(self,
                 num_inducing=64,
                 time_range: Tuple[float, float] = None,
                 variational_mean_init: torch.Tensor = None,
                 name_prefix="mixture_gp",
                 lr = 0.01,
                 num_particles = 64):
        super().__init__()
        self.automatic_optimization = False
        if time_range is None:
            time_range = (0, 1)
        self.gp = QuantizedApproxGP(num_inducing=num_inducing,
                                 time_range=time_range,
                                 variational_mean_init=variational_mean_init,
                                 name_prefix=name_prefix)
        self.lr = lr
        self.num_particles = num_particles
        self.save_hyperparameters()

    def forward(self, x):
        output = self.gp(x)
        return output
    
    # def predict(self, X):
    #     return self.gp.predict(X)

    def configure_optimizers(self):
        optimizer = pyro.optim.Adam({"lr": self.lr})
        elbo = pyro.infer.Trace_ELBO(num_particles=self.num_particles, vectorize_particles=True, retain_graph=True)
        self.svi = pyro.infer.SVI(self.gp.model, self.gp.guide, optimizer, elbo)
        self.train()
        return None
        
    def training_step(self, batch, batch_idx, *args, **kwargs):
        tr_x, tr_y = batch
        self.zero_grad()
        loss = torch.as_tensor(self.svi.step(tr_x, tr_y))
        # Output from model
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': {'train_loss': loss}}


# def fit(module,
#         train_dataloader,
#         max_epochs=1000,
#         patience=10,
#         min_delta=1e-4,
#         verbose=True,
#         enable_logger=True,
#         enable_checkpointing = True):
#     '''Runs training with earlystopping and constructs default trainer for you.'''
#     callbacks = [
#         EarlyStopping(
#             monitor='train_loss',
#             min_delta=min_delta,
#             patience=patience,
#             verbose=verbose,
#             mode='min',
#             check_on_train_epoch_end=True
#         )
#     ]

#     if enable_checkpointing:
#         checkpoint_callback = ModelCheckpoint(
#             monitor='train_loss',
#             save_top_k=1,
#             mode='min',
#         )
#         callbacks += checkpoint_callback

#     # trainer = pl.Trainer(gpus=8) (if you have GPUs)
#     trainer = Trainer(
#         max_epochs=max_epochs,
#         callbacks=callbacks,
#         checkpoint_callback=enable_checkpointing,
#         logger=enable_logger
#     )
#     trainer.fit(module, train_dataloader)
#     # gp_state_dict = module.gp.state_dict()
#     # torch.save(gp_state_dict, Path(trainer.log_dir) / Path('gp_state_dict.pth'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')

    num_inducing=128
    num_iter=1000
    num_particles=32
    lr=0.01
    patience = 50
    slow_mode = False  # enables checkpointing and logging
    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 256
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

    model = QuantizedApproxGPpl(
        num_inducing=num_inducing,
        time_range=(X_tens[0], X_tens[-1]),
        variational_mean_init=variational_mean_init,
        num_particles=num_particles,
        lr=lr
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

    # Get E[exp(f)] via f_i ~ GP, 1/n \sum_{i=1}^{n} exp(f_i).
    # Similarly get the 5th and 95th percentiles
    samples = output(torch.Size([1000]))
    lower, fn_mean, upper = QuantizedApproxGP.percentiles_from_samples(samples)

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

