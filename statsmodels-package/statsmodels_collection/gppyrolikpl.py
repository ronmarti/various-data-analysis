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
from factorio.gpmodels.gppyrolik import SingleApproxGP
from pathlib import Path


class SingleApproxGPpl(LightningModule):
    def __init__(self,
                 num_inducing=64,
                 time_range: Tuple[float, float] = None,
                 name_prefix="mixture_gp",
                 lr = 0.01,
                 num_particles = 64):
        super().__init__()
        self.automatic_optimization = False
        if time_range is None:
            time_range = (0, 1)
        self.gp = SingleApproxGP(num_inducing=num_inducing,
                                 time_range=time_range,
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
        optimizer = pyro.optim.Adam({"lr": 0.01})
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

    # Here we specify a 'true' latent function lambda
    lat_fn = lambda x: torch.sin(2 * math.pi * x) + torch.sin(3.3 * math.pi * x)
    obs_fn = lambda x, scale, offset: offset + x.exp() + scale*torch.randn(x.size(), dtype=torch.float)

    num_inducing=64
    num_iter=1000
    num_particles=32
    slow_mode = False  # enables checkpointing and logging
    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 1000
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

    model = SingleApproxGPpl(
        num_inducing=num_inducing,
        time_range=(X_tens[0], X_tens[-1]),
        num_particles=num_particles
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
        patience=10,
        verbose=False,
        enable_checkpointing=slow_mode,
        enable_logger=True)
        
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

