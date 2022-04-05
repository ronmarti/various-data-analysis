from typing import Tuple
import pyro
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.functional import Tensor
import torch.optim as optim
from torch.distributions import constraints
from pytorch_lightning import LightningModule
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.ops.indexing import Vindex
from gp_approx_multi_task import MultitaskApproxGP
import gpytorch
from factorio.utils.fitting import fit

import pyro_hmmio as hmm


class GPNARXpl(LightningModule):

    # svi: SVI

    def __init__(self,
                 hidden_dim: int,
                 input_dim: int,
                 obs_dim: int,
                 num_inducing=32,
                 num_latents=2,
                 inputs_range=None,
                 learn_inducing_locations=True,
                 subsample_size=1,
                 lr=0.1,
                 num_particles=10,
                 minibatch_size=512
                 ):
        super().__init__()
        self.automatic_optimization = False
        if inputs_range is not None:
            assert inputs_range.size(0) == input_dim

        self.dynamic_model = GPNARX(
                            hidden_dim=hidden_dim,
                            obs_dim=obs_dim,
                            num_latents=num_latents,
                            input_dim=input_dim,
                            num_inducing=num_inducing,
                            inputs_range=inputs_range,
                            learn_inducing_locations=learn_inducing_locations,
                            subsample_size=subsample_size)

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.num_particles = num_particles
        self.minibatch_size = minibatch_size
        self.save_hyperparameters()

    def forward(self, x: Tensor):
        """Predicts output given the driving inputs.

        Args:
            x (Tensor): the batch of driving inputs of size `(b, T, input_dim)`.

        Returns:
            Tensor: batch of predicted outputs of size `(b, T, obs_dim)`.
        """
        # Not implemented
        return None

    def configure_optimizers(self):
        pyro.clear_param_store()
        optimizer = Adam({"lr": 0.01})
        elbo = Trace_ELBO(num_particles=self.num_particles,
                          vectorize_particles=True,
                          retain_graph=True)
        self.svi = SVI(self.dynamic_model.model,
                       self.dynamic_model.guide,
                       optimizer,
                       elbo)
        self.train()
        return None
        
    def training_step(self, batch, batch_idx, *args, **kwargs):
        tr_inputs, tr_y = batch
        # self.zero_grad()
        loss_raw = self.svi.step(tr_y,
                                 tr_inputs)
        loss_tensor = torch.as_tensor(loss_raw)
        # Output from model
        self.log('train_loss', loss_tensor, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss_tensor, 'log': {'train_loss': loss_tensor}}


class GPNARX():
    def __init__(self,
                 hidden_dim: int,
                 obs_dim: int,
                 num_latents: int,
                 input_dim: int,
                 num_inducing=32,
                 inputs_range=None,
                 learn_inducing_locations=True,
                 subsample_size=1,
                 name_prefix='gpnarx') -> None:
        self.name_prefix = name_prefix

        self.transition_model = MultitaskApproxGP(
            num_tasks=hidden_dim,
            input_dim=input_dim,
            num_latents=num_latents,
            num_inducing=num_inducing,
            inputs_range=inputs_range,
            learn_inducing_locations=learn_inducing_locations
        )

        self.subsample_size = subsample_size
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

    def model(self,
              output_seqs: Tensor,
              intput_seqs: Tensor):
        """Pyro model function for IO state-space model of modal realization.
        All eigenvalues are doubled on the diagonal. To retrive minimal realization,
        merge eigenvalues in block with zeros on the off-diagonal, and sum corresponding
        elements in B and C matrices.

        Args:
            output_seqs (Tensor): size `(batches, lenght, obs_dim, 1)`
            intput_seqs (Tensor): size `(batches, lenght, input_dim)`
            hidden_dim (int): half the searched dimension of the model.
            input_dim (int): input dimension, must be same as `intput_seqs.size(2)`.
            obs_dim (int): output dimension, must be same as `output_seqs.size(2)`.
            num_sequences (int): must be equal to 
                `batches = intput_seqs.size(0) = output_seqs.size(0)`
            length (int): `length = intput_seqs.size(1) = output_seqs.size(1)`
            batch_size (int, optional): number of minibatch subsampled from given sequences.
                Defaults to None.
        """
        # pyro.module(self.name_prefix + ".gp", self) #

        length = output_seqs.size(-2)
        obs_dim = self.obs_dim

        # c = pyro.param('c',
        #            init_tensor= torch.rand(obs_dim, self.hidden_dim, dtype=torch.float))
        # r = pyro.param('r',
        #            init_tensor=0.01 * torch.eye(obs_dim,dtype=torch.float),
        #            constraint=constraints.lower_cholesky)

        # num_sequences = output_seqs.size(0)
        # mu_0 = pyro.param('mu_0',
        #                   init_tensor=torch.zeros(num_sequences,
        #                                           self.hidden_dim,
        #                                           dtype=torch.float))
        # with pyro.plate("sequences", num_sequences, subsample_size = self.subsample_size) as batch:
        #     x_t = mu_0[batch]
        #     for t in pyro.markov(range(length), history=1):
        #         # Get p(f) - prior distribution of latent function
        #         fn_transition = self.transition_model.pyro_model(x_t,
        #                                                          name_prefix=self.name_prefix + f".transition_{t}")
        #         mu_y_t = (c @ x_t.unsqueeze(-1)).squeeze(-1)
        #         pyro.sample(
        #             f"y_{t}",
        #             dist.MultivariateNormal(mu_y_t, scale_tril=r),
        #             obs=Vindex(output_seqs)[batch, t, :, 0],
        #         )
                
        #         # x_t is x_tp1 at this instance, after iteration it will become x_t
        #         x_t = pyro.sample(
        #             f"x_{t}",
        #             fn_transition
        #         )


        # second attempt
        # mu_0 = pyro.param('mu_0',
        #                   init_tensor=torch.zeros(self.hidden_dim,
        #                                           dtype=torch.float))
        # x_t = mu_0.unsqueeze(0)
        # for t in pyro.markov(range(length), history=1):
        #     # Get p(f) - prior distribution of latent function
        #     fn_transition = self.transition_model.pyro_model(x_t,
        #                                                      name_prefix=self.name_prefix + f".transition_{t}")
        #     # x_t is x_tp1 at this instance, after iteration it will become x_t
        #     x_t = pyro.sample(
        #         self.name_prefix + f"x_{t}",
        #         fn_transition
        #     )
        #     mu_y_t = (c @ x_t.unsqueeze(-1)).squeeze(-1)
        #     pyro.sample(
        #         f"y_{t}",
        #         dist.MultivariateNormal(mu_y_t, scale_tril=r),
        #         obs=Vindex(output_seqs)[t, :],
        #     )    
        
        with pyro.plate("sequences", length, subsample_size=self.subsample_size) as t:
            # Get p(f) - prior distribution of latent function
            fn_transition = self.transition_model.pyro_model(intput_seqs[t],
                                                             name_prefix=self.name_prefix + f".transition_{t}")
            # x_t is x_tp1 at this instance, after iteration it will become x_t
            pyro.sample(
                self.name_prefix + f".y_{t}",
                fn_transition,
                obs=Vindex(output_seqs)[t, :]
            )  

    def guide(self,
              output_seqs: Tensor,
              intput_seqs: Tensor):
        length = output_seqs.size(-2)
        # num_sequences = output_seqs.size(0)

        # mu_0 = pyro.param('mu_0',
        #                   init_tensor=torch.zeros(num_sequences,
        #                                           self.hidden_dim,
        #                                           dtype=torch.float))
        # with pyro.plate("sequences", num_sequences, subsample_size=self.subsample_size) as batch:
        #     x_t = mu_0[batch]
        #     for t in pyro.markov(range(length), history=1):
        #         fn_transition = self.transition_model.pyro_guide(x_t,
        #                                                          name_prefix=self.name_prefix + f".transition_{t}")
        #         # x_t is x_tp1 at this instance, after iteration it will become x_t
        #         x_t = pyro.sample(self.name_prefix + f".f(x_{t})",
        #                           fn_transition)

        # Second attempt / success
        # mu_0 = pyro.param('mu_0',
        #                   init_tensor=torch.zeros(self.hidden_dim,
        #                                           dtype=torch.float))
        # x_t = mu_0.unsqueeze(0)
        # for t in pyro.markov(range(length), history=1):
        #     fn_transition = self.transition_model.pyro_guide(x_t,
        #                                                      name_prefix=self.name_prefix + f".transition_{t}")
        #     x_t is x_tp1 at this instance, after iteration it will become x_t
        #     x_t = pyro.sample(self.name_prefix + f"x_{t}",
        #                       fn_transition)
        
        with pyro.plate("sequences", length, subsample_size=self.subsample_size) as t:
            fn_transition = self.transition_model.pyro_guide(intput_seqs[t],
                                                             name_prefix=self.name_prefix + f".transition_{t}")
            pyro.sample(self.name_prefix + f".y_{t}",
                              fn_transition)

# def fit(module,
#         train_dataloader,
#         max_epochs=1000,
#         patience=10,
#         min_delta=0,
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
#         callbacks += [checkpoint_callback]

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


def prepare_data() -> Tuple[Tensor, Tensor, Tensor]:
    input_sequences = torch.zeros(5, 100, 1)
    input_sequences[0, 10:75, :] = torch.sin(torch.linspace(0, 10, 65)).unsqueeze(-1)
    input_sequences[1:, 5:10, :] = 1
    input_sequences[2, 30:45, :] = -1
    input_sequences[3, 40:75, :] = -0.3
    input_sequences[:, 50:70, :] = 0.5
    input_sequences[4, 10:75, :] = -0.3

    input_sequences = input_sequences.repeat(1, 1, 3)
    # print(input_sequencesr.shape)

    # mdl_ground = prepare_ground_mdl(num_states=hidden_dim,
    #                                 num_outputs=output_dim,
    #                                 num_inputs=input_dim)
    mdl_ground = hmm.benchmark_mdl()
    states_sequences, output_sequences = mdl_ground.simulate(input_sequences)
    output_seqs_noisy = output_sequences + \
        0.05 * torch.randn_like(output_sequences).float()

    # fig, axes = plt.subplots(2, 1, figsize=(16, 5))
    # for input_sequence in input_sequences:
    #     axes[0].plot(input_sequence.detach().squeeze())
    # axes[0].set_title(f'Internal States')
    # axes[0].set_title(f'Inputs')
    # for output_seq_noisy in output_seqs_noisy:
    #     axes[1].plot(output_seq_noisy.detach().squeeze())
    # axes[1].set_title(f'Outputs')
    # plt.tight_layout()
    # plt.show()

    mem_y = 2
    mem_u = 2

    o_seq = output_sequences.squeeze(-1)[0]
    i_seq = input_sequences[0]
    len_ini = i_seq.size(0)
    o_seq_fold = o_seq.unfold(0, mem_y, 1).reshape(len_ini-mem_y+1, -1)
    i_seq_fold = i_seq.unfold(0, mem_u, 1).reshape(len_ini-mem_u+1, -1)

    smallest_length = min(i_seq_fold.size(0), i_seq_fold.size(0))
    regressor = torch.cat([o_seq_fold[:smallest_length],
                           i_seq_fold[:smallest_length]], dim=1)
    regressor_trimmed = regressor[:-mem_y]
    out_regr_aligned = o_seq[mem_y:-1]

    # return input_sequences[0], output_sequences.squeeze(-1)[0], output_seqs_noisy.squeeze(-1)[0]
    return regressor_trimmed, out_regr_aligned

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')

    input_seqs, output_seqs = prepare_data()


    num_inducing = 64
    num_latents = 2
    learn_inducing_locations = True,

    learning_rate = 0.01
    max_epochs = 3000
    patience = 30
    minibatch_size = 1
    hidden_dim = 3
    num_particles = 50
    min_delta = 0

    input_dim = input_seqs.size(-1)
    obs_dim = output_seqs.size(-1)
    length = input_seqs.size(-2)

    model = GPNARXpl(
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        obs_dim=obs_dim,
        num_inducing=num_inducing,
        num_latents=num_latents,
        learn_inducing_locations=learn_inducing_locations,
        lr=learning_rate,
        num_particles=num_particles,
        minibatch_size=minibatch_size
    )

    loader = DataLoader(
        TensorDataset(
            input_seqs,
            output_seqs
        ),
        batch_size=256,
        shuffle=True
    )
    fit(model,
        train_dataloader=loader,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        verbose=False,
        enable_logger=True,
        enable_checkpointing=False)

    param_store = pyro.get_param_store()
    for key, val in param_store.items():
        print(f'{key}:\n\t{val}')

    # fig, axes = plt.subplots(2, 1, figsize=(16, 5))
    # axes[0].plot(x.detach()[0].squeeze())
    # axes[0].set_title(f'Internal States')
    # for i in range(y.shape[0]):
    #     axes[1].plot(y.detach()[i].squeeze(), label=f'Estimated {i}')
    #     axes[1].plot(output_seqs_noisy.detach()[
    #                  i].squeeze(), label=f'Noisy Data {i}')
    # axes[1].legend()
    # axes[1].set_title(f'Outputs')
    # plt.tight_layout()
    # plt.show()
    print('Main is done')


    print(f'Done')
