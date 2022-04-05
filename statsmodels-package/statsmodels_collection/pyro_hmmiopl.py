from typing import Tuple
import pyro
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.functional import Tensor
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from ss_deterministic_model import StateSpaceModel

import pyro_hmmio as hmm


class HMMIOpl(LightningModule):

    # svi: SVI

    def __init__(self,
                 hidden_dim: int,
                 input_dim: int,
                 obs_dim: int,
                 lr=0.1,
                 num_particles=10,
                 minibatch_size=512
                 ):
        super().__init__()
        self.automatic_optimization = False

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
        self.svi = SVI(hmm.hmm_model, hmm.hmm_guide, optimizer, elbo)
        self.train()
        return None
        
    def training_step(self, batch, batch_idx, *args, **kwargs):
        tr_inputs, tr_y = batch
        # self.zero_grad()
        loss_raw = self.svi.step(tr_y,
                                 tr_inputs,
                                 hidden_dim=self.hidden_dim,
                                 input_dim=self.input_dim,
                                 obs_dim=self.obs_dim,
                                 num_sequences=tr_inputs.size(0),
                                 length=tr_inputs.size(1),
                                 batch_size=self.minibatch_size)
        loss_tensor = torch.as_tensor(loss_raw)
        # Output from model
        self.log('train_loss', loss_tensor, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss_tensor, 'log': {'train_loss': loss_tensor}}


def fit(module,
        train_dataloader,
        max_epochs=1000,
        patience=10,
        min_delta=0,
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
        callbacks += [checkpoint_callback]

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


def prepare_data() -> Tuple[Tensor, Tensor, Tensor]:
    input_sequences = torch.zeros(5, 100, 1)
    input_sequences[1:, 5:10, :] = 1
    input_sequences[2, 30:45, :] = -1
    input_sequences[3, 40:75, :] = -0.3
    input_sequences[:, 50:70, :] = 0.5
    input_sequences[4, 10:75, :] = -0.3
    input_sequences[0, 10:75, :] = torch.sin(torch.linspace(0, 10, 65)).unsqueeze(-1)

    input_sequences = input_sequences.repeat(1, 1, 3).double()
    # print(input_sequencesr.shape)

    # mdl_ground = prepare_ground_mdl(num_states=hidden_dim,
    #                                 num_outputs=output_dim,
    #                                 num_inputs=input_dim)
    mdl_ground = hmm.benchmark_mdl()
    states_sequences, output_sequences = mdl_ground.simulate(input_sequences)
    output_seqs_noisy = output_sequences + \
        0.05 * torch.randn_like(output_sequences)
    fig, axes = plt.subplots(2, 1, figsize=(16, 5))
    for input_sequence in input_sequences:
        axes[0].plot(input_sequence.detach().squeeze())
    axes[0].set_title(f'Internal States')
    axes[0].set_title(f'Inputs')
    for output_seq_noisy in output_seqs_noisy:
        axes[1].plot(output_seq_noisy.detach().squeeze())
    axes[1].set_title(f'Outputs')
    plt.tight_layout()
    plt.show()
    return input_sequences, output_sequences, output_seqs_noisy

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')

    input_seqs, output_seqs, output_seqs_noisy = prepare_data()

    learning_rate = 0.01
    max_epochs = 3000
    patience = 30
    minibatch_size = 5
    hidden_dim = 3
    num_particles = 50
    min_delta = 0

    input_dim = input_seqs.size(-1)
    obs_dim = output_seqs.size(-2)
    num_sequences = input_seqs.size(0)
    length = input_seqs.size(1)

    model = HMMIOpl(
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        obs_dim=obs_dim,
        lr=learning_rate,
        num_particles=num_particles,
        minibatch_size=minibatch_size
    )

    loader = DataLoader(
        TensorDataset(
            input_seqs,
            output_seqs_noisy
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

    a_params_dict = {
        'a_mean': pyro.param('a_mean'),
        'a_offdiag1': pyro.param('a_offdiag1'),
    }
    a = hmm.form_a(a_params_dict)

    mdl_reconstructed = StateSpaceModel(a,
                                        pyro.param('b'),
                                        pyro.param('c'))
    x, y = mdl_reconstructed.simulate(input_seqs, x0=pyro.param('c').unsqueeze(-1))
    fig, axes = plt.subplots(2, 1, figsize=(16, 5))
    axes[0].plot(x.detach()[0].squeeze())
    axes[0].set_title(f'Internal States')
    for i in range(y.shape[0]):
        axes[1].plot(y.detach()[i].squeeze(), label=f'Estimated {i}')
        axes[1].plot(output_seqs_noisy.detach()[i].squeeze(), label=f'Noisy Data {i}')
    axes[1].legend()
    axes[1].set_title(f'Outputs')
    plt.tight_layout()
    plt.show()
    print('Main is done')


    print(f'Done')
