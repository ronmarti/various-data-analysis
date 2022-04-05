from pyro.infer.trace_elbo import JitTrace_ELBO
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.optim import Adam
from pyro.ops.indexing import Vindex

from tqdm import trange

from ss_deterministic_model import StateSpaceModel


def hmm_model(output_seqs: Tensor,
              intput_seqs: Tensor,
              hidden_dim: int,
              input_dim: int,
              obs_dim: int,
              num_sequences: int,
              length: int,
              batch_size: int=None):
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
    # num_sequences = output_seqs.size(0)
    # length = output_seqs.size(1)
    # obs_dim = output_seqs.size(2)
    # length = output_seqs.size(1)
    # obs_dim = output_seqs.size(2)

    # mu_0 = pyro.param('mu_0', init_tensor=torch.zeros(2*hidden_dim))
    # cov_0 = pyro.param('cov_0',
    #                    init_tensor=torch.diag(0.01 * torch.ones(2*hidden_dim)),
    #                    constraint=constraints.lower_cholesky)

    q_diag = pyro.param('q',
                   init_tensor=0.01 * torch.ones(2*hidden_dim,dtype=torch.double),
                   constraint=constraints._Interval(0., 1.01))
    q = torch.diag(q_diag)
    r = pyro.param('r',
                   init_tensor=0.01 * torch.eye(obs_dim,dtype=torch.double),
                   constraint=constraints.lower_cholesky)
    a_mean = pyro.param('a_mean',
                    init_tensor=0.1 * torch.randn(hidden_dim,dtype=torch.double),
                    constraint=constraints._Interval(-1., 1.))
    a_offdiag1 = pyro.param('a_offdiag1',
                    init_tensor=0.1 * torch.randn(hidden_dim,dtype=torch.double),
                    constraint=constraints._Interval(0., 1.))

    a_params_dict = {
        'a_mean': a_mean,
        'a_offdiag1': a_offdiag1
    }
    a = form_a(a_params_dict)

    b = pyro.param('b',
                   init_tensor= torch.randn(2*hidden_dim, input_dim,dtype=torch.double))
    c = pyro.param('c',
                   init_tensor= torch.rand(obs_dim, 2*hidden_dim,dtype=torch.double))
    c_normalized = c
    weighted_ins = (b @ intput_seqs.unsqueeze(-1)).squeeze(-1)

    mu_0 = pyro.param('mu_0', init_tensor=torch.zeros(num_sequences, 2*hidden_dim, dtype=torch.double))
    with pyro.plate("sequences", num_sequences, subsample_size = batch_size) as batch:
        # x_t = pyro.sample(
        #     'x_0_0',
        #     dist.MultivariateNormal(mu_0, scale_tril=cov_0)
        # )
        # x_t = mu_0
        x_t = mu_0[batch]
        for t in pyro.markov(range(length), history=1):
            mu_y_t = (c_normalized @ x_t.unsqueeze(-1)).squeeze(-1)
            pyro.sample(
                f"y_{t}",
                dist.MultivariateNormal(mu_y_t, scale_tril=r),
                obs=Vindex(output_seqs)[batch, t, :, 0],
            )
            mu_x_tp1 = (a.unsqueeze(-3) @ x_t.unsqueeze(-1)).squeeze(-1) + \
                weighted_ins[batch, t, :]
            # x_t is x_tp1 at this instance, after iteration it will become x_t
            x_t = pyro.sample(
                f"x_{t}",
                dist.MultivariateNormal(mu_x_tp1, scale_tril=q),
            )


def hmm_guide(output_seqs: Tensor,
              intput_seqs: Tensor,
              hidden_dim: int,
              input_dim: int,
              obs_dim: int,  # unused - retained only due to the model signature
              num_sequences: int,
              length: int,
              batch_size=None):
    # num_sequences = output_seqs.size(0)
    # length = output_seqs.size(1)

    # mu_0 = pyro.param('mu_0', init_tensor=torch.zeros(2*hidden_dim))
    # cov_0 = pyro.param('cov_0',
    #                    init_tensor=torch.diag(0.01 * torch.ones(2*hidden_dim)),
    #                    constraint=constraints.lower_cholesky)

    q_diag = pyro.param('q',
                   init_tensor=0.01 * torch.ones(2*hidden_dim,dtype=torch.double),
                   constraint=constraints._Interval(0., 1.01))
    q = torch.diag(q_diag)
    a_mean = pyro.param('a_mean',
                    init_tensor=0.1 * torch.randn(hidden_dim,dtype=torch.double),
                    constraint=constraints._Interval(-1., 1.))
    a_offdiag1 = pyro.param('a_offdiag1',
                    init_tensor=0.1 * torch.randn(hidden_dim,dtype=torch.double),
                    constraint=constraints._Interval(0., 1.))

    a_params_dict = {
        'a_mean': a_mean,
        'a_offdiag1': a_offdiag1
    }
    a = form_a(a_params_dict)

    b = pyro.param('b',
                   init_tensor= torch.randn(2*hidden_dim, input_dim,dtype=torch.double))
    weighted_ins = (b @ intput_seqs.unsqueeze(-1)).squeeze(-1)

    mu_0 = pyro.param('mu_0', init_tensor=torch.zeros(num_sequences, 2*hidden_dim, dtype=torch.double))
    with pyro.plate("sequences", num_sequences, subsample_size = batch_size) as batch:
        # x_t = pyro.sample(
        #     'x_0_0',
        #     dist.MultivariateNormal(mu_0, scale_tril=cov_0)
        # )
        # x_t = mu_0
        x_t = mu_0[batch]
        for t in pyro.markov(range(length), history=1):
            mu_x_tp1 = (a.unsqueeze(-3) @ x_t.unsqueeze(-1)).squeeze(-1) + \
                weighted_ins[batch, t, :]
            # x_t is x_tp1 at this instance, after iteration it will become x_t
            x_t = pyro.sample(
                f"x_{t}",
                dist.MultivariateNormal(mu_x_tp1, scale_tril=q),
            )

def form_a(kwargs: dict) -> Tensor:
    """Generates block-diagonal real matrix. Results in almos modal form
    of state-space model realization.

    Args:
        kwargs (dict): Expects to contain `a_mean` tensor of size `num_dims`,
            `a_offdiag1` of size `num_dims`.

    Returns:
        Tensor: block-diagonal matrix A of `2*num_dims` size.
    """    
    a_mean = kwargs['a_mean']
    a_offdiag1 = kwargs['a_offdiag1']
    a_diag = torch.diag(a_mean.repeat_interleave(2))
    a_offdiag = torch.stack((torch.zeros_like(a_offdiag1,dtype=torch.double), a_offdiag1), dim=1).flatten()[1:]
    a = a_diag + torch.diagflat(a_offdiag, -1) - torch.diagflat(a_offdiag, 1)
    return a


def main(args):

    learning_rate = args['learning_rate']
    learning_steps = args['learning_steps']
    batch_size = args['batch_size']
    hidden_dim = args['hidden_dim']
    input_dim = args['input_dim']
    output_dim = args['output_dim']
    num_particles = args['num_particles']

    input_sequences = torch.zeros(5, 100, 1)
    input_sequences[1:, 5:10, :] = 1
    input_sequences[2, 30:45, :] = -1
    input_sequences[3, 40:75, :] = -0.3
    input_sequences[:, 50:70, :] = 0.5
    input_sequences[4, 10:75, :] = -0.3
    input_sequences[0, 10:75, :] = torch.sin(torch.linspace(0, 10, 65)).unsqueeze(-1)

    input_sequences = input_sequences.repeat(1, 1, 3).double()

    # mdl_ground = prepare_ground_mdl(num_states=hidden_dim,
    #                                 num_outputs=output_dim,
    #                                 num_inputs=input_dim)
    mdl_ground = benchmark_mdl()
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

    # hidden_dim = 2  # states_sequences.size(-2)
    input_dim = input_sequences.size(-1)
    output_dim = output_sequences.size(-2)
    num_sequences = input_sequences.size(0)
    length = input_sequences.size(1)

    # pyro.set_rng_seed(13)
    pyro.clear_param_store()

    # guide = AutoMultivariateNormal(hmm_model)
    # guide = AutoDiagonalNormal(hmm_model)
    guide = hmm_guide

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    optim = Adam({"lr": learning_rate})

    elbo = Trace_ELBO(num_particles=num_particles,
                      vectorize_particles=True,
                      strict_enumeration_warning=True)
    svi = SVI(hmm_model, guide, optim, elbo)

    progressbar = trange(learning_steps)
    # We'll train on small minibatches.
    for step in progressbar:
        loss = svi.step(output_seqs_noisy, input_sequences,  # tensors as args
                        hidden_dim=hidden_dim,  # non-tensors must be passed as **kwargs
                        input_dim=input_dim,
                        obs_dim=output_dim,
                        num_sequences=num_sequences,
                        length=length,
                        batch_size=batch_size)
        progressbar.set_postfix({'loss': loss})

    param_store = pyro.get_param_store()
    for key, val in param_store.items():
        print(f'{key}:\n\t{val}')

    # mu_auto = param_store['AutoDiagonalNormal.loc']

    a_params_dict = {
        'a_mean': pyro.param('a_mean'),
        'a_offdiag1': pyro.param('a_offdiag1'),
    }
    a = form_a(a_params_dict)
    c_raw = pyro.param('c')
    # c = c_raw/torch.linalg.norm(c_raw)
    c = c_raw

    mdl_reconstructed = StateSpaceModel(a,
                                        pyro.param('b'),
                                        c)
    x, y = mdl_reconstructed.simulate(input_sequences, x0=pyro.param('mu_0').unsqueeze(-1))
    # x, y = mdl_reconstructed.simulate(input_sequences)
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


def prepare_ground_mdl(num_states,
                       num_outputs,
                       num_inputs):
    a = torch.diag(torch.rand(num_states))
    a += 0.1*torch.randn_like(a).tril(-1)
    b = 1 * torch.randn(num_states, num_inputs)
    c = 1.00 * torch.randn(num_outputs, num_states)
    return StateSpaceModel(a, b, c)


def benchmark_mdl():
    # a = torch.tensor([[0.5, 0, 0],
    #                   [0.1, 0.2, 0],
    #                   [-0.2, 0.1, 0.8]])

    a_params_dict = {
        'a_mean': torch.tensor([0.77, -0.1, 0.7]),
        'a_offdiag1': torch.tensor([0.3, 0., 0.2])
    }
    a = form_a(a_params_dict)
    b = torch.tensor([[1., 1., 0.5],
                    [1.  , 1., 0.5],
                    [0.2 , 1., 0.5],
                    [0.4 , 1., 0.5],
                    [1.  , 1., 0.5],
                    [0.1 , 1., 0.5]])
    c = torch.ones(1, 6)
    return StateSpaceModel(a, b, c)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    assert pyro.__version__.startswith("1.8.0")
    # parser = argparse.ArgumentParser(
    #     description="MAP Baum-Welch learning Bach Chorales"
    # )
    
    # parser.add_argument("-n", "--num-steps", default=50, type=int)
    # parser.add_argument("-b", "--batch-size", default=8, type=int)

    # args = parser.parse_args()
    args = {
        'learning_rate': 0.01,
        'learning_steps': 2,
        'batch_size': 3,
        'hidden_dim': 3,
        'input_dim': 1,
        'output_dim': 1,
        'num_particles': 50,
    }

    
    
    main(args)
    print('Done')
    
