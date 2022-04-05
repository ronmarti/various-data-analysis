from typing import Tuple
import torch
from torch._C import dtype
from torch.functional import Tensor


class StateSpaceModel():
    """LTI IO model.
    """    
    def __init__(self, a, b, c):
        self.a = a.float()
        self.b = b.float()
        self.c = c.float()

    @property
    def num_states(self) -> int:
        return self.a.size(-2)

    @property
    def num_obs(self) -> int:
        return self.c.size(-2)

    @property
    def num_inputs(self) -> int:
        return self.b.size(-1)

    def simulate(self,
                 input_seq: Tensor,
                 x0: Tensor = None):
        """Simulates response of the system to the input signal.

        Args:
            input (Tensor): `(batch, N, num_inputs)` input signal,
            where N is the number of simulated time-steps.
            x0 (Tensor, optional): Initial condition on internal (hidden) state,
            vector of size `(batch, num_states)`
            or `(num_states)` (init by cloning). Defaults to zero vector (when None).

        Returns:
            (Tensor, Tensor): x, y - simulated states and observations sequences.
        """
        batch_size = 1
        if input_seq.dim() > 2:
            batch_size = input_seq.size(0)

        if x0 is None:
            x0 = torch.zeros(batch_size, self.num_states, 1)
        x0 = x0.double()
        N = input_seq.size(-2)
        x = torch.zeros(batch_size, N, self.num_states, 1, dtype=torch.float)
        inseq_weighted = (self.b @ input_seq.unsqueeze(-1)).squeeze(-1)
        x[:, 0, :] = x0
        for t in torch.arange(1, N):
            x[:, t, ...] = (self.a @ x[:, t-1, ...]) + inseq_weighted[:, t-1, :].unsqueeze(-1)
        y = self.c @ x
        return x, y


def benchmark_mdl():
    a = torch.tensor([[0.7, 0, 0],
                      [0.4, 0.8, 0],
                      [-0.2, -0.3, 0.8]])
    b = torch.tensor([[1,],
                    [0],
                    [0.]])
    c = torch.tensor([[1., 1, 1]])
    return StateSpaceModel(a, b, c)

def benchmark_data() -> Tuple[Tensor, Tensor, Tensor]:
    """Generates training dataset of inputs, benchmark model
    and its response and internal states evolution.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            `(b, T, k_u)` inputs,
            `(b, T, k_x)` states,
            `(b, T, k_y)` outputs
    """    
    input_sequences = torch.zeros(10, 125, 1)
    input_sequences[:, 5:50, :] = 1
    input_sequences[:, 51:125, :] = -1
    model = benchmark_mdl()
    x, y = model.simulate(input_sequences)
    return input_sequences, x, y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    input_sequences, states_sequences, output_sequences = benchmark_data()
    output_seqs_noisy = output_sequences + \
        0.05 * torch.randn_like(output_sequences)
    fig, axes = plt.subplots(2, 1, figsize=(16, 5))
    axes[0].plot(states_sequences.detach()[0].squeeze())
    axes[0].set_title(f'Internal States')
    axes[1].plot(output_sequences.detach()[0].squeeze())
    axes[1].plot(output_seqs_noisy.detach()[0].squeeze())
    axes[1].set_title(f'Outputs')
    plt.show()