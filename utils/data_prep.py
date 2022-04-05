import torch


def make_gpnarx_regressor(u: torch.Tensor,
                            y: torch.Tensor,
                            y_lag: int = 1,
                            y_pred_horizont: int = 1,
                            u_lag: int = 1):
    """Reshapes inputs and outputs into aggregate of lagged inputs and outputs
    and correspondingly aligned output. Sequences are trimmed so that they have
    the same length.

    Args:
        u (torch.Tensor): input sequence of size (N_u, dim_u).
        y (torch.Tensor): output sequence of size (N_y, dim_y).
        y_lag (int, optional): Lag on outputs. Defaults to 1.
        y_pred_horizont (int, optional): number of sample after y_t that are
            to be predicted. Should be >= 1. Defaults to 1.
        u_lag (int, optional): Lag on inputs. Should be `u_lag >= y_lag`.
            Defaults to 1.

    Returns:
        Tuple[Tensor, Tensor]: `regressor`, `out_regr_aligned`: regressor are 
        the aggregated and aligned lagged inputs and outputs. `out_regr_aligned`
        are the left-trimmed outputs. `regressor[t, ...]` should be used to
        predict `out_regr_aligned[t, ...]` and that means:
        $y_{t+1} = f(y_{t-y_lag+1:t-y_pred_horizont+1}, u{t-u_lag+1:t})$.
    """
    u_len_ini = u.size(0)
    y_len_ini = y.size(0)
    u_dim = y.size(-1)
    y_dim = y.size(-1)
    y_seq_fold = y.unfold(0, y_lag-y_pred_horizont+1, 1).reshape(y.size(0)-(y_lag-y_pred_horizont+1)+1, -1)
    u_seq_fold = u.unfold(0, u_lag, 1).reshape(u.size(0)-u_lag+1, -1)

    dif_lag = u_lag - y_lag
    if y_lag > u_lag:
        u_left_trim = u_seq_fold[-dif_lag:]
        y_left_trim = y_seq_fold
        out_regr_aligned = y[y_lag:]
    else:
        u_left_trim = u_seq_fold
        y_left_trim = y_seq_fold[dif_lag:]
        out_regr_aligned = y[u_lag:]

    shorter_len = min(u_left_trim.size(0), y_left_trim.size(0))
    regressor = torch.cat([u_seq_fold[:shorter_len],
                            y_left_trim[:shorter_len, :]], dim=1)

    return regressor[:out_regr_aligned.size(0)], out_regr_aligned


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')

    time = torch.arange(20, dtype=torch.float)
    u = time.unsqueeze(-1) @ torch.tensor([1, 1, 1.]).unsqueeze(0)
    y = 10*u[:, :2]

    regressor, outs = make_gpnarx_regressor(u, y, y_lag=7, y_pred_horizont=3, u_lag=6)
    print(f'regressor: {regressor.shape}')
    print(f'outs: {outs.shape}')

    plt.plot(outs, label='outs aligned')

    plt.plot(regressor[:, -3:], label='outs lagged')
    plt.legend()
    plt.tight_layout()

    plt.show()

