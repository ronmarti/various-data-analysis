import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy, VariationalStrategy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class GPClassificationModel(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        # if not isinstance(inducing_points, torch.Tensor):
        #     inducing_points = torch.as_tensor(inducing_points, dtype=torch.float32)

        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def cast_data(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        return data

    def fit(self, dset: TensorDataset,
            epochs: int = 100,
            batch_size: int = 100) -> dict:
        
        train_loader = DataLoader(dset, batch_size=batch_size, shuffle=True)

        self.train()
        self.likelihood.train()

        params = list(self.parameters()) + list(self.likelihood.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01)

        # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
        # mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=y.size(0))
        mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self, num_data=y.size(0), beta=0.5)

        # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
        # effective for VI.
        epochs_iter = tqdm(range(epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = self(x_batch)
                loss = -mll(output, y_batch)
                epochs_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()

    def predict_prob(self, x):
        x = self.cast_data(x)
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            pred_lik = self.likelihood(self(x))
        return pred_lik.mean

    def predict(self, x):
        x = self.cast_data(x)
        prob = self.predict_prob(x)
        return prob > 0.5


if __name__ == "__main__":
    import utils
    from pathlib import Path

    root = Path('.out')
    root.mkdir(parents=True, exist_ok=True)

    N_mesh = 100  # single dim size of the mesh
    N_ind_pts = 3  # num of N inducing points on a single dim

    data_generator = utils.FakeDataGenerator()
    X, y = data_generator.generate_fake_training_dset(N_mesh)
    mesh = torch.meshgrid(
        torch.linspace(-torch.pi, torch.pi, N_ind_pts),
        torch.linspace(-torch.pi, torch.pi, N_ind_pts),
        torch.linspace(-torch.pi, torch.pi, N_ind_pts)
    )
    inducing_pts = torch.stack(
        [i.flatten() for i in mesh],
        dim=1
    )
    train_dataset = TensorDataset(X, y)
    model = GPClassificationModel(inducing_pts)
    model.fit(train_dataset, 100, 1000)

    X_test, y_test = data_generator.generate_observation(class_id=0, N=3)
    probs = model.predict_prob(X_test)
    pred_classes = model.predict(X_test)
    print(f'Test classification:\nX_test: {X_test}\ny_predicted: {pred_classes}\nprobs: {probs}')

    X_test, y_test = data_generator.generate_observation(class_id=1, N=3)
    probs = model.predict_prob(X_test)
    pred_classes = model.predict(X_test)
    print(f'Test classification:\nX_test: {X_test}\ny_predicted: {pred_classes}\nprobs: {probs}')

    torch.save(model.state_dict(), '.out/model_state.pth')

    state_dict = torch.load('model_state.pth')
    model = GPClassificationModel(inducing_pts)  # Create a new GP model
    model.load_state_dict(state_dict)
else:
    print('Importing module test.py')
