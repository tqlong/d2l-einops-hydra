import numpy as np
from torch.utils.data import Dataset

## return (weights, bias)
def generate_true_weights(num_features, seed=42):
    """Generate true weights for a linear model."""
    rng = np.random.RandomState(seed)
    return rng.normal(scale=1, size=num_features).astype(np.float32), rng.normal(scale=1, size=1).astype(np.float32)[0]

def synthetic_linear_dataset(num_examples, num_features, true_weights, true_bias, noise_sigma=0.01):
    """Generate a synthetic dataset for linear regression."""
    X = np.random.normal(scale=1, size=(num_examples, num_features)).astype(np.float32)
    y = np.dot(X, true_weights) + true_bias + np.random.normal(scale=noise_sigma, size=num_examples).astype(np.float32)
    return X, y

class SyntheticLinearDataset(Dataset):
    """A synthetic dataset for linear regression."""

    def __init__(self, num_examples, num_features, true_weights_seed=42, noise_sigma=0.01):
        super().__init__()
        """Initialize a synthetic dataset for linear regression."""
        self.true_weights, self.true_bias = generate_true_weights(num_features, true_weights_seed)
        self.X, self.y = synthetic_linear_dataset(
            num_examples, num_features, self.true_weights, self.true_bias, noise_sigma
        )

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Return the idx-th example from the dataset."""
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    dataset = SyntheticLinearDataset(num_examples=1000, num_features=10)
    print(len(dataset))
    print(dataset[0])
    print(dataset.true_weights, dataset.true_bias)