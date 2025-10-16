import torch
from torch import nn
import torch.nn.functional as F

class SoftmaxRegression(nn.Module):
    """Softmax Regression (Multinomial Logistic Regression) model."""

    def __init__(self, num_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, num_outputs)  # 28x28 → 784 input features
        )

    def forward(self, X):
        return self.net(X)

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none'
        )
