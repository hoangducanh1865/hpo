import torch
import torch.nn.functional as F
from torch import nn
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.svm import SVC as SklearnSVC


class SoftmaxRegression(nn.Module):
    """Softmax Regression (Multinomial Logistic Regression) model."""

    def __init__(self, num_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, num_outputs)  # 28x28 → 784 input features
        )

    def forward(self, X):
        return self.net(X)

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(Y_hat, Y, reduction="mean" if averaged else "none")


# @QUESTION
def init_cnn(module):
    """Xavier initialization for CNN weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

    def forward(self, X):
        return self.net(X)

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(Y_hat, Y, reduction="mean" if averaged else "none")

    def apply_init(self):
        self.apply(init_cnn)


class SGDClassifier(nn.Module):
    """
    SGD Linear Classifier with Elastic Net regularization.
    Mimics sklearn's SGDClassifier with loss='hinge' (linear SVM).
    """
    def __init__(self, num_classes=10, alpha=0.0001, l1_ratio=0.15):
        """
        Args:
            num_classes: Number of output classes
            alpha: Regularization constant (multiplies the regularization term)
            l1_ratio: Elastic net mixing parameter (0 <= l1_ratio <= 1)
                     l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1
        """
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, num_classes)  # 28x28 → 784 input features
        )
    
    def forward(self, X):
        return self.net(X)
    
    def loss(self, Y_hat, Y, averaged=True):
        """
        Hinge loss (linear SVM) + Elastic Net regularization
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        
        # Multi-class hinge loss
        hinge_loss = F.multi_margin_loss(Y_hat, Y, reduction="mean" if averaged else "none")
        
        # Elastic Net regularization: alpha * (l1_ratio * L1 + (1 - l1_ratio) * L2)
        linear_layer = self.net[1]
        l1_reg = torch.sum(torch.abs(linear_layer.weight))
        l2_reg = torch.sum(linear_layer.weight ** 2)
        elastic_net = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * 0.5 * l2_reg)
        
        return hinge_loss + elastic_net


class SVMClassifier(nn.Module):
    """
    SVM Classifier using RBF kernel approximation.
    Uses Random Fourier Features to approximate the RBF kernel.
    """
    def __init__(self, num_classes=10, C=1.0, gamma=0.001, n_components=100):
        """
        Args:
            num_classes: Number of output classes
            C: Regularization parameter (inverse of alpha, larger C = less regularization)
            gamma: Kernel coefficient for RBF kernel
            n_components: Number of random features for kernel approximation
        """
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.n_components = n_components
        
        # Random Fourier Features for RBF kernel approximation
        self.random_weights = nn.Parameter(
            torch.randn(784, n_components) * torch.sqrt(torch.tensor(2 * gamma)),
            requires_grad=False
        )
        self.random_offset = nn.Parameter(
            torch.rand(n_components) * 2 * torch.pi,
            requires_grad=False
        )
        
        # Linear classifier on top of kernel features
        self.classifier = nn.Linear(n_components, num_classes)
        
    def forward(self, X):
        X = X.view(X.size(0), -1)  # Flatten
        
        # Random Fourier Features: cos(Xw + b)
        projection = torch.matmul(X, self.random_weights) + self.random_offset
        features = torch.cos(projection) * torch.sqrt(torch.tensor(2.0 / self.n_components))
        
        return self.classifier(features)
    
    def loss(self, Y_hat, Y, averaged=True):
        """
        Hinge loss + L2 regularization (inversely controlled by C)
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        
        # Multi-class hinge loss
        hinge_loss = F.multi_margin_loss(Y_hat, Y, reduction="mean" if averaged else "none")
        
        # L2 regularization (alpha = 1 / C)
        alpha = 1.0 / self.C
        l2_reg = alpha * 0.5 * torch.sum(self.classifier.weight ** 2)
        
        return hinge_loss + l2_reg