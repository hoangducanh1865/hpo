from src.model import SoftmaxRegression
from src.utils import load_fashion_mnist
from src.trainer import Trainer
from src.config import Config


def main():
    # Load data
    train_loader, test_loader = load_fashion_mnist(Config.batch_size)

    # Initialize model
    model = SoftmaxRegression(num_outputs=Config.num_outputs)

    # Train model
    trainer = Trainer(model, train_loader, test_loader, lr=Config.lr, num_epochs=Config.num_epochs)
    trainer.fit()


if __name__ == "__main__":
    main()
