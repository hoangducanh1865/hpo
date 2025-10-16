import torch
from tqdm import tqdm
from src.utils import accuracy
from src.config import Config


class Trainer:
    def __init__(self, model, train_loader, test_loader, lr=Config.lr, num_epochs=Config.num_epochs):
        self.model = model.to(Config.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.device = Config.device

    def fit(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0

            for X, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                X, y = X.to(self.device), y.to(self.device)

                y_hat = self.model(X)
                loss = self.model.loss(y_hat, y)

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                total_loss += loss.sum().item()
                total_correct += accuracy(y_hat, y).item()
                total_samples += y.size(0)

            train_acc = total_correct / total_samples
            test_acc = self.evaluate()
            avg_loss = total_loss / total_samples

            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    def evaluate(self):
        self.model.eval()
        total_correct, total_samples = 0, 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                total_correct += accuracy(y_hat, y).item()
                total_samples += y.size(0)

        return total_correct / total_samples
