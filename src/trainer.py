import torch
from tqdm import tqdm
from src.utils import Utils
from src.config import Config


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr, num_epochs):
        self.model = model.to(Config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.device = Config.device

    def fit(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            for X, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                X, y = X.to(self.device), y.to(self.device)

                y_hat = self.model(X)
                loss = self.model.loss(y_hat, y)

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

    def evaluate_train(self):
        self.model.eval()
        total_correct, total_samples = 0, 0

        with torch.no_grad():
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                total_correct += Utils.accuracy(y_hat, y).item()
                total_samples += y.size(0)

        return total_correct / total_samples
    
    def evaluate_test(self):
        self.model.eval()
        total_correct, total_samples = 0, 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                total_correct += Utils.accuracy(y_hat, y).item()
                total_samples += y.size(0)

        return total_correct / total_samples
    
    def evaluate_val(self):
        self.model.eval()
        total_correct, total_samples = 0, 0
       
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                total_correct += Utils.accuracy(y_hat, y).item()
                total_samples += y.size(0)
      
        return total_correct / total_samples

    def validation_error(self):
        return 1.0 - self.evaluate_val()
