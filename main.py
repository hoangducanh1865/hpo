from src.model import SoftmaxRegression
from src.utils import Utils
from src.trainer import Trainer
from src.config import Config
from src.hpo import HPO


def train_fixed():
    train_loader, val_loader, test_loader = Utils.load_fashion_mnist(Config.batch_size)
    model = SoftmaxRegression(num_outputs=Config.num_outputs)
    trainer = Trainer(model, train_loader, val_loader, test_loader, lr=Config.lr, num_epochs=Config.num_epochs)
    trainer.fit()
    train_acc=trainer.evaluate_train()
    val_acc=1.0-trainer.validation_error()
    test_acc=trainer.evaluate_test()
    print(f'Final train accuracy: {train_acc:.4f}')
    print(f'Final validation accuracy: {val_acc:.4f}')
    print(f'Final test accuracy: {test_acc:.4f}')


def train_with_hpo():
    best_config,best_score=HPO.random_search(num_trials=20)
    print(f'Training with best config: {best_config}')
    train_loader,val_loader,test_loader=Utils.load_fashion_mnist(Config.batch_size)
    model=SoftmaxRegression(num_outputs=Config.num_outputs)
    trainer=Trainer(model,train_loader,val_loader,test_loader,lr=best_config['lr'],num_epochs=Config.num_epochs)
    trainer.fit()
    train_acc = trainer.evaluate_train()
    val_acc = 1.0 - trainer.validation_error()
    test_acc = trainer.evaluate_test()
    print(f'Final train accuracy with tuned hyperparameters: {train_acc:.4f}')
    print(f'Final validation accuracy with tuned hyperparameters: {val_acc:.4f}')
    print(f'Final test accuracy with tuned hyperparameters: {test_acc:.4f}')
    
    
def main():
    mode=input('Training mode (fixed/hpo): ')
    if mode=='hpo':
        train_with_hpo()
    else:
        train_fixed()


if __name__ == "__main__":
    main()
