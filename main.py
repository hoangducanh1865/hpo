from src.model import SoftmaxRegression,LeNet
from src.utils import Utils
from src.trainer import Trainer
from src.config import Config
from src.hpo import HPO
def build_model(args):
    if args.model_name=='softmax':
        return SoftmaxRegression(num_outputs=args.num_outputs)
    elif args.model_name=='lenet':
        model=LeNet(num_classes=args.num_outputs)
        model.apply_init()
        return model
    else:
        raise NotImplementedError('Model type not supported, use "softmax" or "lenet" instead')
def train_fixed(args):
    train_loader, val_loader, test_loader = Utils.load_fashion_mnist(args.batch_size)
    model = build_model(args)
    trainer = Trainer(model, train_loader, val_loader, test_loader, lr=Config.lr, num_epochs=Config.num_epochs)
    trainer.fit()
    train_acc=trainer.evaluate_train()
    val_acc=1.0-trainer.validation_error()
    test_acc=trainer.evaluate_test()
    print(f'Final train accuracy: {train_acc:.4f}')
    print(f'Final validation accuracy: {val_acc:.4f}')
    print(f'Final test accuracy: {test_acc:.4f}')
def train_with_hpo(args):
    best_config,best_score=HPO.random_search(args,num_trials=args.num_trials)
    print(f'Training with best config: {best_config}')
    train_loader,val_loader,test_loader=Utils.load_fashion_mnist(args.batch_size)
    model=build_model(args)
    trainer=Trainer(model,train_loader,val_loader,test_loader,lr=best_config['lr'],num_epochs=args.num_epochs)
    trainer.fit()
    train_acc = trainer.evaluate_train()
    val_acc = 1.0 - trainer.validation_error()
    test_acc = trainer.evaluate_test()
    print(f'Final train accuracy with tuned hyperparameters: {train_acc:.4f}')
    print(f'Final validation accuracy with tuned hyperparameters: {val_acc:.4f}')
    print(f'Final test accuracy with tuned hyperparameters: {test_acc:.4f}')
def main():
    parser=Config.new_parser()
    Config.add_training_argument(parser)
    args = parser.parse_args()
    if args.train_mode=='hpo':
        train_with_hpo(args)
    else:
        train_fixed(args)
if __name__ == "__main__":
    main()
