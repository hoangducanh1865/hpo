from scipy import stats
from src.model import SoftmaxRegression,LeNet
from src.trainer import Trainer
from src.utils import Utils
from src.config import Config
class HPO:
    def __init__(self):
        pass
    
    @staticmethod
    def hpo_objective(args,config, num_epochs=None):
        if num_epochs is None:
            num_epochs = args.num_epochs
        lr = config['lr']
        
        # Load data
        train_loader, val_loader, test_loader = Utils.load_fashion_mnist(args.batch_size)
        model = HPO.build_model(args)
        trainer=Trainer(model,train_loader,val_loader,test_loader,lr=lr,num_epochs=num_epochs)
        trainer.fit()
        val_err=trainer.validation_error()
        return val_err
    
    @staticmethod
    def random_search(args,num_trials=10,search_space=None,num_epochs=None):
        if search_space is None:
            search_space={'lr':stats.loguniform(1e-4,1)}
        best_config=None
        best_score=float('inf')
        for i in range(num_trials):
            config={}
            for name,dist in search_space.items():
                config[name]=dist.rvs()
                print(f"Trial {i}: config: {config}")
                score=HPO.hpo_objective(args,config,num_epochs=num_epochs)
                print(f'validation_error = {score}')
                if score<best_score:
                    best_score=score
                    best_config=config
        print(f'Best config: {best_config}, with val_err = {best_score}')
        return best_config,best_score
    
    @staticmethod
    def build_model(args):
        if args.model_name=='softmax':
            return SoftmaxRegression(num_outputs=args.num_outputs)
        elif args.model_name=='lenet':
            model=LeNet(num_classes=args.num_outputs)
            model.apply_init()
            return model
        else:
            raise NotImplementedError('Model type not supported, use "softmax" or "lenet" instead')