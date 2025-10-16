from scipy import stats
from src.model import SoftmaxRegression
from src.trainer import Trainer
from src.utils import Utils
from src.config import Config


class HPO:
    def __init__(self):
        pass
    
    @staticmethod
    def hpo_objective(config, num_epochs=None):
        if num_epochs is None:
            num_epochs = Config.num_epochs
        lr = config['lr']
        
        # Load data
        train_loader, val_loader, test_loader = Utils.load_fashion_mnist(Config.batch_size)
        model = SoftmaxRegression(num_outputs=Config.num_outputs)
        trainer=Trainer(model,train_loader,val_loader,test_loader,lr=lr,num_epochs=num_epochs)
        trainer.fit()
        val_err=trainer.validation_error()
        return val_err
    
    @staticmethod
    def random_search(num_trials=10,search_space=None,num_epochs=None):
        if search_space is None:
            search_space={'lr':stats.loguniform(1e-4,1)}
        best_config=None
        best_score=float('inf')
        for i in range(num_trials):
            config={}
            for name,dist in search_space.items():
                config[name]=dist.rvs()
                print(f"Trial {i}: config: {config}")
                score=HPO.hpo_objective(config,num_epochs=num_epochs)
                print(f'validation_error = {score}')
                if score<best_score:
                    best_score=score
                    best_config=config
        print(f'Best config: {best_config}, with val_err = {best_score}')
        return best_config,best_score