import time 
import torch
from scipy import stats
from src.model import SoftmaxRegression,LeNet
from src.trainer import Trainer
from src.utils import Utils
from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
class HPO:
    def __init__(self):
        pass
    
    @staticmethod
    # def hpo_objective(args,config, num_epochs=None):
    #     if num_epochs is None:
    #         num_epochs = args.num_epochs
    #     lr = config['lr']
        
    #     # Load data
    #     train_loader, val_loader, test_loader = Utils.load_fashion_mnist(args.batch_size)
    #     model = HPO.build_model(args)
    #     trainer=Trainer(model,train_loader,val_loader,test_loader,lr=lr,num_epochs=num_epochs)
    #     trainer.fit()
    #     val_err=trainer.validation_error()
    #     return val_err
    def hpo_objective_fn(args):
        def hpo_objective(config):
            lr=config.get('learning_rate',args.learning_rate)
            batch_size=config.get('batch_size',args.batch_size)
            train_loader,val_loader,test_loader=Utils.load_fashion_mnist(batch_size)
            model=Utils.build_model(args)
            trainer=Trainer(model,train_loader,val_loader,test_loader,lr=lr,num_epochs=args.num_epochs)
            trainer.fit()
            val_err=trainer.validation_error()
            return val_err
        return hpo_objective
            
    @staticmethod
    # def random_search(args,num_trials=10,search_space=None,num_epochs=None):
    #     if search_space is None:
    #         search_space={'lr':stats.loguniform(1e-4,1)}
    #     best_config=None
    #     best_score=float('inf')
    #     for i in range(num_trials):
    #         config={}
    #         for name,dist in search_space.items():
    #             config[name]=dist.rvs()
    #             print(f"Trial {i}: config: {config}")
    #             score=HPO.hpo_objective(args,config,num_epochs=num_epochs)
    #             print(f'validation_error = {score}')
    #             if score<best_score:
    #                 best_score=score
    #                 best_config=config
    #     print(f'Best config: {best_config}, with val_err = {best_score}')
    #     return best_config,best_score
    def random_search(args,config_space=None,initial_config=None):
        searcher=RandomSearcher(config_space,initial_config)
        scheduler=BasicScheduler(searcher)
        objective_fn=HPO.hpo_objective_fn(args)
        tuner=HPOTuner(scheduler=scheduler,objective_fn=objective_fn)
        # Run HPO
        print(f'Starting HPO with {args.num_trials} trials')
        tuner.run(number_of_trials=args.num_trials)
        best_config,best_score=tuner.get_best_config()
        # Summary results
        print('\n'+'='*32)
        print(f"HPO Summary:")
        print(f"Best config: {best_config}")
        print(f"Best validation error: {best_score:.4f}")
        print(f"Total runtime: {tuner.current_runtime:.2f}s")
        print(f"Average time per trial: {tuner.current_runtime/args.num_trials:.2f}s")
        return best_config,best_score,tuner # @QUESTION: why return tuner?
    @staticmethod
    def plot_hpo_progress(tuner, save_path=None):
        """Plot HPO progress over time"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot incumbent error vs trials
        ax1.plot(range(len(tuner.incumbent_trajectory)), tuner.incumbent_trajectory)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Best Validation Error')
        ax1.set_title('HPO Progress: Best Error vs Trials')
        ax1.grid(True)
        
        # Plot incumbent error vs cumulative runtime
        ax2.plot(tuner.cumulative_runtime, tuner.incumbent_trajectory)
        ax2.set_xlabel('Cumulative Runtime (s)')
        ax2.set_ylabel('Best Validation Error')
        ax2.set_title('HPO Progress: Best Error vs Time')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
class HPOSeacher(ABC):
    @abstractmethod
    def sample_config(self)->dict:
        '''Sample new hyperparameter configuration'''
        pass 
    def update(self,config:dict,error:float,additional_info=None):
        '''Update searcher state after trial completion'''
        pass 
class HPOScheduler(ABC):
    @abstractmethod
    def suggest(self)->dict:
        '''Suggest next configuration to evaluate'''
        pass
    @abstractmethod
    def update(self,config:dict,error:float,info=None):
        '''Update scheduler after trial completion'''
        pass
class HPOTuner:
    def __init__(self,scheduler:HPOScheduler,objective_fn:callable):
        self.scheduler=scheduler
        self.objective_fn=objective_fn
        self.incumbent=None # Best performing configuration
        self.incumbent_error=None # Lowest validation error
        self.incumbent_trajectory=[] # Lowest validation errors over time
        self.cumulative_runtime=[]
        self.current_runtime=0
        self.records=[]
    def run(self,number_of_trials):
        for i in range(number_of_trials):
            start_time=time.time()
            config=self.scheduler.suggest()
            print(f'Trial {i} config: {config}')
            error=self.objective_fn(config) # Calculate error with objective function
            '''error=float(error.cpu().detach().numpy)''' # In here we do not use this command because error is already a float number 
                                                          # @QUESTION: why error shoulbe not be a Tensor in HPO context?
            self.scheduler.update(config,error)
            runtime=time.time()-start_time
            self.bookkeeping(config,error,runtime)
            print(f'error: {error:.4f} - runtime: {runtime:.2f}')
    def bookkeeping(self,config:dict,error:float,runtime:float):
        '''Track best configuration and respective performace'''
        self.records.append({
            'config':config,
            'error':error,
            'runtime':runtime
        })
        # Update incumbent
        if self.incumbent is None or error<self.incumbent_error:
            self.incumbent=config
            self.incumbent_error=error
        # Track trajectories
        self.incumbent_trajectory.append(self.incumbent_error)
        self.current_runtime+=runtime
        self.cumulative_runtime.append(self.current_runtime) # @QUESTION: why do not save runtime instead?
    def get_best_config(self):
        return self.incumbent, self.incumbent_error
class RandomSearcher(HPOSeacher):
    def __init__(self,config_space:dict,initial_config:dict):
        self.config_space=config_space
        self.initial_config=initial_config
    def sample_config(self)->dict:
        '''Sample randoom configuration from config space'''
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None  # Clear after first use
            return result
        random_config={
            key:domain.rvs() for key,domain in self.config_space.items()
        }
        return random_config
class BasicScheduler(HPOScheduler):
    def __init__(self,searcher):
        self.searcher=searcher
    def suggest(self)->dict:
        '''Suggest next configuration'''
        return self.searcher.sample_config()
    def update(self,config:dict,error:float,info=None):
        '''Update searcher with trial results'''
        # @TODO: for now, method update does not do anything, maybe implement exploiting past observations here
        self.searcher.update(config,error,additional_info=info)

            