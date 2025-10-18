import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import sys, os
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from src.model import SoftmaxRegression, LeNet
from src.trainer import Trainer
from src.utils import Utils
from collections import defaultdict
from abc import ABC, abstractmethod
from syne_tune import Reporter
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import PythonBackend
from syne_tune.config_space import loguniform, randint
from syne_tune.optimizer.baselines import RandomSearch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class HPO:
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
            lr = config.get("learning_rate", args.learning_rate)
            batch_size = config.get("batch_size", args.batch_size)
            train_loader, val_loader, test_loader = Utils.load_fashion_mnist(batch_size)
            model = Utils.build_model(args)
            trainer = Trainer(
                model,
                train_loader,
                val_loader,
                test_loader,
                lr=lr,
                num_epochs=config.get("num_epochs", args.num_epochs),
            )
            trainer.fit()
            val_err = trainer.validation_error()
            return val_err

        return hpo_objective

    @staticmethod
    def async_hpo_objective_fn(
        learning_rate, batch_size, num_epochs, model_name, num_outputs
    ):
        from src.model import SoftmaxRegression, LeNet
        from src.trainer import Trainer
        from src.utils import Utils
        from syne_tune import Reporter

        train_loader, val_loader, test_loader = Utils.load_fashion_mnist(batch_size)
        if model_name == "softmax":
            model = SoftmaxRegression(num_outputs=num_outputs)
        elif model_name == "lenet":
            model = LeNet(num_classes=num_outputs)
            model.apply_init()
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            test_loader,
            lr=learning_rate,
            num_epochs=num_epochs,
        )
        report = Reporter()
        for epoch in range(num_epochs):
            if epoch == 0:
                trainer.fit()
            else:
                trainer.train_epoch(epoch)
            val_err = trainer.validation_error()
            report(epoch=epoch, val_err=float(val_err))

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
    def random_search(args, config_space, initial_config):
        searcher = RandomSearcher(config_space, initial_config)
        scheduler = BasicScheduler(searcher)
        objective_fn = HPO.hpo_objective_fn(args)
        tuner = HPOTuner(scheduler=scheduler, objective_fn=objective_fn)
        # Run HPO
        print(f"Starting HPO with {args.num_trials} trials")
        tuner.run(number_of_trials=args.num_trials)
        best_config, best_score = tuner.get_best_config()
        # Summary results
        print("\n" + "=" * 32)
        print(f"HPO Summary:")
        print(f"Best config: {best_config}")
        print(f"Best validation error: {best_score:.4f}")
        print(f"Total runtime: {tuner.current_runtime:.2f}s")
        print(f"Average time per trial: {tuner.current_runtime/args.num_trials:.2f}s")
        return best_config, best_score, tuner  # @QUESTION: why return tuner?

    @staticmethod
    def async_random_search(args, config_space):
        trial_backend = PythonBackend(
            tune_function=HPO.async_hpo_objective_fn, config_space=config_space
        )
        scheduler = RandomSearch(config_space, metrics=["val_err"], do_minimize=True)
        stop_criterion = StoppingCriterion(max_wallclock_time=args.max_wallclock_time)
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=args.num_workers,
            print_update_interval=int(args.max_wallclock_time * 0.6),  # @QUESTION
        )
        tuner.run()
        trial_id, best_config = tuner.best_config(metric="val_err")
        best_result = tuner.trial_backend._trial_dict[trial_id].result
        best_score = best_result["val_err"]
        return best_config, best_score, tuner

    @staticmethod
    def multi_fidelity_random_search(args, config_space, initial_config):
        seacher = RandomSearcher(config_space, initial_config)
        scheduler = MultiFidelitycheduler(
            searcher=seacher,
            eta=args.eta,
            r_min=args.min_number_of_epochs,
            r_max=args.max_number_of_epochs,
            prefact=args.prefact,
        )
        objective_fn = HPO.hpo_objective_fn(args)
        tuner = HPOTuner(scheduler=scheduler, objective_fn=objective_fn)
        print(f"Starting HPO with {args.num_trials} trials")
        tuner.run(number_of_trials=args.num_trials)
        best_config, best_score = tuner.get_best_config()
        print("\n" + "=" * 32)
        print(f"Multi-Fidelity HPO Summary:")
        print(f"Best config: {best_config}")
        print(f"Best validation error: {best_score:.4f}")
        print(f"Total runtime: {tuner.current_runtime:.2f}s")
        print(f"Average time per trial: {tuner.current_runtime/30:.2f}s")
        return best_config, best_score, tuner
    
    @staticmethod
    def asha_random_search(args, config_space, initial_config):
        """
        Asynchronous Successive Halving Algorithm (ASHA)
        
        Uses HPOTuner with ASHAScheduler for asynchronous multi-fidelity optimization.
        Workers don't wait for synchronization, leading to better resource utilization.
        """
        searcher = RandomSearcher(config_space, initial_config)
        scheduler = ASHAScheduler(
            searcher=searcher,
            eta=args.eta,
            r_min=args.min_number_of_epochs,
            r_max=args.max_number_of_epochs,
            prefact=args.prefact,
        )
        objective_fn = HPO.hpo_objective_fn(args)
        tuner = HPOTuner(scheduler=scheduler, objective_fn=objective_fn)
        
        print(f"Starting ASHA with {args.num_trials} trials")
        print(f"Rung levels: {scheduler.rung_levels}")
        print(f"eta={args.eta}, r_min={args.min_number_of_epochs}, r_max={args.max_number_of_epochs}")
        
        tuner.run(number_of_trials=args.num_trials)
        best_config, best_score = tuner.get_best_config()
        
        # Print rung statistics
        print("\n" + "=" * 50)
        print("ASHA Rung Statistics:")
        for rung in scheduler.rung_levels:
            n_completed = len(scheduler.completed_trials_at_rungs[rung])
            n_promoted = len(scheduler.promoted_configs[rung])
            print(f"  Rung {rung:3d}: {n_completed:3d} completed, {n_promoted:3d} promoted")
        
        print("\n" + "=" * 50)
        print(f"ASHA HPO Summary:")
        print(f"Best config: {best_config}")
        print(f"Best validation error: {best_score:.4f}")
        print(f"Total runtime: {tuner.current_runtime:.2f}s")
        print(f"Average time per trial: {tuner.current_runtime/args.num_trials:.2f}s")
        
        return best_config, best_score, tuner

    @staticmethod
    def bayesian_search(args, config_space, initial_config, n_random_init=5):
        """Bayesian Optimization using Gaussian Process + Expected Improvement"""
        searcher = BayesianSearcher(config_space, initial_config, n_random_init=n_random_init)
        scheduler = BasicScheduler(searcher)
        objective_fn = HPO.hpo_objective_fn(args)
        tuner = HPOTuner(scheduler=scheduler, objective_fn=objective_fn)
        # Run HPO
        print(f"Starting Bayesian HPO with {args.num_trials} trials")
        print(f"Random initialization: {n_random_init} trials")
        tuner.run(number_of_trials=args.num_trials)
        best_config, best_score = tuner.get_best_config()
        # Summary results
        print("\n" + "=" * 50)
        print(f"Bayesian HPO Summary:")
        print(f"Best config: {best_config}")
        print(f"Best validation error: {best_score:.4f}")
        print(f"Total runtime: {tuner.current_runtime:.2f}s")
        print(f"Average time per trial: {tuner.current_runtime/args.num_trials:.2f}s")
        return best_config, best_score, tuner

    @staticmethod
    def plot_hpo_progress(tuner, save_path=None):
        """Plot HPO progress over time"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot incumbent error vs trials
        ax1.plot(range(len(tuner.incumbent_trajectory)), tuner.incumbent_trajectory)
        ax1.set_xlabel("Trial")
        ax1.set_ylabel("Best Validation Error")
        ax1.set_title("HPO Progress: Best Error vs Trials")
        ax1.grid(True)

        # Plot incumbent error vs cumulative runtime
        ax2.plot(tuner.cumulative_runtime, tuner.incumbent_trajectory)
        ax2.set_xlabel("Cumulative Runtime (s)")
        ax2.set_ylabel("Best Validation Error")
        ax2.set_title("HPO Progress: Best Error vs Time")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return fig


class HPOSeacher(ABC):
    @abstractmethod
    def sample_config(self) -> dict:
        """Sample new hyperparameter configuration"""
        pass

    def update(self, config: dict, error: float, additional_info=None):
        """Update searcher state after trial completion"""
        pass


class HPOScheduler(ABC):
    @abstractmethod
    def suggest(self) -> dict:
        """Suggest next configuration to evaluate"""
        pass

    @abstractmethod
    def update(self, config: dict, error: float, info=None):
        """Update scheduler after trial completion"""
        pass


class HPOTuner:
    def __init__(self, scheduler: HPOScheduler, objective_fn: callable):
        self.scheduler = scheduler
        self.objective_fn = objective_fn
        self.incumbent = None  # Best performing configuration
        self.incumbent_error = None  # Lowest validation error
        self.incumbent_trajectory = []  # Lowest validation errors over time
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []

    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            print(f"Trial {i} config: {config}")
            error = self.objective_fn(config)  # Calculate error with objective function
            """error=float(error.cpu().detach().numpy)"""  # In here we do not use this command because error is already a float number
            # @QUESTION: why error shoulbe not be a Tensor in HPO context?
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"error: {error:.4f} - runtime: {runtime:.2f}")

    def bookkeeping(self, config: dict, error: float, runtime: float):
        """Track best configuration and respective performace"""
        self.records.append({"config": config, "error": error, "runtime": runtime})
        # Update incumbent
        if self.incumbent is None or error < self.incumbent_error:
            self.incumbent = config
            self.incumbent_error = error
        # Track trajectories
        self.incumbent_trajectory.append(self.incumbent_error)
        self.current_runtime += runtime
        self.cumulative_runtime.append(
            self.current_runtime
        )  # @QUESTION: why do not save runtime instead?

    def get_best_config(self):
        return self.incumbent, self.incumbent_error


class RandomSearcher(HPOSeacher):
    def __init__(self, config_space: dict, initial_config: dict):
        self.config_space = config_space
        self.initial_config = initial_config

    def sample_config(self) -> dict:
        """Sample randoom configuration from config space"""
        if self.initial_config is not None:
            config = self.initial_config
            self.initial_config = None
            return config
        random_config = {key: domain.rvs() for key, domain in self.config_space.items()}
        return random_config


class BayesianSearcher(HPOSeacher):
    """Bayesian Optimization searcher using GP + Expected Improvement"""
    def __init__(self, config_space: dict, initial_config: dict, n_random_init=5):
        self.config_space = config_space
        self.initial_config = initial_config
        self.n_random_init = n_random_init
        self.trial_count = 0
        
        # Track observations: X (configs as arrays), y (errors)
        self.X_observed = []
        self.y_observed = []
        
        # Build bounds for each hyperparameter
        self.param_names = list(config_space.keys())
        self.bounds = []
        for key in self.param_names:
            domain = config_space[key]
            self.bounds.append((domain.a, domain.b)) # domain.a is lower bound, domain.b is upper bound
    
    def sample_config(self) -> dict:
        """Sample next configuration using BO or random initialization"""
        self.trial_count += 1
        
        # Use initial config first
        if self.initial_config is not None:
            config = self.initial_config
            self.initial_config = None
            return config
        
        # Random initialization phase
        if self.trial_count <= self.n_random_init:
            return {key: domain.rvs() for key, domain in self.config_space.items()}
        
        # Bayesian optimization phase
        if len(self.X_observed) < 2:
            # Fallback to random if not enough observations
            return {key: domain.rvs() for key, domain in self.config_space.items()}
        
        # Fit GP and optimize acquisition function
        try:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            
            # Fit GP
            kernel = Matern(nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
            gp.fit(X, y)
            
            # Optimize Expected Improvement
            next_x = self.optimize_acquisition(gp, y.min())
            
            # Convert array back to config dict
            config = {name: float(next_x[i]) for i, name in enumerate(self.param_names)}
            
            # Handle integer parameters (like batch_size)
            for key, domain in self.config_space.items():
                if isinstance(domain, stats._distn_infrastructure.rv_discrete_frozen):
                    config[key] = int(round(config[key]))
            
            return config
        except Exception as e:
            print(f"BO failed ({e}), falling back to random sampling")
            return {key: domain.rvs() for key, domain in self.config_space.items()}
    
    def update(self, config: dict, error: float, additional_info=None):
        """Record observation for GP fitting"""
        # Convert config to array (in consistent order)
        x = np.array([config[name] for name in self.param_names])
        self.X_observed.append(x)
        self.y_observed.append(error)
    
    def optimize_acquisition(self, gp, y_min, n_restarts=25):
        """Optimize Expected Improvement acquisition function"""
        best_x = None
        best_acquisition_value = -np.inf
        
        # Multi-start optimization
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            
            # Minimize negative EI
            result = minimize(
                fun=lambda x: -self.expected_improvement(x, gp, y_min),
                x0=x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.success and -result.fun > best_acquisition_value:
                best_acquisition_value = -result.fun
                best_x = result.x
        
        return best_x if best_x is not None else x0
    
    def expected_improvement(self, x, gp, y_min, xi=0.01):
        """Expected Improvement acquisition function"""
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if sigma == 0:
            return 0.0
        
        # EI formula
        z = (y_min - mu - xi) / sigma
        ei = (y_min - mu - xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei


class BasicScheduler(HPOScheduler):
    def __init__(self, searcher):
        self.searcher = searcher

    def suggest(self) -> dict:
        """Suggest next configuration"""
        return self.searcher.sample_config()

    def update(self, config: dict, error: float, info=None):
        """Update searcher with trial results"""
        # @TODO: for now, method update does not do anything, maybe implement exploiting past observations here
        self.searcher.update(config, error, additional_info=info)


class MultiFidelitycheduler(HPOScheduler):
    def __init__(self, searcher, eta, r_min, r_max, prefact):
        self.searcher = searcher
        self.eta = eta
        self.r_min = r_min
        self.r_max = r_max
        self.prefact = prefact  # @QUESTION
        self.K = int(
            np.log(r_max / r_min) / np.log(eta)
        )  # Compute K, which is later used to determine the number of configurations
        self.rung_levels = [
            r_min * (eta**k) for k in range(self.K + 1)
        ]  # Define the rungs
        if r_max not in self.rung_levels:
            self.rung_levels.append(r_max)  # The final rung should be r_max
            self.K += 1
        # Bookkeeping
        self.observed_error_at_rungs = defaultdict(list)
        self.all_observed_error_at_rungs = defaultdict(list)
        self.queue = []  # Processing queue

    def suggest(self):
        if len(self.queue) == 0:
            # Start a new round of successive halving
            # Number of configurations for the first rung:
            n0 = int(self.prefact * self.eta**self.K)
            for _ in range(n0):
                config = self.searcher.sample_config()
                config["num_epochs"] = self.r_min  # Set r = r_min
                self.queue.append(config)
        # Return an element from the queue
        return self.queue.pop()

    def update(self, config: dict, error: float, info=None):
        ri = int(config["num_epochs"])  # Rung r_i
        # Update our searcher, e.g if we use Bayesian optimization later
        self.searcher.update(config, error, additional_info=info)
        self.all_observed_error_at_rungs[ri].append((config, error))
        if ri < self.r_max:
            # Bookkeeping
            self.observed_error_at_rungs[ri].append((config, error))
            # Determine how many configurations should be evaluated on this rung
            ki = self.K - self.rung_levels.index(ri)
            ni = int(self.prefact * (self.eta**ki))
            # If we observed all configuration on this rung r_i, we estimate the
            # top 1 / eta configuration, add them to queue and promote them for
            # the next rung r_{i+1}
            if len(self.observed_error_at_rungs[ri]) >= ni:
                kiplus1 = ki - 1
                niplus1 = int(self.prefact * (self.eta**kiplus1))
                best_performing_configurations = self.get_top_n_configurations(
                    rung_level=ri, n=niplus1
                )
                riplus1 = self.rung_levels[self.K - kiplus1]  # r_{i+1}
                # Queue may not be empty: insert new entries at the beginning
                self.queue = [
                    dict(config, num_epochs=riplus1)
                    for config in best_performing_configurations
                ] + self.queue
                self.observed_error_at_rungs[ri] = []  # Reset

    def get_top_n_configurations(self, rung_level, n):
        """Configurations are sorted based on their observed performance on the current rung"""
        rung = self.observed_error_at_rungs[rung_level]
        if not rung:
            return []
        sorted_rung = sorted(rung, key=lambda x: x[1])
        return [x[0] for x in sorted_rung[:n]]


class ASHAScheduler(HPOScheduler):
    """
    Asynchronous Successive Halving Algorithm (ASHA)
    
    Key differences from synchronous SH:
    - Promotes configs as soon as enough results are available (not all)
    - No synchronization barriers, workers stay busy
    - Checks rungs from top to bottom for promotion opportunities
    """
    def __init__(self, searcher, eta, r_min, r_max, prefact=1):
        self.searcher = searcher
        self.eta = eta
        self.r_min = r_min
        self.r_max = r_max
        self.prefact = prefact
        
        # Compute rung levels
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        self.rung_levels = [r_min * (eta**k) for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            self.rung_levels.append(r_max)
            self.K += 1
        
        # Track completed trials at each rung
        self.completed_trials_at_rungs = defaultdict(list)  # (config, error) pairs
        
        # Track which configs have been promoted from each rung
        self.promoted_configs = defaultdict(set)  # rung -> set of config hashes
        
        # Track number of configs started at each rung
        self.configs_started_at_rung = defaultdict(int)
    
    def _config_hash(self, config):
        """Create hashable representation of config (excluding num_epochs)"""
        items = [(k, v) for k, v in sorted(config.items()) if k != 'num_epochs']
        return tuple(items)
    
    def suggest(self):
        """
        ASHA suggest logic:
        1. Check rungs from top to bottom for promotion opportunities
        2. If found, promote a config to next rung
        3. Otherwise, start new config at r_min
        """
        # Check rungs from highest to lowest (excluding r_max)
        for i in range(len(self.rung_levels) - 2, -1, -1):
            rung = self.rung_levels[i]
            next_rung = self.rung_levels[i + 1]
            
            # Number of configs that should be started at this rung
            ki = self.K - i
            ni = int(self.prefact * (self.eta ** ki))
            
            # Check if we have enough completed trials to consider promotion
            completed = self.completed_trials_at_rungs[rung]
            
            if len(completed) >= self.eta:  # Need at least eta completed trials
                # Get top 1/eta configs that haven't been promoted yet
                sorted_trials = sorted(completed, key=lambda x: x[1])  # Sort by error
                
                for config, error in sorted_trials:
                    config_hash = self._config_hash(config)
                    
                    # If this config hasn't been promoted from this rung yet
                    if config_hash not in self.promoted_configs[rung]:
                        # Check if we should promote (top 1/eta fraction)
                        top_k = max(1, len(completed) // self.eta)
                        top_configs = [self._config_hash(c) for c, _ in sorted_trials[:top_k]]
                        
                        if config_hash in top_configs:
                            # Promote this config
                            self.promoted_configs[rung].add(config_hash)
                            promoted_config = dict(config)
                            promoted_config['num_epochs'] = next_rung
                            return promoted_config
        
        # No promotion opportunity found, start new config at r_min
        new_config = self.searcher.sample_config()
        new_config['num_epochs'] = self.r_min
        self.configs_started_at_rung[self.r_min] += 1
        return new_config
    
    def update(self, config: dict, error: float, info=None):
        """Record completed trial"""
        ri = int(config['num_epochs'])
        
        # Update searcher
        self.searcher.update(config, error, additional_info=info)
        
        # Record this completion
        self.completed_trials_at_rungs[ri].append((config, error))