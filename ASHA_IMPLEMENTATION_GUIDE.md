# ASHA (Asynchronous Successive Halving) Implementation Guide

This guide shows how to implement ASHA in your existing codebase without modifying your current code.

## Overview

ASHA improves upon synchronous Successive Halving by:
1. **Avoiding worker idle time**: Workers don't wait for all trials at a rung to complete
2. **Asynchronous promotion**: Configs are promoted as soon as enough results are available
3. **Better resource utilization**: New configs start immediately when workers become free

## Key Differences: Synchronous SH vs ASHA

### Synchronous SH (your current implementation)
- Waits for ALL trials at rung r_i to complete before promoting
- Promotes exactly top 1/eta fraction after full synchronization
- Can cause worker idle time (stragglers block progress)

### ASHA
- Promotes when ENOUGH trials at rung r_i are complete (not ALL)
- Checks from top rung down: if ≥ eta completed trials exist, promote top 1/eta
- Otherwise, starts new config at bottom rung (r_min)
- Workers stay busy, minimal idle time

## Implementation

### Step 1: Add ASHA Scheduler Class to `src/hpo.py`

Add this new class after your `MultiFidelitycheduler` class:

```python
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
```

### Step 2: Add ASHA HPO Function to `HPO` Class

Add this static method to your `HPO` class in `src/hpo.py`:

```python
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
```

### Step 3: Add Training Function to `main.py`

Add this function to your `main.py`:

```python
def train_with_asha(args):
    """Train using ASHA (Asynchronous Successive Halving)"""
    config_space = {
        "learning_rate": stats.loguniform(1e-4, 1),
        "batch_size": stats.randint(35, 512),
    }
    initial_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    
    best_config, best_score, tuner = HPO.asha_random_search(
        args, config_space=config_space, initial_config=initial_config
    )
    
    print(f"\nTraining final model with best config: {best_config}")
    train_loader, val_loader, test_loader = Utils.load_fashion_mnist(
        best_config["batch_size"]
    )
    model = Utils.build_model(args)
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        lr=best_config["learning_rate"],
        num_epochs=args.num_epochs,
    )
    trainer.fit()
    
    train_acc = trainer.evaluate_train()
    val_acc = 1.0 - trainer.validation_error()
    test_acc = trainer.evaluate_test()
    
    print(f"Final train accuracy with ASHA-tuned hyperparameters: {train_acc:.4f}")
    print(f"Final validation accuracy with ASHA-tuned hyperparameters: {val_acc:.4f}")
    print(f"Final test accuracy with ASHA-tuned hyperparameters: {test_acc:.4f}")
    
    HPO.plot_hpo_progress(tuner, save_path="asha_hpo_progress.png")
```

### Step 4: Update Config to Support ASHA

Update your `src/config.py` to add "asha" as a train_mode option:

```python
parser.add_argument(
    "--train_mode",
    type=str,
    default="asha",  # or keep your current default
    choices=["hpo", "fixed", "async_hpo", "multi_fidelity_hpo", "asha"],
)
```

### Step 5: Update Main Function

Update the `main()` function in `main.py` to handle ASHA:

```python
def main():
    parser = Config.new_parser()
    Config.add_training_argument(parser)
    args = parser.parse_args()
    
    if args.train_mode == "hpo":
        train_with_hpo(args)
    elif args.train_mode == "async_hpo":
        train_with_async_hpo(args)
    elif args.train_mode == "multi_fidelity_hpo":
        train_with_multi_fidelity_hpo(args)
    elif args.train_mode == "asha":
        train_with_asha(args)
    else:
        train_fixed(args)
```

## Usage Examples

### Example 1: Basic ASHA with default parameters
```bash
python main.py --train_mode asha --num_trials 20
```

### Example 2: ASHA with custom eta and resource range
```bash
python main.py --train_mode asha \
    --num_trials 30 \
    --eta 3 \
    --min_number_of_epochs 2 \
    --max_number_of_epochs 27
```

### Example 3: ASHA with LeNet model
```bash
python main.py --train_mode asha \
    --model_name lenet \
    --num_trials 20 \
    --eta 2 \
    --min_number_of_epochs 10 \
    --max_number_of_epochs 50
```

## How ASHA Works (Step-by-Step Example)

Let's trace through an ASHA run with:
- eta = 2
- r_min = 10
- r_max = 40
- Rungs: [10, 20, 40]

### Timeline:

1. **Trial 1**: Start config A at rung 10
2. **Trial 2**: Start config B at rung 10
3. **Trial 3**: Config A completes at rung 10 (error=0.5)
4. **Trial 4**: Config B completes at rung 10 (error=0.3)
   - Now we have 2 completed trials at rung 10 (≥ eta=2)
   - Top 1/2 = config B (lowest error)
   - **Promote config B to rung 20**
5. **Trial 5**: Start config C at rung 10 (new config)
6. **Trial 6**: Config B completes at rung 20 (error=0.25)
7. **Trial 7**: Config C completes at rung 10 (error=0.4)
8. **Trial 8**: Start config D at rung 10 (new config)
9. ... and so on

**Key observation**: No waiting! As soon as a worker is free, it either:
- Promotes a config (if enough trials completed at a rung)
- Starts a new config at r_min

## Comparison: Synchronous SH vs ASHA

### Synchronous SH (your current `MultiFidelitycheduler`)
```
Rung 10: [Config A, Config B, Config C, Config D] ← ALL must complete
         ↓ (wait for ALL)
Rung 20: [Config B, Config C] ← top 1/2 promoted
         ↓ (wait for ALL)
Rung 40: [Config B] ← top 1/2 promoted
```

**Problem**: Workers idle while waiting for stragglers

### ASHA (new `ASHAScheduler`)
```
Configs evaluated asynchronously:
- A completes at rung 10
- B completes at rung 10 → promote to 20 (enough trials done)
- C starts at rung 10
- B completes at rung 20 → promote to 40
- D starts at rung 10
- C completes at rung 10 → promote to 20
- E starts at rung 10
... (no synchronization barriers!)
```

**Benefit**: Workers always busy, better utilization

## Expected Behavior

When you run ASHA, you'll see:
1. Many configs start at r_min (rung 10)
2. As they complete, top performers get promoted immediately
3. Poor performers get discarded (not promoted)
4. Final rungs have only the best configs
5. **Rung statistics** show how many configs reached each level

Example output:
```
ASHA Rung Statistics:
  Rung  10:  20 completed,  10 promoted
  Rung  20:  10 completed,   5 promoted
  Rung  40:   5 completed,   0 promoted
```

This means:
- 20 configs tried at lowest budget (10 epochs)
- 10 were promoted to medium budget (20 epochs)
- 5 were promoted to highest budget (40 epochs)
- Best of those 5 is your final recommendation

## Tuning num_trials for ASHA

For ASHA, the number of trials should be **larger** than synchronous SH because:
- Trials complete asynchronously
- More trials = more opportunities for good configs
- Typical: 2-5x the n_0 of synchronous SH

For your defaults (eta=2, r_min=10, r_max=50):
- Synchronous SH n_0 ≈ 4
- **Recommended for ASHA**: 20-30 trials

## Advanced: Visualizing ASHA Behavior

You can extend the plotting function to show ASHA-specific metrics:

```python
@staticmethod
def plot_asha_rungs(tuner, scheduler, save_path=None):
    """Plot ASHA rung progression"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot trials at each rung
    for rung in scheduler.rung_levels:
        trials = scheduler.completed_trials_at_rungs[rung]
        if trials:
            errors = [err for _, err in trials]
            rungs = [rung] * len(errors)
            ax.scatter(rungs, errors, alpha=0.6, s=50)
    
    ax.set_xlabel('Rung (Resource Level)')
    ax.set_ylabel('Validation Error')
    ax.set_title('ASHA: Trial Distribution Across Rungs')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

## Key Takeaways

1. **ASHA = Async SH**: Removes synchronization barriers from Successive Halving
2. **Better utilization**: Workers stay busy, no idle time waiting for stragglers
3. **Same outcome**: Still allocates more resources to promising configs
4. **More trials needed**: Because of async nature, run more trials than sync SH
5. **Easy to use**: Same interface as your existing HPO methods

## Next Steps

1. Copy the `ASHAScheduler` class into `src/hpo.py`
2. Add the `asha_random_search` method to the `HPO` class
3. Add the `train_with_asha` function to `main.py`
4. Update config choices
5. Run with `python main.py --train_mode asha --num_trials 20`

Enjoy faster, more efficient multi-fidelity HPO! 🚀
