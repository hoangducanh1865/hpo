from src.model import SoftmaxRegression, LeNet
from src.utils import Utils
from src.trainer import Trainer
from src.config import Config
from src.hpo import HPO
from scipy import stats
from syne_tune.config_space import loguniform, randint


def train_fixed(args):
    fixed_config = {"learning_rate": args.learning_rate, "batch_size": args.batch_size}
    print(f"Training with fixed config: {fixed_config}")
    train_loader, val_loader, test_loader = Utils.load_fashion_mnist(args.batch_size)
    model = Utils.build_model(args)
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        lr=Config.lr,
        num_epochs=Config.num_epochs,
    )
    trainer.fit()
    train_acc = trainer.evaluate_train()
    val_acc = 1.0 - trainer.validation_error()
    test_acc = trainer.evaluate_test()
    print(f"Final train accuracy: {train_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")


def train_with_hpo(args):
    config_space = {
        "learning_rate": stats.loguniform(1e-4, 1),
        "batch_size": stats.randint(35, 512),
    }
    initial_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    best_config, best_score, tuner = HPO.random_search(
        args, config_space=config_space, initial_config=initial_config
    )
    print(f"Training with best config: {best_config}")
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
    print(f"Final train accuracy with tuned hyperparameters: {train_acc:.4f}")
    print(f"Final validation accuracy with tuned hyperparameters: {val_acc:.4f}")
    print(f"Final test accuracy with tuned hyperparameters: {test_acc:.4f}")
    HPO.plot_hpo_progress(tuner, save_path="hpo_progress.png")


def train_with_async_hpo(args):
    config_space = {
        "learning_rate": loguniform(1e-4, 1),
        "batch_size": randint(32, 512),
        "num_epochs": args.num_epochs,
        "model_name": args.model_name,
        "num_outputs": args.num_outputs,
    }
    """initial_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }"""
    best_config, best_score, tuner = HPO.async_random_search(
        args, config_space=config_space
    )
    print(f"Training with best config: {best_config}")
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
    print(f"Final train accuracy with tuned hyperparameters: {train_acc:.4f}")
    print(f"Final validation accuracy with tuned hyperparameters: {val_acc:.4f}")
    print(f"Final test accuracy with tuned hyperparameters: {test_acc:.4f}")
    HPO.plot_hpo_progress(tuner, save_path="hpo_progress.png")


def train_with_multi_fidelity_hpo(args):
    config_space = {
        "learning_rate": stats.loguniform(1e-4, 1),
        "batch_size": stats.randint(35, 512),
    }
    initial_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    best_config, best_score, tuner = HPO.multi_fidelity_random_search(
        args, config_space=config_space, initial_config=initial_config
    )
    print(f"Training with best config: {best_config}")
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
    print(f"Final train accuracy with tuned hyperparameters: {train_acc:.4f}")
    print(f"Final validation accuracy with tuned hyperparameters: {val_acc:.4f}")
    print(f"Final test accuracy with tuned hyperparameters: {test_acc:.4f}")
    HPO.plot_hpo_progress(tuner, save_path="hpo_progress.png")


def train_with_asha_hpo(args):
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
        num_epochs=best_config["num_epochs"],
    )
    trainer.fit()

    train_acc = trainer.evaluate_train()
    val_acc = 1.0 - trainer.validation_error()
    test_acc = trainer.evaluate_test()

    print(f"Final train accuracy with ASHA-tuned hyperparameters: {train_acc:.4f}")
    print(f"Final validation accuracy with ASHA-tuned hyperparameters: {val_acc:.4f}")
    print(f"Final test accuracy with ASHA-tuned hyperparameters: {test_acc:.4f}")

    HPO.plot_hpo_progress(tuner, save_path="asha_hpo_progress.png")


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
    elif args.train_mode == "asha_hpo":
        train_with_asha_hpo(args)
    else:
        train_fixed(args)


if __name__ == "__main__":
    main()
