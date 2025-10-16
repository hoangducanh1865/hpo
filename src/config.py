import argparse
import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def new_parser(name=None):
        return argparse.ArgumentParser(prog=name)

    @staticmethod
    def add_training_argument(parser):
        parser.add_argument(
            "--train_mode",
            type=str,
            default="async_hpo",
            choices=["hpo", "fixed", "async_hpo"],
        )
        parser.add_argument(
            "--model_name", type=str, default="lenet", choices=["lenet", "softmax"]
        )
        parser.add_argument("--num_epochs", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=0.1)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--num_outputs", type=int, default=10)
        parser.add_argument("--num_trials", type=int, default=10)
        parser.add_argument("--max_wallclock_time", type=int, default=60 * 60)
