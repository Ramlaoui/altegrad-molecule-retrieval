from src.trainer import BaseTrainer
import argparse
import yaml
from pathlib import Path
from src.trainer import BaseTrainer
from src.models.baseline import Baseline

MODELS_DICT = {
    "baseline": Baseline,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help="path to the config file",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="name of the model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for training",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--epochs", type=int, help="number of epochs to train")

    parser.add_argument(
        "--is_debug",
        action="store_true",
        help="whether to run in debug mode",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="altegrad-molecular",
        help="name of the wandb project for logs",
    )

    # TODO Parse the arguments to override the config when specified

    args = parser.parse_args()
    config_type = args.config.split("-")
    config_file = config_type[0]
    config_model = "default"
    if len(config_type) > 1:
        config_model = config_type[1]
    config_path = Path("configs") / (args.config + ".yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = MODELS_DICT[args.config]

    config = config[config_model]
    config["name_model"] = args.config_file
    config["model_object"] = model

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    trainer = BaseTrainer(**config, load=True)
    trainer.train()
    trainer.submit_run()
