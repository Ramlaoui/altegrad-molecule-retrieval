from src.trainer import BaseTrainer
import argparse
import yaml
from pathlib import Path
from src.trainer import BaseTrainer
from src.model import Baseline

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
        "--num_node_features", type=int, default=9, help="number of node features"
    )
    parser.add_argument(
        "--nout", type=int, default=128, help="size of the output layer"
    )
    parser.add_argument(
        "--nhid", type=int, default=256, help="size of the hidden layer"
    )
    parser.add_argument(
        "--graph_hidden_channels",
        type=int,
        default=128,
        help="size of the hidden layer of the graph encoder",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="weight decay"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--save_model",
        type=str,
        default="model.pt",
        help="path to save the trained model",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default="results.csv",
        help="path to save the results",
    )
    parser.add_argument(
        "--save_embeddings",
        type=str,
        default="embeddings.csv",
        help="path to save the embeddings",
    )
    parser.add_argument(
        "--save_attention",
        type=str,
        default="attention.csv",
        help="path to save the attention weights",
    )
    parser.add_argument(
        "--save_graphs", type=str, default="graphs.csv", help="path to save the graphs"
    )
    parser.add_argument(
        "--save_predictions",
        type=str,
        default="predictions.csv",
        help="path to save the predictions",
    )
    parser.add_argument(
        "--save_losses", type=str, default="losses.csv", help="path to save the losses"
    )
    parser.add_argument(
        "--save_metrics",
        type=str,
        default="metrics.csv",
        help="path to save the metrics",
    )
    parser.add_argument(
        "--save_logs", type=str, default="logs.csv", help="path to save the logs"
    )
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
    config_path = Path("configs") / (args.config + ".yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = MODELS_DICT[args.config]

    # TODO Need to change that later depending on an argument
    config = config["default"]
    config["name_model"] = args.config
    config["model_object"] = model

    config["is_debug"] = args.is_debug
    config["wandb_project"] = args.wandb_project

    trainer = BaseTrainer(**config, load=True)
    trainer.train()
    trainer.submit_run()
