from src.dataloader import GraphTextDataset, GraphDataset, TextDataset
import wandb
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from src.models.baseline import Baseline
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import AdamW
import uuid


def contrastive_loss(v1, v2, CE=torch.nn.CrossEntropyLoss()):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


class BaseTrainer:
    def __init__(self, load=True, **kwargs):
        run_dir = kwargs.get("run_dir", "./runs")
        model = kwargs.get("model_object", Baseline)

        self.config = {
            **kwargs,
            "model_object": model,
            "checkpoint_dir": str(Path(run_dir) / "checkpoints"),
            "results_dir": str(Path(run_dir) / "results"),
            "logs_dir": str(Path(run_dir) / "logs"),
        }

        self.is_debug = self.config.get("is_debug", False)
        self.silent = self.config.get("silent", False)

        self.epoch = 0
        self.losses = []
        self.step = 0
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.datasets = {}
        self.loaders = {}

        self.timestamp_id = time.strftime("%Y%m%d-%H%M%S")
        self.job_id = uuid.uuid4().hex[:8]
        self.config["timestamp_id"] = self.timestamp_id
        self.run_name = f"{kwargs['config_name']}_{self.timestamp_id}"
        # Early stopping with file creation?

        if load:
            self.load()

    def load(self):
        self.load_logger()
        self.load_train_val_datasets()
        self.load_model()
        self.load_optimizer()
        self.load_loss()
        self.get_dataloader()

    def load_logger(self):
        self.logger = None
        if not self.is_debug:
            logger = self.config.get("logger", "wandb")
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Logger name not provided"

            self.logger = wandb.init(
                project=self.config["wandb_project"],
                name=self.run_name,
                config=self.config,
                dir=self.config["logs_dir"],
            )

    def load_train_val_datasets(
        self,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

        self.val_dataset = GraphTextDataset(
            root="./data/", gt=self.gt, split="val", tokenizer=self.tokenizer
        )
        self.train_dataset = GraphTextDataset(
            root="./data/", gt=self.gt, split="train", tokenizer=self.tokenizer
        )

    def load_test_dataloader(self):
        if "tokenizer" not in self.__dict__:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if "gt" not in self.__dict__:
            self.gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

        self.test_cids_dataset = GraphDataset(
            root="./data/", gt=self.gt, split="test_cids"
        )
        self.test_text_dataset = TextDataset(
            file_path="./data/test_text.txt", tokenizer=self.tokenizer
        )

        self.test_loader = DataLoader(
            self.test_cids_dataset,
            batch_size=self.config["optim"]["eval_batch_size"],
            shuffle=False,
        )

    def get_dataloader(self):
        batch_size = self.config["optim"]["batch_size"]
        max_epochs = self.config["optim"].get("max_epochs", -1)
        max_steps = self.config["optim"].get("max_steps", -1)
        max_samples = self.config["optim"].get("max_samples", -1)

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )

        # Use samplers?
        # Use normalizers?

    def load_model(self):
        loader = list(self.loaders.values())[0] if self.loaders else None
        if loader:
            sample = loader.dataset[0]
            # extract inputs and targets if necessary

        model_config = {
            # add other features if necessary
            **self.config["model"],
        }

        # Extract model class from name
        self.model = self.config["model_object"](**model_config).to(self.device)

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")

        optimizer = eval(optimizer)

        self.optimizer = optimizer(
            self.model.parameters(),
            lr=float(self.config["optim"]["lr_initial"]),
            **self.config["optim"].get("optimizer_params", {}),
        )

    def load_loss(
        self,
    ):
        loss_name = self.config["optim"].get("loss", "contrastive")
        if self.config["optim"]["loss"] == "contrastive":
            self.CE = torch.nn.CrossEntropyLoss()
            self.loss = lambda v1, v2: contrastive_loss(v1, v2, self.CE)
        else:
            raise NotImplementedError(f"Loss {loss_name} not implemented")

    def train(self):
        for i in tqdm(range(self.epoch, self.config["optim"]["max_epochs"])):
            if not self.silent:
                print("-----EPOCH{}-----".format(i + 1))
            self.epoch = i
            self.model.train()
            start_time = time.time()
            print_every = self.config.get("print_every", 20)
            count_iter = 0
            self.best_validation_loss = np.inf
            loss = 0
            for batch in self.train_loader:
                input_ids = batch.input_ids
                batch.pop("input_ids")
                attention_mask = batch.attention_mask
                batch.pop("attention_mask")
                graph_batch = batch

                # if not self.is_debug:
                #     self.logger.log(
                #         {
                #             "lr": self.optimizer.param_groups[0]["lr"],
                #         }
                #     )

                x_graph, x_text = self.model(
                    graph_batch.to(self.device),
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                )
                current_loss = contrastive_loss(x_graph, x_text)
                self.optimizer.zero_grad()
                current_loss.backward()
                self.optimizer.step()
                loss += current_loss.item()

                count_iter += 1
                if count_iter % print_every == 0:
                    time2 = time.time()
                    self.losses.append(loss)
                    if not self.is_debug:
                        self.logger.log(
                            {
                                "loss": loss / print_every,
                                "epoch": i,
                                "iteration": count_iter,
                            }
                        )
                    loss = 0
            self.model.eval()
            val_loss = 0
            for batch in self.val_loader:
                input_ids = batch.input_ids
                batch.pop("input_ids")
                attention_mask = batch.attention_mask
                batch.pop("attention_mask")
                graph_batch = batch
                x_graph, x_text = self.model(
                    graph_batch.to(self.device),
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                )
                current_loss = contrastive_loss(x_graph, x_text)
                val_loss += current_loss.item()
            self.best_validation_loss = min(self.best_validation_loss, val_loss)
            if not self.silent:
                print(
                    "-----EPOCH" + str(i + 1) + "----- done.  Validation loss: ",
                    str(val_loss / len(self.val_loader)),
                )
            if not self.is_debug:
                self.logger.log(
                    {
                        "validation_loss": val_loss / len(self.val_loader),
                        "epoch": i,
                    }
                )
            if self.best_validation_loss == val_loss:
                if not self.silent:
                    print("validation loss improoved saving checkpoint...")
                self.save_path = (
                    self.config["checkpoint_dir"]
                    + f"/best_checkpoint_{self.run_name}.pt"
                )
                if not (Path(self.config["checkpoint_dir"])).exists():
                    os.makedirs(Path(self.config["checkpoint_dir"]), exist_ok=True)
                # If file exists, delete it
                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "validation_accuracy": val_loss,
                        "loss": loss,
                    },
                    self.save_path,
                )
                if not self.silent:
                    print("checkpoint saved to: {}".format(self.save_path))

    def submit_run(self):
        if not self.silent:
            print("loading best model...")
        self.load_test_dataloader()

        # delete train val loaders to free memory
        del self.train_loader
        del self.val_loader
        del self.train_dataset
        del self.val_dataset

        self.load_model()
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        batch_size = self.config["optim"]["eval_batch_size"]

        graph_model = self.model.get_graph_encoder()
        text_model = self.model.get_text_encoder()

        idx_to_cid = self.test_cids_dataset.get_idx_to_cid()

        graph_embeddings = []
        for batch in self.test_loader:
            for output in graph_model(batch.to(self.device)):
                graph_embeddings.append(output.tolist())

        test_text_loader = TorchDataLoader(
            self.test_text_dataset, batch_size=batch_size, shuffle=False
        )
        text_embeddings = []
        for batch in test_text_loader:
            for output in text_model(
                batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            ):
                text_embeddings.append(output.tolist())

        similarity = cosine_similarity(text_embeddings, graph_embeddings)

        solution = pd.DataFrame(similarity)
        solution["ID"] = solution.index
        solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
        if not (Path(self.config["results_dir"])).exists():
            os.makedirs(Path(self.config["results_dir"]), exist_ok=True)
        solution.to_csv(
            os.path.join(
                self.config["results_dir"], f"results_{self.timestamp_id}.csv"
            ),
            index=False,
        )


# model = Model(
#     model_name=model_name,
#     num_node_features=300,
#     nout=768,
#     nhid=300,
#     graph_hidden_channels=300,
# )  # nout = bert model hidden dim
# model.to(device)

# optimizer = optim.AdamW(
#     model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01
# )

# epoch = 0
# loss = 0
# losses = []
# count_iter = 0
# time1 = time.time()
# printEvery = 50
# best_validation_loss = 1000000
