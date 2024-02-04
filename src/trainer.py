from src.dataloader import GraphTextDataset, GraphDataset, TextDataset
import yaml
import wandb
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from src.models.baseline import Baseline
from src.models.gat import GATModel
from src.models.gin import GINMol
from src.models.basicproj import BasicProj
from src.models.qformer import QFormer
from src.models.crossatt import CrossAttentionModel
from src.models.monet import Monet
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ConstantLR
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
        self.saved_checkpoint = False
        # Early stopping with file creation?

        if load:
            self.load()

    def load(self, checkpoint_name=None):
        self.load_logger()
        self.load_train_val_datasets()
        if checkpoint_name is not None:
            self.load_checkpoint(checkpoint_name)
        else:
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

    def load_val_text_graph_dataloaders(self):
        if "tokenizer" not in self.__dict__:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if "gt" not in self.__dict__:
            self.gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

        self.val_cids_dataset = GraphDataset(
            root="./data/", gt=self.gt, split="val_cids"
        )
        self.val_text_dataset = TextDataset(
            file_path="./data/val_text.txt", tokenizer=self.tokenizer
        )

        self.val_gt_loader = DataLoader(
            self.val_cids_dataset,
            batch_size=self.config["optim"]["eval_batch_size"],
            shuffle=False,
        )

    def get_dataloader(self):
        batch_size = self.config["optim"]["batch_size"]
        eval_batch_size = self.config["optim"]["eval_batch_size"]
        max_epochs = self.config["optim"].get("max_epochs", -1)
        max_steps = self.config["optim"].get("max_steps", -1)
        max_samples = self.config["optim"].get("max_samples", -1)

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=eval_batch_size, shuffle=True
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
            weight_decay=float(self.config["optim"].get("weight_decay", 0.01)),
        )

        if self.config["optim"].get("scheduler", "cosine") == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(self.config["optim"].get("warmup_steps", 100)),
                # T_mult=self.config["optim"].get("T_mult", 2),
                eta_min=float(self.config["optim"].get("lr_min", 0)),
            )
        else:
            self.scheduler = ConstantLR(
                self.optimizer,
                last_epoch=-1,
            )
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")

    def load_loss(
        self,
    ):
        loss_name = self.config["optim"].get("loss", "contrastive")
        if loss_name == "contrastive":
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

                if self.device.type != "cuda":
                    scaler = None
                    dtype = torch.bfloat16
                else:
                    scaler = torch.cuda.amp.GradScaler()
                    dtype = torch.float16
                scaler = None
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("precision", "foat32") == "float16",
                ):
                    if self.config["model_object"] == QFormer:
                        loss_gtc, loss_gtm = self.model(
                            graph_batch.to(self.device),
                            input_ids.to(self.device),
                            attention_mask.to(self.device),
                        )
                        current_loss = loss_gtm + loss_gtc
                    else:
                        x_graph, x_text = self.model(
                            graph_batch.to(self.device),
                            input_ids.to(self.device),
                            attention_mask.to(self.device),
                        )
                        current_loss = contrastive_loss(x_graph, x_text)
                    if scaler is not None:
                        self.optimizer.zero_grad()
                        scaler.scale(current_loss).backward()
                        scaler.unscale_(self.optimizer)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.zero_grad()
                        current_loss.backward()
                        self.optimizer.step()

                if self.config["optim"].get("scheduler", "cosine") == "cosine":
                    self.scheduler.step(i + count_iter / len(self.train_loader))
                else:
                    self.scheduler.step()
                loss += current_loss.item()

                count_iter += 1
                if not self.is_debug:
                    self.logger.log(
                        {
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )
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
            if self.config["model_object"] == QFormer:
                val_loss_gtc = 0
                val_loss_gtm = 0
            val_loss = 0
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            for batch in self.val_loader:
                input_ids = batch.input_ids
                batch.pop("input_ids")
                attention_mask = batch.attention_mask
                batch.pop("attention_mask")
                graph_batch = batch
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("precision", "foat32") == "float16",
                ):
                    if self.config["model_object"] == QFormer:
                        loss_gtc, loss_gtm = self.model(
                            graph_batch.to(self.device),
                            input_ids.to(self.device),
                            attention_mask.to(self.device),
                        )
                        current_loss = loss_gtm + loss_gtc
                        val_loss_gtc += loss_gtc.item()
                        val_loss_gtm += loss_gtm.item()
                    else:
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
                log_dict = {
                    "validation_loss": val_loss / len(self.val_loader),
                    "epoch": i,
                }
                if self.config["model_object"] == QFormer:
                    log_dict["validation_loss_gtc"] = val_loss_gtc / len(
                        self.val_loader
                    )
                    log_dict["validation_loss_gtm"] = val_loss_gtm / len(
                        self.val_loader
                    )
                self.logger.log(
                    log_dict,
                )
            if self.best_validation_loss == val_loss:
                if not self.silent:
                    print("validation loss improved saving checkpoint...")
                self.save_path = (
                    self.config["checkpoint_dir"]
                    + f"/best_checkpoint_{self.run_name}.pt"
                )
                if not (Path(self.config["checkpoint_dir"])).exists():
                    os.makedirs(Path(self.config["checkpoint_dir"]), exist_ok=True)
                # If file exists, delete it
                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                if not self.saved_checkpoint:
                    self.saved_checkpoint = True
                    config_to_save = self.config.copy()
                    config_to_save["model_object"] = config_to_save[
                        "model_object"
                    ].__name__
                    yaml.dump(
                        config_to_save,
                        open(
                            self.save_path.replace(".pt", ".yaml").replace(
                                "best_checkpoint_", ""
                            ),
                            "w",
                        ),
                    )
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

                cosine_similarity, mrr = self.get_mrr_val(load_checkpoint=False)
                if not self.is_debug:
                    self.logger.log({"mrr_val": mrr})
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Give MRR with best checkpoint
        cosine_similarity, mrr = self.get_mrr_val(load_checkpoint=True)
        if not self.is_debug:
            self.logger.log({"mrr_val": mrr})

    def load_checkpoint(self, checkpoint_name=None):
        if checkpoint_name is None:
            checkpoint_name = f"best_checkpoint_{self.run_name}.pt"
        checkpoint_path = self.config["checkpoint_dir"] + f"/{checkpoint_name}"
        self.save_path = checkpoint_path
        self.load_model()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()

    def get_mrr_val(self, load_checkpoint=True):
        if not self.silent:
            print("Submitting run on validation set...")

        cosine_similarity = self.submit_run(
            split="val", load_checkpoint=load_checkpoint
        )

        # true_cids = self.val_cids_dataset.cids
        true_cids_ranking = np.eye(cosine_similarity.shape[0])

        from sklearn.metrics import label_ranking_average_precision_score

        mrr = label_ranking_average_precision_score(
            true_cids_ranking, cosine_similarity
        )

        if not self.silent:
            print("MRR on validation set: {}".format(mrr))

        return cosine_similarity, mrr

    def submit_run(self, split="test", load_checkpoint=True):
        if split == "val":
            self.load_val_text_graph_dataloaders()
        else:
            self.load_test_dataloader()

        # delete train val loaders to free memory
        if split == "test":
            try:
                del self.train_loader
                del self.val_loader
                del self.train_dataset
                del self.val_dataset
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except:
                pass

        if load_checkpoint:
            if not self.silent:
                print("loading best model...")

            self.load_model()
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)["model_state_dict"]
            )
        self.model.eval()

        batch_size = self.config["optim"]["eval_batch_size"]

        if self.config["model_object"] == QFormer:
            graph_model = lambda batch_inputs: self.model.graph_forward(
                batch_inputs.to(self.device)
            )
            text_model = lambda input_ids, attention_mask: self.model.text_forward(
                input_ids.to(self.device), attention_mask.to(self.device)
            )
        else:
            graph_model = self.model.get_graph_encoder()
            text_model = self.model.get_text_encoder()

        if split == "val":
            text_dataset = self.val_text_dataset
            graph_dataset = self.val_cids_dataset
            loader = self.val_gt_loader
        else:
            text_dataset = self.test_text_dataset
            graph_dataset = self.test_cids_dataset
            loader = self.test_loader

        graph_embeddings = []
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.config.get("precision", "foat32") == "float16",
        ):
            for batch in loader:
                for output in graph_model(batch.to(self.device)):
                    graph_embeddings.append(output.tolist())

        text_loader = TorchDataLoader(
            text_dataset, batch_size=batch_size, shuffle=False
        )
        text_embeddings = []

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.config.get("precision", "foat32") == "float16",
        ):
            for batch in text_loader:
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
                self.config["results_dir"],
                f"results_{split}_{self.run_name}.csv",
            ),
            index=False,
        )

        if split == "val":
            del self.val_cids_dataset
            del self.val_text_dataset
            del self.val_gt_loader
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return similarity
