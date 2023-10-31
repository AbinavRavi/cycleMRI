from network.model import Generator
from dataloader.dataloader import loader_instance
from torch import nn
from torch.optim import Adam
import torch
from utils.utils import read_config, get_device, set_seed
import mlflow
from tqdm import tqdm
import numpy as np
import sys
from os.path import abspath
from os.path import dirname as d

parent_dir = f"{d(d(abspath(__file__)))}"
sys.path.append(parent_dir + "/config/")


class GeneratorTrainer:
    def __init__(self, config_path: str) -> None:
        config = read_config(config_path)
        set_seed(config["seed"])
        self.lr = config["train"]["lr"]
        self.epochs = config["train"]["epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.weight_decay = config["train"]["weight_decay"]
        self.experiment_name = config["train"]["experiment_name"]
        self.image_size = config["train"]["image_size"]
        train_dataset_path = config["train"]["data"]
        train_label_path = config["train"]["label"]
        val_dataset_path = config["val"]["data"]
        val_label_path = config["val"]["label"]
        self.train_loader = loader_instance(
            train_dataset_path, train_label_path, self.image_size, self.batch_size
        )
        self.val_loader = loader_instance(
            val_dataset_path, val_label_path, self.image_size, self.batch_size
        )
        self.device = get_device()
        self.model = Generator().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.save_path = config["train"]["checkpoint_path"]

    def _forward(self, data, label, tqdm_loop):
        data, label = data.to(self.device, dtype=torch.float), label.to(
            self.device, dtype=torch.float
        )
        output = self.model(data)
        criterion = nn.MSELoss()
        loss = criterion(output, label)
        tqdm_loop.set_postfix({"loss": loss.item()})
        return loss

    def training_loop(self, epoch_num):
        train_loss = []
        self.model.train()
        train_tqdm = tqdm(self.train_loader, desc="train_iter", leave=False)
        for idx, (data, label) in enumerate(train_tqdm):
            self.optimizer.zero_grad()
            loss = self._forward(data, label, train_tqdm)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            mlflow.log_metric("Itr/Train", loss.item(), epoch_num * len(self.train_loader) + idx)
        epoch_train_loss = np.array(train_loss).mean()
        return epoch_train_loss

    def validation_loop(self, epoch_num):
        val_loss = []
        self.model.eval()
        val_tqdm = tqdm(self.val_loader, desc="val_iter", leave=False)
        with torch.no_grad():
            for idx, (data, label) in enumerate(val_tqdm):
                loss_value = self._forward(data, label, val_tqdm)
                val_loss.append(loss_value.item())
                mlflow.log_metric(
                    "Itr/Validation",
                    loss_value.item(),
                    epoch_num * len(self.val_loader) + idx,
                )
        epoch_val_loss = np.array(val_loss).mean()
        return epoch_val_loss

    def training(self):
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            for i in range(self.epochs):
                epoch_train_loss = self.training_loop(i)
                print(f"epoch ={i} epoch_train_loss={epoch_train_loss}")
                epoch_val_loss = self.validation_loop(i)
                print(f"epoch ={i} epoch_val_loss={epoch_val_loss}")
                mlflow.log_metric("Epoch/trainloss", epoch_train_loss, i)
                mlflow.log_metric("Epoch/valloss", epoch_val_loss, i)
                print(
                    "epoch:{} \t".format(i + 1),
                    "trainloss:{}".format(epoch_train_loss),
                    "\t",
                    "valloss:{}".format(epoch_val_loss),
                )
                if (i + 1) % 2 == 0:
                    torch.save(
                        self.model,
                        f"{self.save_path}model_{self.batch_size}_{self.lr}_{i+1}.pt",
                    )
        mlflow.end_run()


if __name__ == "__main__":
    parent_dir = f"{d(d(abspath(__file__)))}"
    config_file = parent_dir + "/src/config/config.yml"
    net = GeneratorTrainer(config_file)
    net.training()
