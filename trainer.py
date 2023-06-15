import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pytorch_warmup as warmup
from tqdm import tqdm
import os
import yaml

class Trainer:
    def __init__(self, model, train_dataloader, device, version, val_dataloader=None):
        self.read_config()

        self.model = model

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(model.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.n_iter * len(self.train_dataloader))
        self.warmup = warmup.LinearWarmup(self.optim, 10000)
        self.device = device

        self.writer = SummaryWriter(log_dir=f"tb_logs/{version}")

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        trainer_params = params["TrainerParams"]

        self.n_iter = trainer_params["n_iter"]
        self.lr = trainer_params["lr"]
        self.weight_decay = trainer_params["weight_decay"]

    def train_loop(self, epoch):
        self.model.train()
        with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch}", leave=False) as pbar:
            total_loss = 0.0
            total_acc = 0.0

            for i, batch in pbar:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x)

                loss = self.loss_fn(predictions, y)

                self.optim.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

                self.writer.add_scalar("lr", self.optim.param_groups[0]['lr'],
                                       global_step=epoch * len(self.train_dataloader) + i)
                #with self.warmup.dampening():
                self.lr_scheduler.step()

                pbar.set_postfix(loss='{:.10f}'.format(loss.item()))
                total_loss += loss.item()
                step_acc = (torch.argmax(predictions, dim=-1) == y).float().mean().item()
                total_acc += step_acc
                self.writer.add_scalar("loss/train_step", loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)
                self.writer.add_scalar("acc/train_step", step_acc,
                                       global_step=epoch * len(self.train_dataloader) + i)

        epoch_loss = total_loss / len(self.train_dataloader)
        epoch_acc = total_acc / len(self.train_dataloader)

        return epoch_loss, epoch_acc

    def val_loop(self, epoch):
        self.model.eval()
        with torch.no_grad():
            with tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), desc=f"Epoch {epoch}", leave=False) as pbar:
                total_loss = 0.0
                total_acc = 0.0

                for i, batch in pbar:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)

                    predictions = self.model(x)

                    loss = self.loss_fn(predictions, y)

                    pbar.set_postfix(loss='{:.10f}'.format(loss.item()))
                    total_loss += loss.item()
                    step_acc = (torch.argmax(predictions, dim=-1) == y).float().mean().item()
                    total_acc += step_acc

        epoch_loss = total_loss / len(self.val_dataloader)
        epoch_acc = total_acc / len(self.val_dataloader)

        return epoch_loss, epoch_acc

    def train(self):
        try:
            for epoch in range(self.n_iter):
                epoch_loss, epoch_acc = self.train_loop(epoch)

                self.writer.add_scalar("loss/train_epoch", epoch_loss, global_step=epoch)
                self.writer.add_scalar("acc/train_epoch", epoch_acc, global_step=epoch)

                if self.val_dataloader is None:
                    continue

                epoch_loss, epoch_acc = self.val_loop(epoch)

                self.writer.add_scalar("loss/val_epoch", epoch_loss, global_step=epoch)
                self.writer.add_scalar("acc/val_epoch", epoch_acc, global_step=epoch)
        except KeyboardInterrupt:
            print("Training interrupted. Logging hyperparameters.")

        self.log_hparams()

    def log_hparams(self):
        trainer_params = {key: vars(self)[key] for key in vars(self)
                          if (type(vars(self)[key]) == int or type(vars(self)[key]) == float)}
        self.writer.add_text("trainer_hparams", str(trainer_params), 0)

        net_params = {key: vars(self.model)[key] for key in vars(self.model)
                      if (type(vars(self.model)[key]) == int or type(vars(self.model)[key]) == float)}
        self.writer.add_text("model_hparams", str(net_params), 0)

        self.writer.flush()
        self.writer.close()
