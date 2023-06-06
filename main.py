import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup as warmup
from tqdm import tqdm

from dataset import ImageNet
from models.vit import ViT

if __name__ == "__main__":
    # ------------------ ARGUMENT PARSING ------------------
    parser = argparse.ArgumentParser(description="ViT training", allow_abbrev=False)

    parser.add_argument("--version", "-v", required=True, type=str, help="Version name for TensorBoard.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ LOAD DATA ------------------
    train_dataset = ImageNet(type="train")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=512,
                                  num_workers=32,
                                  shuffle=True,
                                  pin_memory=False)

    val_dataset = ImageNet(type="val")
    val_dataloader = DataLoader(train_dataset,
                                batch_size=512,
                                num_workers=32,
                                shuffle=False,
                                pin_memory=False)

    # ------------------ GET MODEL ------------------
    vit_model = ViT().to(device)
    vit_model = nn.DataParallel(vit_model)

    # ------------------ TRAINING PARAMATERS, LOGGING ------------------
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(vit_model.parameters(), weight_decay=0.03, lr=3e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=300)
    warmup_scheduler = warmup.LinearWarmup(optim, warmup_period=5)

    writer = SummaryWriter(log_dir=f"tb_logs/{args.version}")

    # ------------------ TRAIN LOOP ------------------
    for e in range(300):
        # ------------------ TRAIN ------------------
        vit_model.train()
        writer.add_scalar("lr", lr_scheduler.get_lr(), global_step=e)
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {e}", leave=False) as pbar:
            total_loss = 0.0
            total_acc = 0.0

            for i, batch in pbar:
                for key in batch:
                    batch[key] = batch[key].to(device)

                predictions = vit_model(batch)

                loss = loss_fn(predictions, batch["label"])

                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.set_postfix(loss='{:.10f}'.format(loss.item()))

                total_loss += loss.item()
                writer.add_scalar("loss/train_step", loss.item(), global_step=e * len(train_dataloader) + i)

                corrects = (torch.argmax(predictions, dim=-1) == batch["label"]).sum()
                total_acc += corrects
                writer.add_scalar("acc/train_step", corrects / len(predictions),
                                  global_step=e * len(train_dataloader) + i)

        with warmup_scheduler.dampening():
            lr_scheduler.step()

        writer.add_scalar("loss/train_epoch", total_loss / len(train_dataloader), global_step=e * len(train_dataloader) + i)
        writer.add_scalar("acc/train_epoch", total_acc / len(train_dataset), global_step=e * len(train_dataloader) + i)

        # ------------------ VALIDATE ------------------
        vit_model.eval()
        with tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {e}", leave=False) as pbar:
            total_loss = 0.0
            total_acc = 0.0

            for i, batch in pbar:
                for key in batch:
                    batch[key] = batch[key].to(device)

                predictions = vit_model(batch)

                loss = loss_fn(predictions, batch["label"])

                pbar.set_postfix(loss='{:.10f}'.format(loss.item()))

                total_loss += loss.item()

                corrects = (torch.argmax(predictions, dim=-1) == batch["label"]).sum()
                total_acc += corrects

        writer.add_scalar("loss/val_epoch", total_loss / len(val_dataloader), global_step=e * len(val_dataloader) + i)
        writer.add_scalar("acc/val_epoch", total_acc / len(val_dataset), global_step=e * len(val_dataloader) + i)

