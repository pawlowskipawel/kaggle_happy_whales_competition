# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/training.ipynb (unless otherwise specified).

__all__ = ['scheduler_step', 'train_one_epoch', 'train_one_step', 'validate_one_epoch', 'validate_one_step']

# Cell
from .metrics import map_per_set
from tqdm import tqdm
import numpy as np
import torch

# Cell
def scheduler_step(model, lr_scheduler, valid_dataloader, criterion, disable_bar=True, device="cuda"):
    valid_loss, _ = validate_one_epoch(0, model, valid_dataloader, criterion, disable_bar=True, device=device)
    lr_scheduler.step(valid_loss)
    model.train()


# Cell
def train_one_epoch(epoch, model, criterion, optimizer, train_dataloader, grad_accum_iter=1, valid_dataloader=None, lr_scheduler=None, device="cuda"):

    model.train()

    total_loss = 0

    with tqdm(train_dataloader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}') as progress_bar:
        progress_bar.set_description(f"Epoch {epoch+1}".ljust(25))

        for step, batch in enumerate(progress_bar, 1):

            batch_loss = train_one_step(model, batch, criterion, device)

            total_loss += batch_loss.item()

            batch_loss.backward()

            if ((step + 1) % grad_accum_iter == 0) or (step == len(train_dataloader)):
                optimizer.step()

                # More efficient than optimizer.zero_grad()
                for p in model.parameters():
                    p.grad = None

                if lr_scheduler: lr_scheduler.step()

            progress_bar.set_postfix({"train loss": total_loss / step})

            # if valid_dataloader and lr_scheduler and (step % scheduler_step) == 0 and step > 0:
            #     scheduler_step(model, lr_scheduler, valid_dataloader, criterion, disable_bar=True, device="cuda")

    total_loss /= len(train_dataloader)

    return total_loss

# Cell
def train_one_step(model, batch, criterion, device="cuda"):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    outputs = model(images, labels, return_embeddings=False)

    logits = outputs["logits"]
    step_loss = criterion(logits, labels)

    return step_loss

# Cell
@torch.no_grad()
def validate_one_epoch(epoch, model, dataloader, criterion, disable_bar=False, device="cuda"):
    model.eval()

    total_valid_loss = 0
    total_valid_map = 0

    with tqdm(dataloader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}', disable=disable_bar) as progress_bar:
        progress_bar.set_description(f"Validation after epoch {epoch+1}".ljust(25))

        for step, batch in enumerate(progress_bar, 1):

            batch_loss, batch_map = validate_one_step(model, batch, criterion, device)

            total_valid_loss += batch_loss.item()
            total_valid_map += batch_map

            progress_bar.set_postfix({"validation loss": total_valid_loss / step, "validation map@5": total_valid_map / step})

    total_valid_loss /= len(dataloader)
    total_valid_map /= len(dataloader)

    return total_valid_loss, total_valid_map

# Cell
def validate_one_step(model, batch, criterion, device):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    outputs = model(images, labels, return_embeddings=False)
    logits = outputs["logits"]

    step_loss = criterion(logits, labels)

    _, sorted_predictions = torch.sort(logits, descending=True)
    step_map = map_per_set(labels.cpu().tolist(), sorted_predictions.cpu().tolist())

    return step_loss, step_map
