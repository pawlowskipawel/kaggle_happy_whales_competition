from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import timm

from tqdm import tqdm

from happy_whales.data import *
from happy_whales.models import *
from happy_whales.metrics import *

import random
import glob
import os


def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def fix_species(species_col):
    return species_col.replace({"globis": "short_finned_pilot_whale",
                                "pilot_whale": "short_finned_pilot_whale",
                                "kiler_whale": "killer_whale",
                                "bottlenose_dolpin": "bottlenose_dolphin"})

def train_epoch(epoch, model, criterion, optimizer, train_dataloader, valid_dataloader=None, lr_scheduler=None, amp=False, scaler=None, device="cuda"):
    
    assert (amp and scaler) or not (amp and scaler), "Wrong setting amp/scaler!"
    
    model.train()
    
    batch_losses = []
    
    with tqdm(train_dataloader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}') as progress_bar:
        progress_bar.set_description(f"Epoch {epoch+1}".ljust(25))
        
        for step, batch in enumerate(progress_bar, 1):
            
            # More efficient than optimizer.zero_grad()            
            for p in model.parameters():
                p.grad = None
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            #images = images.permute(0, 3, 1, 2)
            
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images, labels, return_embeddings=True)
                    
                    embeddings = outputs["embeddings"]
                    logits = outputs["logits"]
                    loss = criterion(logits, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                outputs = model(images, return_embeddings=True)
                
                embeddings = outputs["embeddings"]
                logits = outputs["logits"]
                
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
            
            batch_losses.append(loss.item())
            
            progress_bar.set_postfix({"train loss": np.array(batch_losses).mean()})
            
            if valid_dataloader and lr_scheduler and (step % cfg.scheduler_step) == 0:
                val_loss, val_accuracy = validate(epoch, model, valid_dataloader, criterion, amp=False, scaler=None, disable_bar=True, device=cfg.device)
                lr_scheduler.step(val_loss)
                model.train()
                
        epoch_loss = np.array(batch_losses).mean()
    
    return epoch_loss

@torch.no_grad()
def validate(epoch, model, dataloader, criterion, amp=False, scaler=None, disable_bar=False, device="cuda"):
    model.eval()
    
    
    val_batch_losses = []
    val_batch_maps = []
    
    with tqdm(dataloader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}', disable=disable_bar) as progress_bar:
        progress_bar.set_description(f"Validation after epoch {epoch+1}".ljust(25))
        for batch in progress_bar:
            
            # More efficient than optimizer.zero_grad()            
            for p in model.parameters():
                p.grad = None
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images, labels, return_embeddings=True)
                    
                    embeddings = outputs["embeddings"]
                    logits = outputs["logits"]
                    
                    loss = criterion(logits, labels)
                
                loss = scaler.scale(loss)
                
            else:
                outputs = model(images, labels, return_embeddings=True)
                
                embeddings = outputs["embeddings"]
                logits = outputs["logits"]
                
                loss = criterion(logits, labels)
                
            sorted_outputs, sorted_predictions = torch.sort(logits, descending=True)
            
            val_batch_losses.append(loss.item())
            
            val_batch_maps.append(map_per_set(labels.cpu().tolist(), sorted_predictions.cpu().tolist()))
                        
            progress_bar.set_postfix({"validation loss": np.array(val_batch_losses).mean(), "validation map@5": np.array(val_batch_maps).mean()})
        
        val_loss = np.array(val_batch_losses).mean()
        val_accuracy = np.array(val_batch_maps).mean()
    
    return val_loss, val_accuracy


class cfg:
    device = "cuda"
    epochs = 10
    batch_size = 128
    num_classes = 10
    model_name = "efficientnet_b0"
    image_shape = (224, 224)
    learning_rate = 0.001
    embedding_dim = 1280
    scheduler_step = 5

if __name__ == '__main__':
    seed_everything()
    
    TRAIN_DATA_IMG_DIR = "data/train_images_512x512"
    TRAIN_DATA_ANNOTATION_PATH = "data/train.csv"
    
    dataset = pd.read_csv(TRAIN_DATA_ANNOTATION_PATH)
    dataset["species"] = fix_species(dataset["species"])

    dataset["image"] = TRAIN_DATA_IMG_DIR + '/' + dataset["image"]
    
    label_encoder = LabelEncoder()
    dataset['individual_id'] = label_encoder.fit_transform(dataset['individual_id'])
    
    dataset = extract_top_n_classes(dataset, cfg.num_classes)
    
    train_df, valid_df = train_test_split(dataset, test_size=.15)
    
    train_dataset = HappyWhalesDataset(train_df, cfg.image_shape)
    valid_dataset = HappyWhalesDataset(valid_df, cfg.image_shape)
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=16)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, pin_memory=True, num_workers=16)

    model = HappyWhalesModel(cfg.model_name, cfg.embedding_dim, cfg.num_classes).to(cfg.device)
    
    criterion = nn.CrossEntropyLoss()#ArcFaceLoss(crit="bce")
    optimizer = optim.AdamW([{'params': model.parameters()}, 
                             {'params': criterion.parameters()}], 
                            lr=cfg.learning_rate)
    
    lr_scheduler = ReduceLROnPlateau(optimizer, min_lr=.00000001, verbose=True, patience=5)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(cfg.epochs):
        loss = train_epoch(epoch, model, criterion, optimizer, train_dataloader, valid_dataloader, lr_scheduler, amp=True, scaler=scaler, device=cfg.device)
        val_loss, val_accuracy = validate(epoch, model, valid_dataloader, criterion, amp=False, scaler=None, device=cfg.device)
    