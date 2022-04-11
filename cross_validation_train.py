from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CyclicLR
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import imgaug.augmenters as iaa
import torch.optim as optim
import torch.nn as nn
import torch

import pandas as pd
import numpy as np

from happy_whales.conf import *
from happy_whales.data import *
from happy_whales.utils import *
from happy_whales.models import *
from happy_whales.losses import *
from happy_whales.metrics import *
from happy_whales.training import *

import pickle
import random
import sys
import os
import gc

sys.path.append("./configs")

def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':
    seed_everything()
    cfg = parse_cfg()

    label_encoder_id = load_label_encoder("label_encoders/id_label_encoder.pickle")
    label_encoder_species = load_label_encoder("label_encoders/species_label_encoder.pickle")
    
    dataset = pd.read_csv(cfg.TRAIN_DATA_ANNOTATION_PATH)
    dataset["species"] = fix_species(dataset["species"])
    dataset["image"] = cfg.TRAIN_DATA_IMG_DIR + '/' + dataset["image"]
    
    dataset["individual_id"] = label_encoder_id.transform(dataset["individual_id"])
    dataset["species"] = label_encoder_species.transform(dataset["species"])
    
    dataset = extract_top_n_classes(dataset, cfg.num_classes)

    if cfg.bbox_path:
        bbox_dict = pickle.load(open(cfg.bbox_path, "rb"))
    
    for fold_i in range(len(dataset["fold"].unique())):
            
        print(f"FOLD: {fold_i}")
        cfg.output_prefix = f"fold{fold_i}_{cfg.output_prefix_base}"
        
        train_df = dataset[dataset["fold"] != fold_i].drop("fold", axis=1).reset_index(drop=True)
        valid_df = dataset[dataset["fold"] == fold_i].drop("fold", axis=1).reset_index(drop=True)
        
        train_df.to_csv(f"data/train_df_fold_{fold_i}.csv", index=False)
        valid_df.to_csv(f"data/valid_df_fold_{fold_i}.csv", index=False)
        
        new_individual_ids = set(valid_df["individual_id"]) - set(train_df["individual_id"])
        valid_df = valid_df[~valid_df["individual_id"].isin(new_individual_ids)]
        
        train_dataset = HappyWhalesDataset(train_df, transforms=cfg.train_transforms)
        valid_dataset = HappyWhalesDataset(valid_df, transforms=cfg.valid_transforms)
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=12, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=12, shuffle=False)

        if cfg.from_checkpoint:
            model = HappyWhalesModel.from_checkpoint(cfg.model_name, cfg.checkpoint_path, cfg.output_embedding_dim, cfg.num_classes)
            print(f"Model loaded from {cfg.checkpoint_path}")
        else:
            model = HappyWhalesModel(cfg.model_name, cfg.output_embedding_dim, cfg.num_classes, dropout=cfg.dropout)
            
        model.to(cfg.device)
        
        criterion = ArcFaceLoss(s=cfg.s, 
                                m=cfg.m, 
                                crit=cfg.crit)
        
        species_criterion = None#ArcFaceLoss(s=cfg.s, 
                                       # m=cfg.m, 
                                       # crit=cfg.crit)
        
        optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), 
                               lr=cfg.learning_rate, 
                               weight_decay=cfg.weight_decay)
        
        #optimizer = optim.SGD([{'params': model.parameters()}, {'params': criterion.parameters()}], 
        #                      lr=cfg.learning_rate, 
        #                      momentum=0.9, 
        #                      nesterov=True, 
        #                      weight_decay=cfg.weight_decay)
        
        
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                       num_warmup_steps=len(train_dataloader), 
                                                       num_training_steps=int(len(train_dataloader) * (cfg.epochs)))
        
        best_valid_loss = np.inf
        best_train_loss = np.inf
        if cfg.output_prefix is not None:
            output_path = f"{cfg.output_prefix}_{cfg.model_name}_best_loss.pth"
        else:
            output_path = f"{cfg.model_name}_best_loss.pth"
            
        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(
                epoch=epoch, 
                model=model, 
                criterion=criterion,
                species_criterion=species_criterion,
                optimizer=optimizer, 
                train_dataloader=train_dataloader, 
                grad_accum_iter=cfg.grad_accum_iter, 
                valid_dataloader=valid_dataloader, 
                lr_scheduler=lr_scheduler, 
                device=cfg.device
            )
            
            valid_loss = validate_one_epoch(
                epoch=epoch, 
                model=model, 
                dataloader=valid_dataloader, 
                criterion=criterion, 
                species_criterion=species_criterion,
                device=cfg.device
            )
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                
                torch.save(model.state_dict(), f"trained_models/valid_{output_path}")
            
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                
                torch.save(model.state_dict(), f"trained_models/train_{output_path}")
            
        del model, criterion, optimizer, lr_scheduler
        torch.cuda.empty_cache()
        gc.collect()
        print()
        break
    