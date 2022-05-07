from happy_whales.training import HappyWhalesTrainer, get_optimizer, get_scheduler
from happy_whales.utils import load_label_encoder
from happy_whales.models import HappyWhalesModel
from happy_whales.data import HappyWhalesDataset
from happy_whales.losses import ArcFaceLoss
from happy_whales.conf import parse_cfg
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import random
import pickle
import torch
import wandb
import os
import gc


def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    seed_everything()
    cfg, wandb_log = parse_cfg()
    
    label_encoder_id = load_label_encoder("label_encoders/id_label_encoder.pickle")
    label_encoder_species = load_label_encoder("label_encoders/species_label_encoder.pickle")
    
    df = pd.read_csv(cfg.TRAIN_DATA_ANNOTATION_PATH)
    bbox_dict = pickle.load(open(cfg.bbox_path, "rb")) if cfg.bbox_path is not None else None
    
    df["individual_id"] = label_encoder_id.transform(df["individual_id"])
    df["species"] = label_encoder_species.transform(df["species"])
    
    loss_list = []
    
    for fold_i in range(len(df["fold"].unique())): 
        if cfg.fold_to_run is not None:
            fold_i = cfg.fold_to_run
        
        print(f"---- FOLD {fold_i} ----")
        
        train_df = df[df["fold"] != fold_i].drop("fold", axis=1)
        valid_df = df[df["fold"] == fold_i].drop("fold", axis=1)
        
        train_dataset = HappyWhalesDataset(
            df=train_df, 
            transforms=cfg.train_transforms,
            mode="train_val", 
            normalization=cfg.normalization,
            bbox_dict=bbox_dict
        )
        
        valid_dataset = HappyWhalesDataset(
            df=valid_df,
            transforms=cfg.valid_transforms,
            mode="train_val",
            normalization=cfg.normalization,
            bbox_dict=bbox_dict
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=12, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=12, shuffle=False)
        
        model = HappyWhalesModel(
            model_name=cfg.model_name,
            output_embedding_dim=cfg.output_embedding_dim, 
            num_classes=cfg.num_classes, 
            dropout=cfg.dropout, 
            freeze_backbone_batchnorm=cfg.freeze_backbone_batchnorm,
            add_species_head=bool(cfg.criterion_species)
        )
        model.to(cfg.device)
        
        if wandb_log:
            wandb.watch(model)
        
        optimizer = get_optimizer(
            optimizer_name=cfg.optimizer_name,
            model=model,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        criterion = ArcFaceLoss(s=cfg.arc_s, m=cfg.arc_m, crit=cfg.arc_crit)
        criterion_species = ArcFaceLoss(s=cfg.arc_s, m=cfg.arc_m, crit=cfg.arc_crit) if cfg.criterion_species is not None else None
        
        lr_scheduler = get_scheduler(cfg, optimizer, len(train_dataloader))
        
        trainer = HappyWhalesTrainer(
            model_name=cfg.model_name, 
            model=model, 
            criterion=criterion, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            metrics_dict=cfg.metrics_dict, 
            grad_accum_iter=cfg.grad_accum_iter, 
            wandb_log=wandb_log, 
            criterion_species=criterion_species, 
            device=cfg.device
        )
        
        fold_loss = trainer.fit(
            epochs=cfg.epochs,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            save_path=cfg.save_path,
            fold_i=fold_i
        )
        
        loss_list.append(fold_loss)
        
        del model, criterion, optimizer, lr_scheduler, trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        if cfg.fold_to_run is not None:
            break
    
    if wandb_log:
        wandb.run.summary["cv_loss"] = np.array(loss_list).mean()