import pandas as pd
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import imgaug.augmenters as iaa
import torch.optim as optim
import torch.nn as nn
import torch

from happy_whales.data import *
from happy_whales.utils import *
from happy_whales.models import *
from happy_whales.metrics import *
from happy_whales.training import *

import pickle
import h5py

class cfg:
    device = "cuda"
    epochs = 10
    batch_size = 8
    grad_accum_iter = 8
    num_classes = 1000
    model_name = "tf_efficientnet_b0"
    image_shape = (224, 224)
    learning_rate = 0.0001
    minimum_learning_rate = 1e-7
    embedding_dim = 1280
    scheduler_step = 1000
    find_lr = True
    
if __name__ == '__main__':
    seed_everything()
    
    TRAIN_HDF5_IMGS = None#"data/train_456.h5"
    TRAIN_DATA_IMG_DIR = "data/train_images"
    TRAIN_DATA_ANNOTATION_PATH = "data/train.csv"
    
    hdf5_data = h5py.File(TRAIN_HDF5_IMGS, 'r') if TRAIN_HDF5_IMGS else None
        
    train_transforms = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Sometimes(0.3, iaa.Rotate((-10, 10))),
        iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.01, 0.15)))
    ])
     
    dataset = pd.read_csv(TRAIN_DATA_ANNOTATION_PATH)
    
    dataset["species"] = fix_species(dataset["species"])
    dataset = duplicate_ones(dataset)
    
    dataset["image"] = TRAIN_DATA_IMG_DIR + '/' + dataset["image"]
    
    dataset['individual_id'] = fit_transform_and_dump_label_encoder(dataset['individual_id'], "id_label_encoder", dest_dir="label_encoders")
    dataset['species'] = fit_transform_and_dump_label_encoder(dataset['species'], "species_label_encoder", dest_dir="label_encoders")
    
    bbox_dict = pickle.load(open("data/bbox_train.pickle", "rb"))
    
    dataset = extract_top_n_classes(dataset, cfg.num_classes)
    
    train_df, valid_df = train_valid_split(dataset)
    
    train_dataset = HappyWhalesDataset(train_df, cfg.image_shape, transforms=train_transforms, bbox_dict=bbox_dict, hdf5=hdf5_data)
    valid_dataset = HappyWhalesDataset(valid_df, cfg.image_shape, bbox_dict=bbox_dict, hdf5=hdf5_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=16, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=16, shuffle=False)

    model = HappyWhalesModel(cfg.model_name, cfg.embedding_dim, cfg.num_classes).to(cfg.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.000001)
         
    lr_scheduler = ReduceLROnPlateau(optimizer, min_lr=cfg.minimum_learning_rate, verbose=True, patience=5)
    
    best_valid_map = 0
    best_valid_loss = np.inf
    
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            epoch=epoch, 
            model=model, 
            criterion=criterion, 
            optimizer=optimizer, 
            train_dataloader=train_dataloader, 
            grad_accum_iter=cfg.grad_accum_iter, 
            valid_dataloader=valid_dataloader, 
            lr_scheduler=None, 
            device=cfg.device
        )
        
        valid_loss, valid_map = validate_one_epoch(
            epoch=epoch, 
            model=model, 
            valid_dataloader=valid_dataloader, 
            criterion=criterion, 
            device=cfg.device
        )
        
        if valid_map > best_valid_map:
            best_valid_map = valid_map
            
            torch.save(model.state_dict(), f"best_map_model.pth")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            torch.save(model.state_dict(), f"best_loss_model.pth")
    
    if hdf5_data:
        hdf5_data.close()