from happy_whales.inference import create_and_search_index, get_embeddings, get_best_threshold, get_predictions
from happy_whales.training import HappyWhalesTrainer, get_optimizer, get_scheduler
from happy_whales.utils import load_label_encoder
from happy_whales.models import HappyWhalesModel
from happy_whales.data import HappyWhalesDataset
from happy_whales.losses import ArcFaceLoss
from happy_whales.conf import parse_cfg
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import pickle
import glob

if __name__ == '__main__':
    cfg, _ = parse_cfg()
    
    label_encoder_id = load_label_encoder("label_encoders/id_label_encoder.pickle")
    label_encoder_species = load_label_encoder("label_encoders/species_label_encoder.pickle")
    
    dataset = pd.read_csv(cfg.TRAIN_DATA_ANNOTATION_PATH)

    test_df = pd.DataFrame()
    test_df["image"] = glob.glob(f"{cfg.TEST_DATA_IMG_DIR}/*.jpg")
    
    dataset["individual_id"] = label_encoder_id.transform(dataset["individual_id"])
    dataset["species"] = label_encoder_species.transform(dataset["species"])
    
    if cfg.bbox_path:
        bbox_dict = pickle.load(open(cfg.bbox_path, "rb"))
    
    for fold_i in range(len(dataset["fold"].unique())):
        
        if cfg.fold_to_run is not None:
            fold_i = cfg.fold_to_run
        
        print(f"FOLD: {fold_i}")
        
        train_df = dataset[dataset["fold"] != fold_i].drop("fold", axis=1).reset_index(drop=True)
        valid_df = dataset[dataset["fold"] == fold_i].drop("fold", axis=1).reset_index(drop=True)
        
        new_individual_ids = set(valid_df["individual_id"]) - set(train_df["individual_id"])
        
        train_dataset = HappyWhalesDataset(train_df, transforms=cfg.valid_transforms)
        valid_dataset = HappyWhalesDataset(valid_df, transforms=cfg.valid_transforms)
        test_dataset = HappyWhalesDataset(test_df, transforms=cfg.valid_transforms, mode="inference")
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=12)
        valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=12)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=12)

        model_path = f"{cfg.save_path}/{cfg.model_name}/fold{fold_i}_best_state_dict.pth"
        model = HappyWhalesModel.from_checkpoint(cfg.model_name, model_path, cfg.output_embedding_dim, cfg.num_classes)
        model.to(cfg.device)
        
        print(f"Model loaded from {model_path}")
        
        train_embeddings, train_labels, train_names = get_embeddings(model, train_dataloader, cfg.device, normalization=True, norm="l2")
        valid_embeddings, valid_labels, valid_names = get_embeddings(model, valid_dataloader, cfg.device, normalization=True, norm="l2")
        
        similarities, indexes = create_and_search_index(train_embeddings, valid_embeddings, k=300, embedding_dim=cfg.output_embedding_dim)
        best_threshold = get_best_threshold(indexes, similarities, valid_names, train_labels, valid_labels, label_encoder_id)
        
        full_embeddings = np.concatenate([train_embeddings, valid_embeddings])
        full_labels = np.concatenate([train_labels, valid_labels])
        
        test_embeddings, test_labels, test_names = get_embeddings(model, test_dataloader, cfg.device, normalization=True, norm="l2")

        similarities, indexes = create_and_search_index(full_embeddings, test_embeddings, k=300, embedding_dim=512)
        submission = get_predictions(indexes, similarities, test_names, full_labels, label_encoder_id, threshold=best_threshold)
        
        submission.to_csv(f"submissions/submission.csv", index=False)
        
        if cfg.fold_to_run is not None:
            break