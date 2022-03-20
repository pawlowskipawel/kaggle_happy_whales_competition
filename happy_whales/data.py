# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/data.ipynb (unless otherwise specified).

__all__ = ['extract_top_n_classes', 'HappyWhalesDataset']

# Cell
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import os

# Cell
def extract_top_n_classes(df, num_classes):
    df = df.copy(deep=True)

    top_n_classes = df["individual_id"].value_counts().keys()[:num_classes]
    df = df[df["individual_id"].isin(top_n_classes)]

    mapping = {c: m for m, c in enumerate(df["individual_id"].unique())}
    df["individual_id"] = df["individual_id"].map(mapping)

    return df

# Cell
class HappyWhalesDataset(Dataset):
    def __init__(self, df, image_shape, bbox_dict=None, normalization='imagenet'):
        super().__init__()

        self.df = df
        self.normalization = normalization
        self.num_examples = len(df.index)
        self.image_shape = image_shape
        self.bbox_dict = bbox_dict

    def _normalize_image(self, image):
        image = (image / 255.).astype(np.float32)

        if self.normalization == "imagenet":
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            image[0] = (image[0] - means[0]) / stds[0]
            image[1] = (image[1] - means[1]) / stds[2]
            image[2] = (image[2] - means[2]) / stds[1]

        return image.astype(np.float32)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):

        image_path = self.df.iloc[index, 0]
        image_name = os.path.split(image_path)[-1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.bbox_dict:
            bbox = self.bbox_dict[image_name]
            xmin, ymin, xmax, ymax = bbox
            image = image[ymin: ymax, xmin: xmax]

        image = cv2.resize(image, self.image_shape)
        image = self._normalize_image(image)

        label = self.df.iloc[index, 2]

        return {
            "image": torch.tensor(image, dtype=torch.float).permute(2, 0, 1),
            "label": torch.tensor(label, dtype=torch.long)
        }