# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/utils.ipynb (unless otherwise specified).

__all__ = ['seed_everything']

# Cell
import numpy as np
import random
import torch
import os

# Cell
def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True