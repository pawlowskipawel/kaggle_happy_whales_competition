from happy_whales.data import get_folds, fix_species
import pandas as pd
import argparse
import random
import os


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--df', type=str, default='data/train.csv')
    argparser.add_argument('--n-splits', type=int, default=5)
    argparser.add_argument('--stratify', type=str, default=None)

    return argparser.parse_args()


def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
if __name__ == '__main__':
    seed_everything()
    args = parse_args()

    dataset = pd.read_csv(args.df)
    dataset["species"] = fix_species(dataset["species"])
    
    folds = get_folds(dataset, n_splits=args.n_splits, stratify=args.stratify)
    
    output = "stratified_train_folded.csv" if args.stratify else "train_folded.csv"
    
    folds.to_csv(f"data/{output}", index=False)