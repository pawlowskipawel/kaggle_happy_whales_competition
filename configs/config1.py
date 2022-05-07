from happy_whales.metrics import TopKAccuracy
import imgaug.augmenters as iaa 

args = {
    # global
    "TRAIN_DATA_ANNOTATION_PATH": "data/stratified_train_folds.csv",
    "TRAIN_DATA_IMG_DIR": "data/custom_train_images",
    "criterion_species": "arcface",
    "save_path": "checkpoints",
    "fold_to_run": None,
    "bbox_path": None, 
    "device": "cuda",
    
    # model parameters
    "model_name": "tf_efficientnet_b3_ns",
    "output_embedding_dim": 512,
    "num_classes": 15587,
    "dropout": 0.3,
    
    # dataset
    "image_shape": (384, 384),
    "normalization": "imagenet",
    
    # training
    "freeze_backbone_batchnorm": True,
    "grad_accum_iter": 6,
    "batch_size": 16,
    "epochs": 40,
    
    # loss
    "label_smothing_eps": 0,
    "arc_m": 0.45,
    "arc_s": 30.0,
    "arc_crit": "crossentropy",
    "criterion_species": None,#"arcface",
    
    # optimizer
    "optimizer_name": "adam",
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    
    # lr scheduler
    "lr_scheduler": "onecyclelr",
    "scheduler_warmup_epochs": 0,
    
    # onecycle_lr
    "max_learning_rate": 0.001,
    "div_factor": 5.0,
    "final_div_factor": 5,
    
    # inference
    "TEST_DATA_IMG_DIR": "data/custom_test_images",
}

args["train_transforms"] = iaa.Sequential([
        iaa.Resize(args["image_shape"]),
        iaa.Fliplr(0.5),
        iaa.Sometimes(0.25, iaa.OneOf([
                iaa.MotionBlur(k=5),
                iaa.AverageBlur(k=(2, 5)),
                iaa.GaussianBlur(sigma=(0.01, 1.5)),
                iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
                iaa.imgcorruptlike.Brightness(severity=(1, 3)),
                iaa.imgcorruptlike.Contrast(severity=(1, 2)),
        ])),
        iaa.Sometimes(0.3, iaa.OneOf([
                iaa.PerspectiveTransform(scale=(0.01, 0.08)),
                iaa.TranslateY((0.01, 0.1)),
                iaa.TranslateX((0.01, 0.1)),
                iaa.ShearX((-10, 10)),
                iaa.ShearY((-10, 10)),
                iaa.Rotate((-25, 25)),
        ])),
])

args["valid_transforms"] = iaa.Sequential([
        iaa.Resize(args["image_shape"])
])

args["metrics_dict"] = {
        "top5accuracy": TopKAccuracy(k=5),
        "top1accuracy": TopKAccuracy(k=1)
}