import gc
import glob

import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel

from data.dataset import Seg3dDataset, SegDataset
from data.transforms import get_transfos
from model_zoo.models import define_model
from model_zoo.models_seg import define_model as define_model_seg
from training.train import fit
from training.train_seg import fit as fit_seg
from util.torch import count_parameters, save_model_weights, seed_everything


# TODO dot this or that


def train(config, df_train, df_val, fold, log_folder=None, run=None):
    """
    Train a segmentation model.

    Args:
        config (Config): Configuration parameters for training.
        df_train (pandas DataFrame): Metadata for training dataset.
        df_val (pandas DataFrame): Metadata for validation dataset.
        fold (int): Fold number for cross-validation.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.

    Returns:
        tuple: A tuple containing predictions and metrics.
    """
    if not config.use_3d:
        train_dataset = SegDataset(
            df_train,
            transforms=get_transfos(strength=config.aug_strength, resize=config.resize),
            for_classification=config.for_classification,
        )

        val_dataset = SegDataset(
            df_val,
            transforms=get_transfos(augment=False, resize=config.resize),
            for_classification=config.for_classification,
        )
    else:
        train_dataset = Seg3dDataset(df_train, train=True)
        val_dataset = Seg3dDataset(df_val)

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(
            ".pt"
        ) or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[
                0
            ]
    else:
        pretrained_weights = None

    if config.for_classification:
        model = define_model(
            config.name,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            use_gem=config.use_gem,
            replace_pad_conv=config.replace_pad_conv,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            n_channels=config.n_channels,
            pretrained_weights=pretrained_weights,
            reduce_stride=config.reduce_stride,
            increase_stride=config.increase_stride,
            verbose=(config.local_rank == 0),
        ).cuda()
    else:
        model = define_model_seg(
            config.decoder_name,
            config.name,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            increase_stride=config.increase_stride,
            use_cls=config.use_cls,
            n_channels=config.n_channels,
            pretrained_weights=pretrained_weights,
            use_3d=config.use_3d,
            verbose=(config.local_rank == 0),
        )
        model = model.cuda()

    if config.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    model.zero_grad(set_to_none=True)
    model.train()

    n_parameters = count_parameters(model)
    if config.local_rank == 0:
        print(f"    -> {len(train_dataset)} training images")
        print(f"    -> {len(val_dataset)} validation images")
        print(f"    -> {n_parameters} trainable parameters\n")

    fit_fct = fit if config.for_classification else fit_seg
    _ = fit_fct(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        epochs=config.epochs,
        verbose_eval=config.verbose_eval,
        use_fp16=config.use_fp16,
        distributed=config.distributed,
        local_rank=config.local_rank,
        world_size=config.world_size,
        log_folder=log_folder,
        run=run,
        fold=fold,
    )

    if (log_folder is not None) and (config.local_rank == 0):
        save_model_weights(
            model.module if config.distributed else model,
            f"{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()


def k_fold(config, df, log_folder=None, run=None):
    """
    Perform k-fold cross-validation training for a segmentation model.

    Args:
        config (dict): Configuration parameters for training.
        df (pandas DataFrame): Main dataset metadata.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.
    """
    folds = pd.read_csv(config.folds_file)
    df = df.merge(folds, how="left")

    for fold in range(config.k):
        if fold in config.selected_folds:
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n"
                )
            seed_everything(config.seed + fold)

            df_train = df[df["fold"] != fold].reset_index(drop=True)
            df_val = df[df["fold"] == fold].reset_index(drop=True)

            if not config.use_3d:
                df_val = df_val[
                    df_val[[c for c in df_val.columns if "norm" in c]].max(1) > 0.1
                ].reset_index(drop=True)

            train(config, df_train, df_val, fold, log_folder=log_folder, run=run)

            if log_folder is None or config.pretrain:
                return

    if config.fullfit and len(config.selected_folds) == 4:
        for ff in range(config.n_fullfit):
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                )
            seed_everything(config.seed + ff)

            train(
                config,
                df,
                df.tail(100).reset_index(drop=True),
                f"fullfit_{ff}",
                log_folder=log_folder,
                run=run,
            )

    if run is not None:
        print()
        run.stop()
