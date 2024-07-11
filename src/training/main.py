import gc
import torch
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from training.train import fit
from model_zoo.models import define_model

from data.dataset import AbdominalDataset
from data.transforms import get_transfos

from util.torch import seed_everything, count_parameters, save_model_weights


def train(
    config, df_train, df_val, df_img_train, df_img_val, fold, log_folder=None, run=None
):
    """
    Train a classification model.

    Args:
        config (Config): Configuration parameters for training.
        df_train (pandas DataFrame): Metadata for training dataset.
        df_val (pandas DataFrame): Metadata for validation dataset.
        df_img_train (pandas DataFrame): Metadata containing image information for training.
        df_img_val (pandas DataFrame): Metadata containing image information for validation.
        fold (int): Fold number for cross-validation.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.

    Returns:
        tuple: A tuple containing predictions and metrics.
    """
    transfos = get_transfos(
        strength=config.aug_strength, resize=config.resize, crop=config.crop
    )
    print("num frames: ", config.n_frames)
    print("frame channel", config.frames_chanel)

    train_dataset = AbdominalDataset(
        df_train,
        df_img_train,
        transforms=transfos,
        frames_chanel=config.frames_chanel,
        n_frames=config.n_frames,
        stride=config.stride,
        train=True,
    )

    transfos = get_transfos(augment=False, resize=config.resize, crop=config.crop)
    val_dataset = AbdominalDataset(
        df_val,
        df_img_val,
        transforms=transfos,
        frames_chanel=config.frames_chanel,
        n_frames=config.n_frames,
        stride=config.stride,
        train=False,
    )

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(
            ".pt"
        ) or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = config.pretrained_weights + f"{config.name}_{fold}.pt"
    else:
        pretrained_weights = None

    model = define_model(
        config.name,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        use_gem=config.use_gem,
        head_3d=config.head_3d,
        n_frames=config.n_frames,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_channels=config.n_channels,
        pretrained_weights=pretrained_weights,
        reduce_stride=config.reduce_stride,
        verbose=(config.local_rank == 0),
    ).cuda()

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
        print(f"    -> {len(train_dataset)} training injuries")
        print(f"    -> {len(val_dataset)} validation injuries")
        print(f"    -> {n_parameters} trainable parameters\n")

    preds, metrics = fit(
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

    return preds, metrics


def k_fold(config, df, df_img, log_folder=None, run=None):
    """
    Perform k-fold cross-validation training for a classification model.

    Args:
        config (dict): Configuration parameters for training.
        df (pandas DataFrame): Main dataset metadata.
        df_img (pandas DataFrame): Metadata containing image information.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.
    """
    folds = pd.read_csv(config.folds_file)
    df = df.merge(folds, how="left")
    df_img = df_img.merge(folds, how="left")

    all_metrics = []
    for fold in range(config.k):
        if fold in config.selected_folds:
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n"
                )
            seed_everything(config.seed + fold)

            df_train = df[df["fold"] != fold].reset_index(drop=True)
            df_val = df[df["fold"] == fold].reset_index(drop=True)

            df_img_train = df_img[df_img["fold"] != fold].reset_index(drop=True)
            df_img_val = df_img[df_img["fold"] == fold].reset_index(drop=True)

            if len(df) <= 1000:
                df_train, df_val = df, df
                df_img_train, df_img_val = df_img, df_img
            print(">>>> code is updated ")
            preds, metrics = train(
                config,
                df_train,
                df_val,
                df_img_train,
                df_img_val,
                fold,
                log_folder=log_folder,
                run=run,
            )
            all_metrics.append(metrics)

            if log_folder is None:
                return

            if config.local_rank == 0:
                np.save(log_folder + f"pred_val_{fold}", preds)
                df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)

    if config.local_rank == 0:
        print("\n-------------   CV Scores  -------------\n")

        for k in all_metrics[0].keys():
            avg = np.mean([m[k] for m in all_metrics])
            print(f"- {k.split('_')[0][:7]} score\t: {avg:.3f}")
            if run is not None:
                run[f"global/{k}"] = avg

        if run is not None:
            run["global/logs"].upload(log_folder + "logs.txt")

        np.save(log_folder + f"pred_val_{fold}", preds)
        df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)

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
                df_img,
                df_img,
                f"fullfit_{ff}",
                log_folder=log_folder,
                run=run,
            )

    if run is not None:
        print()
        run.stop()
