import gc
import glob
import json
import torch
import operator
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from training.train import fit
from model_zoo.models_lvl2 import define_model

from data.dataset import PatientFeatureDataset

from util.torch import seed_everything, count_parameters, save_model_weights


def train(config, df_train, df_val, df_img_train, df_img_val, fold, log_folder=None, run=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
        run (neptune.Run): Nepture run. Defaults to None.

    Returns:
        dict: Dice scores at different thresholds.
    """
    train_dataset = PatientFeatureDataset(
        df_train,
        df_img_train,
        config.exp_folders,
        max_len=config.max_len,
    )

    val_dataset = PatientFeatureDataset(
        df_val,
        df_img_val,
        config.exp_folders,
        max_len=config.max_len,
    )

    model = define_model(
        config.name,
        ft_dim=config.ft_dim,
        layer_dim=config.layer_dim,
        dense_dim=config.dense_dim,
        p=config.p,
        use_msd=config.use_msd,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_fts=config.n_fts,
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
        print(f"    -> {len(train_dataset)} training images")
        print(f"    -> {len(val_dataset)} validation images")
        print(f"    -> {n_parameters} trainable parameters\n")

    preds, preds_aux = fit(
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

    return preds, preds_aux


def k_fold(config, df, df_img, df_extra=None, log_folder=None, run=None):
    """
    Trains a k-fold.

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        df_extra (pandas dataframe or None, optional): Extra metadata. Defaults to None.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
        run (None or Nepture run): Nepture run. Defaults to None.

    Returns:
        dict: Dice scores at different thresholds.
    """
    folds = pd.read_csv(config.folds_file)
    df = df.merge(folds, how="left")
    df_img = df_img.merge(folds, how="left")
    
#     pred_oof, pred_oof_aux = [], []
#     pred_oof = np.zeros((len(df), config.num_classes))
#     pred_oof_aux = np.zeros((len(df), config.num_classes_aux))
    for fold in range(config.k):
        if fold in config.selected_folds:
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n"
                )
            seed_everything(config.seed + fold)

            df_train = df[df['fold'] != fold].reset_index(drop=True)
            df_val = df[df['fold'] == fold].reset_index(drop=True)
            val_idx = list(df[df["fold"] == fold].index)
            
            df_img_train = df_img[df_img['fold'] != fold].reset_index(drop=True)
            df_img_val = df_img[df_img['fold'] == fold].reset_index(drop=True)

            if len(df) <= 1000:
                df_train, df_val = df, df
                df_img_train, df_img_val = df_img, df_img

            preds, preds_aux = train(
                config, df_train, df_val, df_img_train, df_img_val, fold, log_folder=log_folder, run=run
            )
            
            if log_folder is None:
                return preds, preds_aux 

            if config.local_rank == 0:
                np.save(log_folder + f"pred_val_{fold}", preds)
                df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)

#                 pred_oof[val_idx] = preds
#                 if config.num_classes_aux:
#                     pred_oof_aux[val_idx] = preds_aux

#                 if run is not None:
#                     run[f"fold_{fold}/pred_val"].upload(
#                         log_folder + f"df_val_{fold}.csv"
#                     )

#     if config.local_rank == 0:
#         np.save(log_folder + "pred_oof", pred_oof)
#         np.save(log_folder + "pred_oof_aux", pred_oof_aux)

                
#     if config.local_rank == 0 and len(config.selected_folds):
#         print(f"\n\n -> CV Dice : {dice:.3f}  -  th : {th:.2f}")

#         if run is not None:
#             run["global/logs"].upload(log_folder + "logs.txt")
#             run["global/cv"] = dice
#             run["global/th"] = th

    if config.fullfit:
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

    return pred_oof, pred_oof_aux
