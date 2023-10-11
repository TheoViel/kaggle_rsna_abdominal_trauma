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
from util.metrics import rsna_loss
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
        resize=config.resize,
        restrict=config.restrict,
        use_other_series=config.use_other_series,
        refine_target=config.refine_target,
    )

    val_dataset = PatientFeatureDataset(
        df_val,
        df_img_val,
        config.exp_folders,
        max_len=config.max_len,
        resize=config.resize,
        restrict=config.restrict,
        use_other_series=config.use_other_series,
    )

    model = define_model(
        config.name,
        ft_dim=config.ft_dim,
        layer_dim=config.layer_dim,
        n_layers=config.n_layers,
        dense_dim=config.dense_dim,
        p=config.p,
        use_msd=config.use_msd,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_fts=config.n_fts,
        use_other_series=config.use_other_series,
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
        print(f"    -> {len(train_dataset)} training studies")
        print(f"    -> {len(val_dataset)} validation studies")
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
    
    pred_oof, pred_oof_aux = [], []
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

    if config.local_rank == 0:
        df_oof, pred_oof = retrieve_preds(df, df_img, config, log_folder)
        
        np.save(log_folder + "pred_oof.npy", pred_oof)
        df_oof.to_csv(log_folder + "df_oof.csv", index=False)

        losses, avg_loss = rsna_loss(pred_oof, df_oof)

        print()
        for k, v in losses.items():
            print(f"- {k.split('_')[0][:8]} loss\t: {v:.3f}")
        print(f'\n -> CV Score : {avg_loss :.3f}')

        if run is not None:
            run["global/logs"].upload(log_folder + "logs.txt")
            run["global/cv"] = avg_loss
            for k, v in losses.items():
                run[f"global/{k}"] = v

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


def retrieve_preds(df_patient, df_img, config, exp_folder, custom_agg=False, folds=None):
    dfs = []
    for fold in config.selected_folds if folds is None else folds:

        df_val = df_patient[df_patient['fold'] == fold]

        dataset = PatientFeatureDataset(df_val, df_img[df_img['fold'] == fold], [])
        patients = [d[0] for d in dataset.ids]
        df_preds = pd.DataFrame({"patient_id": patients})

        preds = np.load(exp_folder + f"pred_val_{fold}.npy")

        preds_cols = []
        for i in range(preds.shape[1]):
            preds_cols.append(f'pred_{i}')
            df_preds[f'pred_{i}'] = preds[:, i]

        if custom_agg:
            df_preds_avg = df_preds.groupby('patient_id').mean()
            df_preds_max = df_preds.groupby('patient_id').max()
            
            df_preds = df_preds_avg.copy()
            df_preds['pred_0'] = df_preds_max['pred_0']
#             df_preds['pred_3'] = df_preds_max['pred_3']
#             df_preds['pred_4'] = df_preds_max['pred_4']
#             df_preds['pred_2'] = 1 - df_preds['pred_3'] - df_preds['pred_4']
            
#             df_preds['pred_6'] = df_preds_max['pred_6']
#             df_preds['pred_7'] = df_preds_max['pred_7']
#             df_preds['pred_5'] = 1 - df_preds['pred_6'] - df_preds['pred_7']
            
#             df_preds['pred_9'] = df_preds_max['pred_9']
#             df_preds['pred_10'] = df_preds_max['pred_10']
#             df_preds['pred_8'] = 1 - df_preds['pred_10'] - df_preds['pred_9']
        else:
            df_preds = df_preds.groupby('patient_id').mean()
        df = df_val.merge(df_preds, on="patient_id")
        
        
        dfs.append(df)

    df_oof = pd.concat(dfs, ignore_index=True)
    pred_oof = df_oof[preds_cols].values

    return df_oof, pred_oof
