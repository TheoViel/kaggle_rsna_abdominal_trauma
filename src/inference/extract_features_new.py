import json
import os 
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel

from data.dataset import AbdominalInfDataset
from data.transforms import get_transfos
from data.loader import define_loaders
from model_zoo.models import define_model
from util.metrics import rsna_loss
from util.torch import load_model_weights, sync_across_gpus
from params import IMAGE_TARGETS


class Config:
    """
    Placeholder to load a config from a saved json
    """
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)

def predict_distributed(
    model,
    dataset,
    loss_config,
    batch_size=64,
    use_fp16=False,
    num_workers=8,
    distributed=True,
    world_size=0,
    local_rank=0,
    save_every=10,
):
    """
    Make predictions using a PyTorch model on a given dataset.
    Uses DDP.

    Args:
        model (nn.Module): PyTorch model for making predictions.
        dataset (torch.utils.data.Dataset): Dataset used for prediction.
        loss_config (dict): Configuration for the loss function.
        batch_size (int, optional): Batch size for inference. Defaults to 64.
        use_fp16 (bool, optional): Whether to use mixed-precision (fp16) inference. Defaults to False.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        distributed (bool, optional): Whether to use distributed inference. Defaults to True.
        world_size (int, optional): Number of GPUs used in distributed inference. Defaults to 0.
        local_rank (int, optional): Local rank for distributed inference. Defaults to 0.
        save_every (int, optional): Frequency (in batches) to save results. Defaults to 100.

    Returns:
        Tuple: If `local_rank` is 0, returns a tuple containing:
            - preds (numpy.ndarray): Predictions from the model.
            - fts (numpy.ndarray): Extracted features from the model.
        Otherwise, returns (0, 0) for non-zero `local_rank`.
    """
    model.eval()
    preds, fts = [], []
    batches_processed = 0

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    with torch.no_grad():
        for img, _, _ in tqdm(loader, disable=(local_rank != 0)):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, ft = model(img.cuda(non_blocking=True), return_fts=True)

            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)
            elif loss_config["activation"] == "patient":
                y_pred[:, :2] = y_pred[:, :2].sigmoid()
                y_pred[:, 2:5] = y_pred[:, 2:5].softmax(-1)
                y_pred[:, 5:8] = y_pred[:, 5:8].softmax(-1)
                y_pred[:, 8:] = y_pred[:, 8:].softmax(-1)

            preds.append(y_pred.detach())
            fts.append(ft.detach())
            
            batches_processed += 1

            # Save results periodically
            if batches_processed % save_every == 0:
                preds_result = torch.cat(preds, dim=0)
                fts_result = torch.cat(fts, dim=0)

                if distributed:
                    fts_result = sync_across_gpus(fts_result, world_size)
                    preds_result = sync_across_gpus(preds_result, world_size)
                    torch.distributed.barrier()

                if local_rank == 0:
                    preds_result = preds_result.cpu().numpy()
                    fts_result = fts_result.cpu().numpy()
                    yield preds_result, fts_result

                preds, fts = [], []
                
                # Clear GPU memory
                torch.cuda.empty_cache()

    # Final yield of remaining results
    preds_result = torch.cat(preds, dim=0)
    fts_result = torch.cat(fts, dim=0)

    if distributed:
        fts_result = sync_across_gpus(fts_result, world_size)
        preds_result = sync_across_gpus(preds_result, world_size)
        torch.distributed.barrier()

    if local_rank == 0:
        preds_result = preds_result.cpu().numpy()
        fts_result = fts_result.cpu().numpy()
        yield preds_result, fts_result
    else:
        yield 0, 0
 
def kfold_inference(
    df_patient,
    df_img,
    exp_folder,
    debug=False,
    use_fp16=False,
    save=False,
    num_workers=8,
    batch_size=None,
    distributed=False,
    config=None,
    save_every=10,
):
    """
    Perform k-fold inference on a dataset using a trained model.

    Args:
        df_patient (pd.DataFrame): Dataframe containing patient information.
        df_img (pd.DataFrame): Dataframe containing image information.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): If True, enable debug mode. Defaults to False.
        use_fp16 (bool, optional): Whether to use mixed-precision (fp16) inference. Defaults to False.
        save (bool, optional): If True, save inference results. Defaults to False.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        batch_size (int, optional): Batch size for inference. Defaults to None.
        distributed (bool, optional): Whether to use distributed inference. Defaults to False.
        config (Config, optional): Configuration object for the experiment. Defaults to None.
        save_every (int, optional): Frequency (in batches) to save results. Defaults to 100.
    """
    if config is None:
        config = Config(json.load(open(exp_folder + "config.json", "r")))

    if "fold" not in df_patient.columns:
        folds = pd.read_csv(config.folds_file)
        df_patient = df_patient.merge(folds, how="left")
        df_img = df_img.merge(folds, how="left")

    for fold in config.selected_folds:
        if config.local_rank == 0:
            print(f"\n- Fold {fold + 1}")

        model = define_model(
            config.name,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            use_gem=config.use_gem,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            n_channels=config.n_channels,
            reduce_stride=config.reduce_stride,
            increase_stride=config.increase_stride if hasattr(config, "increase_stride") else False,
            pretrained=False,
        )
        model = model.cuda().eval()

        weights = exp_folder + f"{config.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=config.local_rank == 0)

        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[config.local_rank],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        df_val = df_img[df_img["fold"] == fold].reset_index(drop=True)

        transforms = get_transfos(
            augment=False,
            resize=config.resize,
            crop=config.crop,
        )

        dataset = AbdominalInfDataset(
            df_val,
            transforms=transforms,
            frames_chanel=config.frames_chanel if hasattr(config, "frames_chanel") else 0,
            n_frames=config.n_frames if hasattr(config, "n_frames") else 1,
            stride=config.stride if hasattr(config, "stride") else 1,
        )
        # Clear previous results file if it exists
        if save and config.local_rank == 0:
            preds_file = exp_folder + f"pred_val_{fold}.npy"
            if os.path.exists(preds_file):
                os.remove(preds_file) 

        generator = predict_distributed(
            model,
            dataset,
            config.loss_config,
            batch_size=config.data_config["val_bs"] if batch_size is None else batch_size,
            use_fp16=use_fp16,
            num_workers=num_workers,
            distributed=distributed,
            world_size=config.world_size,
            local_rank=config.local_rank,
            save_every=save_every,
        )

        preds_accumulator, fts_accumulator = [], []
        batches_processed = 0

        for preds, fts in generator:
            preds_accumulator.append(preds)
            fts_accumulator.append(fts)
            batches_processed += 1
            # Save and process results every save_every batches
            if batches_processed % save_every == 0:
                save_batch_results(
                    exp_folder,
                    fold,
                    df_val,
                    df_patient,
                    preds_accumulator,
                    fts_accumulator,
                )
                preds_accumulator, fts_accumulator = [], []

        # Final save of remaining results
        if preds_accumulator:
            save_batch_results(
                exp_folder,
                fold,
                df_val,
                df_patient,
                preds_accumulator,
                fts_accumulator,
            )

def save_batch_results(
    exp_folder,
    fold,
    df_val,
    df_patient,
    preds_accumulator,
    fts_accumulator,
):
    """
    Save batch results to disk.

    Args:
        exp_folder (str): Path to the experiment folder.
        fold (int): Fold number.
        df_val (pd.DataFrame): DataFrame containing validation image information.
        df_patient (pd.DataFrame): DataFrame containing patient information.
        preds_accumulator (list): List of prediction arrays to be saved.
        fts_accumulator (list): List of feature arrays to be saved.
    """
    if not preds_accumulator:
        print("No predictions to save.")
        return
    
    preds_result = np.concatenate(preds_accumulator, axis=0)
    fts_result = np.concatenate(fts_accumulator, axis=0)

    if len(preds_result) != len(df_val):
        print(f"Warning: Length of predictions ({len(preds_result)}) does not match length of df_val ({len(df_val)}).")
        return

    np.save(exp_folder + f"pred_val_{fold}.npy", preds_result)

    pred_cols = []
    for i, tgt in enumerate(IMAGE_TARGETS):
        df_val[f"pred_{tgt}"] = preds_result[: len(df_val), i]
        pred_cols.append(f"pred_{tgt}")
        
    df_val_patient = (
        df_val[["patient_id"] + pred_cols].groupby("patient_id").mean()
    )

    df_val_patient = df_val_patient.merge(
        df_patient[df_patient["fold"] == fold], on="patient_id", how="left"
    )
    

    print()
    for tgt in IMAGE_TARGETS:
        auc = roc_auc_score(df_val_patient[tgt], df_val_patient[f"pred_{tgt}"])
        print(f"- {tgt} auc : {auc:.3f}")

    losses, avg_loss = rsna_loss(
        df_val_patient[
            ["pred_bowel_injury", "pred_extravasation_injury"]
        ].values,
        df_val_patient,
    )
    for k, v in losses.items():
        print(f"- {k.split('_')[0][:8]} loss\t: {v:.3f}")
 
 