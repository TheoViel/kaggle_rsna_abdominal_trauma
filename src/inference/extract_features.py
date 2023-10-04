import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel

from data.preparation import prepare_data
from data.dataset import AbdominalInfDataset  # Abdominal2DInfDataset
from data.transforms import get_transfos
from data.loader import define_loaders
from model_zoo.models import define_model
from util.metrics import rsna_loss
from util.torch import load_model_weights, sync_across_gpus
from params import DATA_PATH, IMAGE_TARGETS

class Config:
    """
    Placeholder to load a config from a saved json
    """
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def predict(model, dataset, loss_config, batch_size=64, device="cuda", use_fp16=False, num_workers=8):
    """
    Perform inference using a single model and generate predictions for the given dataset.

    Args:
        model (torch.nn.Module): Trained model for inference.
        dataset (torch.utils.data.Dataset): Dataset for which to generate predictions.
        loss_config (dict): Configuration for loss function and activation.
        batch_size (int, optional): Batch size for prediction. Defaults to 64.
        device (str, optional): Device for inference, 'cuda' or 'cpu'. Defaults to 'cuda'.
        use_fp16 (bool, optional): Whether to use mixed-precision (FP16) inference. Defaults to False.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 8.

    Returns:
        np array [N x C]: Predicted probabilities for each class for each sample.
        list: Empty list, placeholder for the auxiliary task.
    """
    model.eval()
    preds, fts = [], []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for img, _, _ in loader:
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, ft = model(img.cuda(), return_fts=True)

            # Get probabilities
            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)
            elif loss_config["activation"] == "patient":
                y_pred[:, :2] = y_pred[:, :2].sigmoid()
                y_pred[:, 2:5] = y_pred[:, 2:5].softmax(-1)
                y_pred[:, 5:8] = y_pred[:, 5:8].softmax(-1)
                y_pred[:, 8:] = y_pred[:, 8:].softmax(-1)

            preds.append(y_pred.detach().cpu().numpy())
            fts.append(ft.detach().cpu().numpy())

    return np.concatenate(preds), np.concatenate(fts)


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
):
    model.eval()
    preds, fts = [], []
    
    loader = define_loaders(
        dataset,
        dataset,
        batch_size=batch_size,
        val_bs=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        world_size=world_size,
        local_rank=local_rank,
    )[1]

    with torch.no_grad():
        for img, _, _ in tqdm(loader, disable=(local_rank!=0)):

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, ft = model(img.cuda(), return_fts=True)

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

    preds = torch.cat(preds, 0)
    fts = torch.cat(fts, 0)

    if distributed:
        fts = sync_across_gpus(fts, world_size)
        preds = sync_across_gpus(preds, world_size)
        torch.distributed.barrier()

    if local_rank == 0:
        preds = preds.cpu().numpy()
        fts = fts.cpu().numpy()
        return preds, fts
    else:
        return 0, 0


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
    config=None
):
    """
    Perform k-fold cross-validation for model inference on the validation set.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        use_fp16 (bool, optional): Whether to use mixed precision inference. Defaults to False.
        save (bool, optional): Whether to save the predictions. Defaults to False.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 8.
        batch_size (int, optional): Batch size. If None, uses the value in the config. Defaults to None.

    Returns:
        List[np.ndarray]: List of arrays containing the predicted probabilities for each class for each fold.
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
            replace_pad_conv=config.replace_pad_conv,
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

        df_val = df_img[df_img['fold'] == fold].reset_index(drop=True)  # if "fold" in df_img.columns else df_img

        transforms = get_transfos(
            augment=False,
            resize=config.resize,  # None if config.use_mask else 
            crop=config.crop,
        )

#         dataset = Abdominal2DInfDataset(
#             df_val,
#             transforms=transforms,
#             frames_chanel=config.frames_chanel if hasattr(config, "frames_chanel") else 0,
#             use_mask=config.use_mask,
#         )
        dataset = AbdominalInfDataset(
            df_val,
            transforms=transforms,
            frames_chanel=config.frames_chanel if hasattr(config, "frames_chanel") else 0,
            n_frames=config.n_frames if hasattr(config, "n_frames") else 1,
            stride=config.stride if hasattr(config, "stride") else 1,
            use_crops=config.use_crops if hasattr(config, "use_crops") else False
        )

        if distributed:
            pred, fts = predict_distributed(
                model,
                dataset,
                config.loss_config,
                batch_size=config.data_config["val_bs"] if batch_size is None else batch_size,
                use_fp16=use_fp16,
                num_workers=num_workers,
                distributed=True,
                world_size=config.world_size,
                local_rank=config.local_rank,
            )
            if config.local_rank == 0:
                pred, fts = pred[:len(dataset)], fts[:len(dataset)]
        else:
            print('\nWarning, this is slow !\n')
            pred, fts = predict(
                model,
                dataset,
                config.loss_config,
                batch_size=config.data_config["val_bs"] if batch_size is None else batch_size,
                use_fp16=use_fp16,
                num_workers=num_workers,
            )

        if save and config.local_rank == 0:
            np.save(exp_folder + f"pred_val_{fold}.npy", pred)
   
            pred_cols = []
            for i, tgt in enumerate(IMAGE_TARGETS):
                df_val[f"pred_{tgt}"] = pred[:len(df_val), i]
                pred_cols.append(f"pred_{tgt}")
            df_val_patient = df_val[['patient_id'] + pred_cols].groupby('patient_id').mean()

            df_val_patient = df_val_patient.merge(
                df_patient[df_patient['fold'] == fold], on="patient_id", how="left"
            )

            print()
            for tgt in IMAGE_TARGETS:
                auc = roc_auc_score(df_val_patient[tgt], df_val_patient[f"pred_{tgt}"])
                print(f'- {tgt} auc : {auc:.3f}')
                
            losses, avg_loss = rsna_loss(
                df_val_patient[["pred_bowel_injury", "pred_extravasation_injury"]].values,
                df_val_patient
            )
            for k, v in losses.items():
                print(f"- {k.split('_')[0][:8]} loss\t: {v:.3f}")
