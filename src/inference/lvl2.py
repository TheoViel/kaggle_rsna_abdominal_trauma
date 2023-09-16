import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader


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
                y_pred, _ = model(img.cuda())

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

    return np.concatenate(preds)


def to_sub_format(df):
    new_df = df[['patient']].copy()
    new_df.columns = ["patient_id"]

    new_df['bowel_healthy'] = 1 - df['pred_0']
    new_df['bowel_injury'] = df['pred_0']

    new_df['extravasation_healthy'] = 1 - df['pred_1']
    new_df['extravasation_injury'] = df['pred_1']
    
    new_df['kidney_healthy'] = df['pred_2']
    new_df['kidney_low'] = df['pred_3']
    new_df['kidney_high'] = df['pred_4']
    
    new_df['liver_healthy'] = df['pred_5']
    new_df['liver_low'] = df['pred_6']
    new_df['liver_high'] = df['pred_7']
    
    new_df['spleen_healthy'] = df['pred_8']
    new_df['spleen_low'] = df['pred_9']
    new_df['spleen_high'] = df['pred_10']
    
    return new_df
