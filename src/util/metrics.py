import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from params import PATIENT_TARGETS


WEIGHTS = {
    "bowel_injury": {0: 1, 1: 2},
    "extravasation_injury": {0: 1, 1: 6},
    "kidney": {0: 1, 1: 2, 2: 4},
    "liver": {0: 1, 1: 2, 2: 4},
    "spleen": {0: 1, 1: 2, 2: 4},
    "any_injury": {0: 1, 1: 6},
}


def rsna_loss(preds, truths, eps=1e-6):
    """
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.

    preds: list of the 5 predictions probabilities
    truths: df
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.clone().cpu().numpy()

    preds = preds.copy()
    if isinstance(preds, np.ndarray):
        if preds.shape[-1] == 11:
            preds = [
                preds[:, 0],
                preds[:, 1],
                preds[:, 2:5],
                preds[:, 5:8],
                preds[:, 8:],
            ]
        elif preds.shape[-1] == 2:
            preds = [preds[:, 0], preds[:, 1]]

    for i, pred in enumerate(preds):
        if pred.shape[-1] in [2, 3]:
            preds[i] = pred / pred.sum(1, keepdims=True)

    losses = {}
    for i, tgt in enumerate(WEIGHTS.keys()):
        if tgt == "any_injury":
            injury_preds = np.concatenate(
                [
                    1 - p[:, :1] if p.shape[-1] in [2, 3] else p.reshape(-1, 1)
                    for p in preds
                ],
                axis=1,
            )
            pred = injury_preds.max(-1)
        #             print(pred)
        else:
            pred = preds[i]

        truth = truths[tgt].values
        sample_weight = truths[tgt].map(WEIGHTS[tgt]).values

        #         print(tgt, sample_weight)
        #         print(pred.shape)

        if pred.shape[-1] in [2, 3]:  # softmax
            #             pred = pred / pred.sum(1, keepdims=True)
            labels = np.arange(pred.shape[-1])
        else:  # sigmoid was used, we have p of injury
            labels = [0, 1]

        loss = log_loss(truth, pred, sample_weight=sample_weight, labels=labels)
        losses[tgt] = loss

        if len(preds) == 2:
            if tgt == "extravasation_injury":
                break

    return losses, np.mean(list(losses.values()))


def rsna_score_study(preds, dataset, eps=1e-6):
    """
    Calculate the RSNA score for a study-level classification task.

    Args:
        preds (numpy.ndarray): Model predictions for study-level targets.
        dataset (PatientFeatureDataset): Dataset containing patient information.
        eps (float, optional): A small value to prevent division by zero. Defaults to 1e-6.

    Returns:
        float: RSNA score for the study-level classification task.
    """
    preds = preds.astype(np.float64)
    patients = [d[0] for d in dataset.ids]
    df_preds = pd.DataFrame({"patient_id": patients})

    preds_cols = []
    for i in range(preds.shape[1]):
        preds_cols.append(f"pred_{i}")
        df_preds[f"pred_{i}"] = preds[:, i]

    df_preds = df_preds.groupby("patient_id").mean()

    df = dataset.df_patient.merge(df_preds, on="patient_id")
    preds = df[preds_cols].values

    preds = np.clip(preds, eps, 1 - eps)

    return rsna_loss(preds, df)


def rsna_score_organs(preds, dataset, eps=1e-6):
    """
    Calculate the RSNA score for organ-level classification tasks.

    Args:
        preds (numpy.ndarray): Model predictions for organ-level targets.
        dataset (AbdominalDataset): Dataset containing patient information.
        eps (float, optional): A small value to prevent division by zero. Defaults to 1e-6.

    Returns:
        float: RSNA score for the organ-level classification tasks.
    """
    preds = preds.astype(np.float64)
    preds = preds.reshape(-1, 5, preds.shape[-1]).max(1)
    preds = np.clip(preds, eps, 1 - eps)
    return rsna_loss(preds, dataset.df_patient)


def roc_auc_score_organs(preds, dataset):
    """
    Calculate the ROC AUC score for organ-level classification tasks.

    Args:
        preds (numpy.ndarray): Model predictions for organ-level targets.
        dataset (AbdominalDataset): Dataset containing patient information and organ-level targets.

    Returns:
        float: Mean ROC AUC score for organ-level classification tasks.
    """
    preds = preds.reshape(-1, len(dataset.classes), preds.shape[-1]).max(1)
    mapping = {
        "bowel_injury": 0,
        "extravasation_injury": 1,
        "kidney": 2,
        "liver": 5,
        "spleen": 8,
    }
    aucs = []
    for i, tgt in enumerate(PATIENT_TARGETS):
        if i >= len(dataset.classes):
            break

        if "injury" in tgt:
            auc = roc_auc_score(dataset.df_patient[tgt] > 0, preds[:, mapping[tgt]])
        else:
            auc = roc_auc_score(dataset.df_patient[tgt] <= 0, preds[:, mapping[tgt]])
        aucs.append(auc)
    return np.mean(aucs)
