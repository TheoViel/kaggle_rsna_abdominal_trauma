import torch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import log_loss, roc_auc_score

from params import PATIENT_TARGETS


WEIGHTS = {
    'bowel_injury': {0: 1, 1: 2},
    'extravasation_injury': {0: 1, 1: 6},
    'kidney': {0: 1, 1: 2, 2: 4},
    'liver': {0: 1, 1: 2, 2: 4},
    'spleen': {0: 1, 1: 2, 2: 4},
    'any_injury': {0: 1, 1: 6},
}


def rsna_loss(preds, truths, eps=1e-6):
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    
    preds: list of the 5 predictions probabilities
    truths: df
    '''
    if isinstance(preds, torch.Tensor):
        preds = preds.clone().cpu().numpy()

    preds = preds.copy()
    if isinstance(preds, np.ndarray):
        if preds.shape[-1] == 11:
            preds = [preds[:, 0], preds[:, 1], preds[:, 2: 5], preds[:, 5: 8], preds[:, 8:]]
        elif preds.shape[-1] == 2:
            preds = [preds[:, 0], preds[:, 1]]
            
    for i, pred in enumerate(preds):
        if pred.shape[-1] in [2, 3]:
            preds[i] = pred / pred.sum(1, keepdims=True)

    losses = {}
    for i, tgt in enumerate(WEIGHTS.keys()):
        if tgt == "any_injury":
            injury_preds = np.concatenate(
                [1 - p[:, :1] if p.shape[-1] in [2, 3] else p.reshape(-1, 1) for p in preds],
                axis=1
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
    preds = preds.astype(np.float64)
    patients = [d[0] for d in dataset.ids]
    df_preds = pd.DataFrame({"patient_id": patients})
    
    preds_cols = []
    for i in range(preds.shape[1]):
        preds_cols.append(f'pred_{i}')
        df_preds[f'pred_{i}'] = preds[:, i]
        
    df_preds = df_preds.groupby('patient_id').mean()
    
    df = dataset.df_patient.merge(df_preds, on="patient_id")
    preds = df[preds_cols].values
    
    preds = np.clip(preds, eps, 1 - eps)

    return rsna_loss(preds, df)


def rsna_score_organs(preds, dataset, eps=1e-6):
    preds = preds.astype(np.float64)
    preds = preds.reshape(-1, 5, preds.shape[-1]).max(1)
    preds = np.clip(preds, eps, 1 - eps)
    return rsna_loss(preds, dataset.df_patient)


def roc_auc_score_organs(preds, dataset):
    preds = preds.reshape(-1, 5, preds.shape[-1]).max(1)
    mapping = {'bowel_injury': 0, 'extravasation_injury': 1, 'kidney': 2, 'liver': 5, 'spleen': 8}
    aucs = []
    for tgt in PATIENT_TARGETS:
        if "injury" in tgt:
            auc = roc_auc_score(dataset.df_patient[tgt] > 0, preds[:, mapping[tgt]])
        else:
#             try:
            auc = roc_auc_score(dataset.df_patient[tgt] <= 0, preds[:, mapping[tgt]])
#             except:
#                 pass

        aucs.append(auc)
    return np.mean(aucs)


def iou_score(bbox1, bbox2):
    """
    IoU metric between boxes in the pascal_voc format.

    Args:
        bbox1 (np array or list [4]): Box 1.
        bbox2 (np array or list [4]): Box 2.

    Returns:
        float: IoU.
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def precision_calc(gt_boxes, pred_boxes, threshold=0.1, return_assignment=False):
    """
    Counts TPs, FPs and FNs for a given IoU threshold between boxes in the pascal_voc format.
    If return_assignment is True, it returns the assigments between predictions and GTs.

    Args:
        gt_boxes (np array or list [n x 4]): Ground truth boxes.
        pred_boxes (np array or list [m x 4]): Prediction boxes.
        threshold (float, optional): _description_. Defaults to 0.25.
        return_assignment (bool, optional): Whether to returns GT/Pred assigment. Defaults to False.

    Returns:
        ints [3]: TPs, FPs, FNs
    """
    cost_matrix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            iou = iou_score(box1, box2)

            if iou < threshold:
                continue

            else:
                cost_matrix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    if return_assignment:
        return cost_matrix, row_ind, col_ind

    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1

    return tp, fp, fn


def compute_metrics(preds, truths):
    """
    Computes metrics for boxes.
    Output contains TP, FP, FN, precision, recall & f1 values.

    Args:
        preds (List of Boxes): Predictions.
        truths (List of Boxes): Truths.

    Returns:
        dict: Metrics
    """
    ftp, ffp, ffn = [], [], []

    if isinstance(preds, list):
        for pred, truth in zip(preds, truths):
            tp, fp, fn = precision_calc(
                truth["pascal_voc"].copy(), pred["pascal_voc"].copy()
            )
            ftp.append(tp)
            ffp.append(fp)
            ffn.append(fn)

            assert ftp + ffn == len(truth)

        tp = np.sum(ftp)
        fp = np.sum(ffp)
        fn = np.sum(ffn)
    else:
        tp, fp, fn = precision_calc(truths.copy(), preds.copy())
        assert tp + fn == len(truths), (tp, fp, fn, len(truths), len(preds))
        assert len(truths)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn)  # if tp + fn else 1

    f1_score = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
