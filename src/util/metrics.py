import torch
import numpy as np
from sklearn.metrics import log_loss


WEIGHTS = {
    'bowel_injury': {0: 1, 1: 2},
    'extravasation_injury': {0: 1, 1: 6},
    'kidney': {0: 1, 1: 2, 2: 4},
    'liver': {0: 1, 1: 2, 2: 4},
    'spleen': {0: 1, 1: 2, 2: 4},
    'any_injury': {0: 1, 1: 6},
}


def rsna_loss(preds, truths):
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
        preds = preds.cpu().numpy()
    if isinstance(preds, np.ndarray):
        if preds.shape[-1] == 11:
            preds = [preds[:, 0], preds[:, 1], preds[:, 2: 5], preds[:, 5: 8], preds[:, 8:]]
        elif preds.shape[-1] == 2:
            preds = [preds[:, 0], preds[:, 1]]

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
            pred = pred / pred.sum(1, keepdims=True)
            labels = np.arange(pred.shape[-1])
        else:  # sigmoid was used, we have p of injury
            labels = [0, 1]
        
        loss = log_loss(truth, pred, sample_weight=sample_weight, labels=labels)
        losses[tgt] = loss
        
        if len(preds) == 2:
            if tgt == "extravasation_injury":
                break

    return losses, np.mean(list(losses.values()))
