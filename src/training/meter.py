import torch
import numpy as np


class SegmentationMeter:
    """
    Meter to handle predictions and metrics for segmentation tasks.

    This class is used to compute and track metrics for segmentation tasks, such as
    Intersection over Union (IoU) and accuracy. It maintains a set of thresholds for
    predictions and provides methods to update the meter with new predictions and to
    reset the metrics.

    Methods:
        __init__(threshold): Constructor.
        update(y, y_aux, y_pred, y_pred_aux): Update the meter with new predictions.
        reset(): Reset all the tracked metrics.

    Attributes:
        threshold (float): Threshold for predictions.
        thresholds (numpy array): Array of thresholds used for computing metrics.
        unions (dict): Dictionary of tensors to track intersection counts for each threshold.
        intersections (dict): Dictionary of tensors to track union counts for each threshold.
        accs (list): List of tensors to track accuracy values.
    """

    def __init__(self, threshold=0.25):
        """
        Constructor.

        Args:
            threshold (float, optional): Threshold for predictions. Defaults to 0.25.
        """
        self.threshold = threshold
        self.thresholds = np.round(np.arange(0.2, 0.6, 0.01), 2)
        self.reset()

    def update(self, y, y_aux, y_pred, y_pred_aux):
        """
        Update the meter with new predictions.

        Args:
            y (torch tensor): Ground truth labels.
            y_aux (torch tensor): Additional ground truth labels (auxiliary).
            y_pred (torch tensor): Predicted logits.
            y_pred_aux (torch tensor): Additional predicted logits (auxiliary).
        """
        y_pred = y_pred[:, :1].contiguous()  # only first class
        y_pred = y_pred.view(1, -1)

        y = y[:, :1].contiguous()  # only first class
        y = y.view(1, -1) > 0

        for th in self.thresholds:
            self.unions[th] += ((y_pred > th).sum(-1) + y.sum(-1)).int()
            self.intersections[th] += (((y_pred > th) & y).sum(-1)).int()

        self.accs.append(
            ((y_pred_aux.view(-1) > self.threshold) == (y_aux.view(-1) > 0)).float()
        )

    def reset(self):
        """
        Reset all the tracked metrics.
        """
        self.unions = {th: torch.zeros(1, dtype=torch.int).cuda() for th in self.thresholds}
        self.intersections = {th: torch.zeros(1, dtype=torch.int).cuda() for th in self.thresholds}
        self.accs = []
