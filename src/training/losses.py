import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """

    def __init__(self, eps=0.0):
        """
        Constructor.
        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x n]): Predictions.
            targets (torch tensor [bs x n] or [bs]): Targets.
        Returns:
            torch tensor [bs]: Loss values.
        """
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)

        if self.eps > 0:
            n_class = inputs.size(1)
            targets = targets * (1 - self.eps) + (1 - targets) * self.eps / (
                n_class - 1
            )

        loss = -targets * F.log_softmax(inputs, dim=1)
        loss = loss.sum(-1)

        return loss


class PatientLoss(nn.Module):
    """
    Custom loss function for patient predictions.

    Attributes:
        eps (float): Smoothing factor for cross-entropy loss.
        weighted (bool): Flag to apply class-weighted loss.
        use_any (bool): Flag to include 'any' label in the loss calculation.
        bce (nn.BCEWithLogitsLoss): BCE loss for bowel & extravasation.
        ce (SmoothCrossEntropyLoss): CE loss for spleen, liver & kidney.
    """
    def __init__(self, eps=0.0, weighted=True, use_any=True):
        """
        Constructor.

        Args:
            eps (float, optional): Smoothing factor for cross-entropy loss. Defaults to 0.0.
            weighted (bool, optional): Flag to apply class-weighted loss. Defaults to True.
            use_any (bool, optional): Flag to include 'any' label in the loss calculation. Defaults to True.
        """
        super().__init__()
        self.eps = eps
        self.ce = SmoothCrossEntropyLoss(eps=eps)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.weighted = weighted
        self.use_any = use_any

    def _forward_soft(self, inputs, targets):
        """
        Compute the loss with soft targets.

        Args:
            inputs (torch.Tensor): Predictions of shape (batch_size, num_classes).
            targets (torch.Tensor): Soft targets of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Loss value.
        """
        assert (targets.size(1) == 11) and (len(targets.size()) == 2), "Wrong target size"
        assert (inputs.size(1) == 11) and (len(inputs.size()) == 2), "Wrong input size"

        w1 = 1
        w2 = 5
        w3 = 2
        w_any = 5

        bowel_pred = inputs[:, 0]
        bowel_target = targets[:, 0]
        bowel_w = bowel_target + w1 if self.weighted else 1  # 1, 2
        bowel_loss = self.bce(bowel_pred, bowel_target) * bowel_w

        extravasion_pred = inputs[:, 1]
        extravasion_target = targets[:, 1]
        extravasion_w = (extravasion_target * w2) + 1 if self.weighted else 1  # 1, 6
        extravasion_loss = (
            self.bce(extravasion_pred, extravasion_target) * extravasion_w
        )

        kidney_pred = inputs[:, 2:5]
        kidney_target = targets[:, 2:5]
        kidney_w = (
            torch.pow(w3, kidney_target.argmax(-1)) if self.weighted else 1
        )  # 1, 2, 4
        kidney_loss = self.ce(kidney_pred, kidney_target) * kidney_w

        liver_pred = inputs[:, 5:8]
        liver_target = targets[:, 5:8]
        liver_w = (
            torch.pow(w3, liver_target.argmax(-1)) if self.weighted else 1
        )  # 1, 2, 4
        liver_loss = self.ce(liver_pred, liver_target) * liver_w

        spleen_pred = inputs[:, 8:11]
        spleen_target = targets[:, 8:11]
        spleen_w = (
            torch.pow(w3, spleen_target.argmax(-1)) if self.weighted else 1
        )  # 1, 2, 4
        spleen_loss = self.ce(spleen_pred, spleen_target) * spleen_w

        if self.use_any:
            any_target = (targets.amax(-1) > 0).float()
            any_pred = torch.stack(
                [
                    bowel_pred.sigmoid(),
                    extravasion_pred.sigmoid(),
                    1 - kidney_pred.softmax(-1)[:, 0],
                    1 - liver_pred.softmax(-1)[:, 0],
                    1 - spleen_pred.softmax(-1)[:, 0],
                ]
            ).amax(0)

            any_w = (any_target * w_any) + 1 if self.weighted else 1  # 1, 6
            any_loss = -any_w * (
                any_target * torch.log(any_pred)
                + (1 - any_target) * torch.log(1 - any_pred)
            )

            loss = (
                (bowel_loss + extravasion_loss + kidney_loss + liver_loss + spleen_loss + any_loss)
                * 1 / 6
            )
        else:
            loss = (
                (bowel_loss + extravasion_loss + kidney_loss + liver_loss + spleen_loss)
                * 1 / 5
            )

        return loss

    def forward(self, inputs, targets):
        """
        Forward pass for the PatientLoss class.

        Args:
            inputs (torch.Tensor): Model predictions of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Loss value.
        """
        if targets.size(-1) == 11:
            return self._forward_soft(inputs, targets)

        assert (targets.size(1) == 5) and (len(targets.size()) == 2), "Wrong target size"
        assert (inputs.size(1) == 11) and (len(inputs.size()) == 2), "Wrong input size"

        w1 = 1
        w2 = 5
        w3 = 2
        w_any = 5

        bowel_pred = inputs[:, 0]
        bowel_target = targets[:, 0]
        bowel_w = bowel_target + w1 if self.weighted else 1  # 1, 2
        bowel_loss = self.bce(bowel_pred, bowel_target) * bowel_w

        extravasion_pred = inputs[:, 1]
        extravasion_target = targets[:, 1]
        extravasion_w = (extravasion_target * w2) + 1 if self.weighted else 1  # 1, 6
        extravasion_loss = (
            self.bce(extravasion_pred, extravasion_target) * extravasion_w
        )

        kidney_pred = inputs[:, 2:5]
        kidney_target = targets[:, 2]
        kidney_w = torch.pow(w3, kidney_target) if self.weighted else 1  # 1, 2, 4
        kidney_loss = self.ce(kidney_pred, kidney_target) * kidney_w

        liver_pred = inputs[:, 5:8]
        liver_target = targets[:, 3]
        liver_w = torch.pow(w3, liver_target) if self.weighted else 1  # 1, 2, 4
        liver_loss = self.ce(liver_pred, liver_target) * liver_w

        spleen_pred = inputs[:, 8:11]
        spleen_target = targets[:, 4]
        spleen_w = torch.pow(w3, spleen_target) if self.weighted else 1  # 1, 2, 4
        spleen_loss = self.ce(spleen_pred, spleen_target) * spleen_w

        if self.use_any:
            any_target = (targets.amax(-1) > 0).float()
            any_pred = torch.stack(
                [
                    bowel_pred.sigmoid(),
                    extravasion_pred.sigmoid(),
                    1 - kidney_pred.softmax(-1)[:, 0],
                    1 - liver_pred.softmax(-1)[:, 0],
                    1 - spleen_pred.softmax(-1)[:, 0],
                ]
            ).amax(0)

            any_w = (any_target * w_any) + 1 if self.weighted else 1  # 1, 6

            any_loss = -any_w * (
                any_target * torch.log(any_pred)
                + (1 - any_target) * torch.log(1 - any_pred)
            )

            loss = (
                (bowel_loss + extravasion_loss + kidney_loss + liver_loss + spleen_loss + any_loss)
                * 1 / 6
            )
        else:
            loss = (
                (bowel_loss + extravasion_loss + kidney_loss + liver_loss + spleen_loss)
                * 1 / 5
            )

        return loss


class AbdomenLoss(nn.Module):
    """
    Custom loss function for the problem.

    Attributes:
        config (dict): Configuration parameters for the loss.
        device (str): Device to perform loss computations (e.g., "cuda" or "cpu").
        aux_loss_weight (float): Weight for the auxiliary loss.
        eps (float): Smoothing factor for the primary loss.
        eps_aux (float): Smoothing factor for the auxiliary loss.
        loss (torch.nn.Module): Loss function for primary predictions.
        loss_aux (torch.nn.Module): Loss function for auxiliary predictions.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor for the AbdomenLoss class.

        Args:
            config (dict): Configuration parameters for the loss.
            device (str, optional): Device to perform loss computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.eps = config.get("smoothing", 0)
        self.eps_aux = config.get("smoothing_aux", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(eps=self.eps)
        elif config["name"] == "patient":
            self.loss = PatientLoss(
                eps=self.eps,
                weighted=config.get("weighted", False),
                use_any=config.get("use_any", False),
            )
        else:
            raise NotImplementedError

        if config["name_aux"] == "bce":
            self.loss_aux = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name_aux"] == "ce":
            self.loss_aux = SmoothCrossEntropyLoss(eps=self.eps_aux)
        elif config["name_aux"] == "patient":
            self.loss_aux = PatientLoss(
                eps=self.eps,
                weighted=config.get("weighted", False),
                use_any=config.get("use_any", False),
            )
        else:
            raise NotImplementedError

    def prepare(self, pred, pred_aux, y, y_aux):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] == "ce":
            y = y.squeeze(1)
        elif self.config["name"] in ["bce", "image"]:
            y = y.float()
            pred = pred.float().view(y.size())
            if self.eps:
                y = torch.clamp(y, self.eps, 1 - self.eps)
        else:
            pass

        if self.config["name_aux"] == "ce":
            y_aux = y_aux.squeeze()
        elif self.config["name_aux"] in ["bce", "image"]:
            y_aux = y_aux.float()
            pred_aux = pred_aux.float().view(y_aux.size())
            if self.eps_aux:
                y_aux = torch.clamp(y_aux, self.eps_aux, 1 - self.eps_aux)
        else:
            pass

        return pred, pred_aux, y, y_aux

    def forward(self, pred, pred_aux, y, y_aux):
        """
        Computes the loss.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, pred_aux, y, y_aux = self.prepare(pred, pred_aux, y, y_aux)

        loss = self.loss(pred, y)

        if self.aux_loss_weight > 0:
            loss_aux = self.loss_aux(pred_aux, y_aux)
            loss = (1 - self.aux_loss_weight) * loss + self.aux_loss_weight * loss_aux

        return loss.mean()


class SegLoss(nn.Module):
    """
    Custom loss function for segmentation tasks.

    Attributes:
        config (dict): Configuration parameters for the loss.
        device (str): Device to perform loss computations (e.g., "cuda" or "cpu").
        aux_loss_weight (float): Weight for the auxiliary loss.
        eps (float): Smoothing factor for the primary loss.
        loss (torch.nn.Module): Loss function for primary predictions.
        loss_aux (torch.nn.Module): Loss function for auxiliary predictions.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor for the SegLoss class.

        Args:
            config (dict): Configuration parameters for the loss.
            device (str, optional): Device to perform loss computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "ce":
            self.loss = nn.CrossEntropyLoss(
                reduction="none"
            )
        else:
            raise NotImplementedError

        if config["name_aux"] == "bce":
            self.loss_aux = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name_aux"] == "ce":
            self.loss_aux = nn.CrossEntropyLoss(
                reduction="none"
            )
        else:
            raise NotImplementedError

    def prepare(self, pred, pred_aux, y, y_aux):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] == "ce":
            y = y.squeeze(1).long()
        else:  # bce, lovasz, focal
            y = (
                F.one_hot(
                    y.squeeze(1).long(), num_classes=self.config["num_classes"] + 1
                )
                .permute(0, 3, 1, 2)[:, 1:]
                .float()
            )
            pred = pred.float().view(y.size())

        if self.config["name_aux"] == "ce":
            pred_aux = pred_aux.float()
            y_aux = y_aux.squeeze(1).long()
        else:
            y_aux = y_aux.float()
            pred_aux = pred_aux.float()

        if self.eps and self.config["name"] == "bce":
            y = torch.clamp(y, self.eps, 1 - self.eps)

        return pred, pred_aux, y, y_aux

    def forward(self, pred, pred_aux, y, y_aux):
        """
        Computes the loss.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, pred_aux, y, y_aux = self.prepare(pred, pred_aux, y, y_aux)

        loss = self.loss(pred, y).mean()

        if not self.aux_loss_weight > 0:
            return loss

        loss_aux = self.loss_aux(pred_aux, y_aux).mean()
        return (1 - self.aux_loss_weight) * loss + self.aux_loss_weight * loss_aux
