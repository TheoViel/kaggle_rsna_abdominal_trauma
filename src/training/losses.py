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
    Cross-entropy loss with label smoothing.
    """
    def __init__(self, eps=0.0, weighted=True, use_any=True):
        """
        Constructor.
        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super().__init__()
        self.eps = eps
        self.ce = SmoothCrossEntropyLoss(eps=eps)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_nologits = nn.BCELoss(reduction="none")
        self.weighted = weighted
        self.use_any = use_any

    def _forward_soft(self, inputs, targets):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x 11]): Predictions.
            targets (torch tensor [bs x 5] or [bs]): Targets.
        Returns:
            torch tensor [bs]: Loss values.
        """
        assert (targets.size(1) == 11) and (len(targets.size()) == 2), "Wrong target size"
        assert (inputs.size(1) == 11) and (len(inputs.size()) == 2), "Wrong input size"
        
        bowel_pred =  inputs[:, 0]
        bowel_target =  targets[:, 0]
        bowel_w = bowel_target + 1 if self.weighted else 1  # 1, 2
        bowel_loss = self.bce(bowel_pred, bowel_target) * bowel_w
        
        extravasion_pred =  inputs[:, 1]
        extravasion_target =  targets[:, 1]
        extravasion_w = (extravasion_target * 5) + 1 if self.weighted else 1  # 1, 6
        extravasion_loss = self.bce(extravasion_pred, extravasion_target) * extravasion_w
        
        kidney_pred =  inputs[:, 2:5]
        kidney_target =  targets[:, 2:5]
        kidney_w = torch.pow(2, kidney_target.argmax(-1)) if self.weighted else 1 # 1, 2, 4
        kidney_loss = self.ce(kidney_pred, kidney_target) * kidney_w

        liver_pred =  inputs[:, 5:8]
        liver_target =  targets[:, 5:8]
        liver_w = torch.pow(2, liver_target.argmax(-1)) if self.weighted else 1  # 1, 2, 4
        liver_loss = self.ce(liver_pred, liver_target) * liver_w
        
        spleen_pred =  inputs[:, 8:11]
        spleen_target =  targets[:, 8:11]
        spleen_w = torch.pow(2, spleen_target.argmax(-1)) if self.weighted else 1  # 1, 2, 4
        spleen_loss = self.ce(spleen_pred, spleen_target) * spleen_w

        if self.use_any:
            any_target = (targets.amax(-1) > 0).float()
            any_pred = torch.stack(
                [
                    bowel_pred.sigmoid(),
                    extravasion_pred.sigmoid(),
                    1 - kidney_pred.softmax(-1)[:, 0],
                    1 - liver_pred.softmax(-1)[:, 0],
                    1 - spleen_pred.softmax(-1)[:, 0]
                ]
            ).amax(0)

            any_w = (any_target * 5) + 1  if self.weighted else 1  # 1, 6
    #         any_loss = self.bce_nologits(any_pred, any_target) * any_w
            any_loss = - any_w * (any_target * torch.log(any_pred) + (1 - any_target) * torch.log(1 - any_pred))

            loss = (
                bowel_loss + extravasion_loss + kidney_loss + liver_loss + spleen_loss + any_loss
            ) * 1 / 6
        else:
            loss = (
                bowel_loss + extravasion_loss + kidney_loss + liver_loss + spleen_loss
            ) * 1 / 5
    
        return loss

    def forward(self, inputs, targets):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x 11]): Predictions.
            targets (torch tensor [bs x 5] or [bs]): Targets.
        Returns:
            torch tensor [bs]: Loss values.
        """
        if targets.size(-1) == 11:
            return self._forward_soft(inputs, targets)
        
        assert (targets.size(1) == 5) and (len(targets.size()) == 2), "Wrong target size"
        assert (inputs.size(1) == 11) and (len(inputs.size()) == 2), "Wrong input size"
        
        bowel_pred =  inputs[:, 0]
        bowel_target =  targets[:, 0]
        bowel_w = bowel_target + 1 if self.weighted else 1  # 1, 2
        bowel_loss = self.bce(bowel_pred, bowel_target) * bowel_w
        
        extravasion_pred =  inputs[:, 1]
        extravasion_target =  targets[:, 1]
        extravasion_w = (extravasion_target * 5) + 1 if self.weighted else 1  # 1, 6
        extravasion_loss = self.bce(extravasion_pred, extravasion_target) * extravasion_w
        
        kidney_pred =  inputs[:, 2:5]
        kidney_target =  targets[:, 2]
        kidney_w = torch.pow(2, kidney_target) if self.weighted else 1 # 1, 2, 4
        kidney_loss = self.ce(kidney_pred, kidney_target) * kidney_w

        liver_pred =  inputs[:, 5:8]
        liver_target =  targets[:, 3]
        liver_w = torch.pow(2, liver_target) if self.weighted else 1  # 1, 2, 4
        liver_loss = self.ce(liver_pred, liver_target) * liver_w
        
        spleen_pred =  inputs[:, 8:11]
        spleen_target =  targets[:, 4]
        spleen_w = torch.pow(2, spleen_target) if self.weighted else 1  # 1, 2, 4
        spleen_loss = self.ce(spleen_pred, spleen_target) * spleen_w

        if self.use_any:
            any_target = (targets.amax(-1) > 0).float()
            any_pred = torch.stack(
                [
                    bowel_pred.sigmoid(),
                    extravasion_pred.sigmoid(),
                    1 - kidney_pred.softmax(-1)[:, 0],
                    1 - liver_pred.softmax(-1)[:, 0],
                    1 - spleen_pred.softmax(-1)[:, 0]
                ]
            ).amax(0)

            any_w = (any_target * 5) + 1  if self.weighted else 1  # 1, 6
    #         any_loss = self.bce_nologits(any_pred, any_target) * any_w
            any_loss = - any_w * (any_target * torch.log(any_pred) + (1 - any_target) * torch.log(1 - any_pred))

            loss = (
                bowel_loss + extravasion_loss + kidney_loss + liver_loss + spleen_loss + any_loss
            ) * 1 / 6
        else:
            loss = (
                bowel_loss + extravasion_loss + kidney_loss + liver_loss + spleen_loss
            ) * 1 / 5
    
        return loss


class ImageLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    def __init__(self, eps=0.0):
        """
        Constructor.
        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x 11]): Predictions.
            targets (torch tensor [bs x 5] or [bs]): Targets.
        Returns:
            torch tensor [bs]: Loss values.
        """
        assert targets.size(-1) == 2, "Wrong target size"
        assert inputs.size(-1) == 2, "Wrong input size"
        
        bowel_pred =  inputs[:, 0]
        bowel_target =  targets[:, 0]
        bowel_w = bowel_target + 1  # 1, 2
        bowel_loss = self.bce(bowel_pred, bowel_target) * bowel_w
        
        extravasion_pred =  inputs[:, 1]
        extravasion_target =  targets[:, 1]
        extravasion_w = (extravasion_target * 5) + 1  # 1, 6
        extravasion_loss = self.bce(extravasion_pred, extravasion_target) * extravasion_w

        loss = (bowel_loss + extravasion_loss) / 2

        return loss


WEIGHTS = {
    'bowel_injury': {0: 1, 1: 2},
    'extravasation_injury': {0: 1, 1: 6},
    'kidney': {0: 1, 1: 2, 2: 4},
    'liver': {0: 1, 1: 2, 2: 4},
    'spleen': {0: 1, 1: 2, 2: 4},
    'any_injury': {0: 1, 1: 6},
}


class AbdomenLoss(nn.Module):
    """
    Loss wrapper for the problem.

    Attributes:
        config (dict): Configuration parameters.
        device (str): Device to use for computations.
        shape_loss_w (float): Weight for the shape loss.
        aux_loss_weight (float): Weight for the auxiliary loss.
        eps (float): Smoothing value. Defaults to 0.
        loss (nn.Module): Loss function.
        loss_aux (nn.Module): Auxiliary loss function.
        shape_loss (nn.Modulee): Shape loss function.

    Methods:
        __init__(self, config, device="cuda"): Constructor.
        prepare(self, pred, y): Prepares the predictions and targets for loss computation.
        forward(self, pred, pred_aux, y, y_aux): Computes the loss.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor.

        Args:
            config (dict): Configuration parameters.
            device (str, optional): Device to use for computations. Defaults to "cuda".
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
        elif config["name"] == "image":
            self.loss = ImageLoss(eps=self.eps)
        elif config["name"] == "patient":
            self.loss = PatientLoss(
                eps=self.eps,
                weighted=config.get('weighted', False),
                use_any=config.get('use_any', False)
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
                weighted=config.get('weighted', False),
                use_any=config.get('use_any', False)
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
        Main predictions are masks for the segmentation task.
        They are of size [BS x C x H x W] where C=7 if the shape loss is used else 1
        Auxiliary predictions are for the (optional) classification task.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
#         print(pred.size(), y.size(), y.max())
        pred, pred_aux, y, y_aux = self.prepare(pred, pred_aux, y, y_aux)
#         print(pred.size(), y.size(), y.max())

        loss = self.loss(pred, y)

        if self.aux_loss_weight > 0:
            loss_aux = self.loss_aux(pred_aux, y_aux)
            loss =  (1 - self.aux_loss_weight) * loss + self.aux_loss_weight * loss_aux
    
        return loss.mean()


    
class SegLoss(nn.Module):
    """
    Loss wrapper for the problem.

    Attributes:
        config (dict): Configuration parameters.
        device (str): Device to use for computations.
        shape_loss_w (float): Weight for the shape loss.
        aux_loss_weight (float): Weight for the auxiliary loss.
        eps (float): Smoothing value. Defaults to 0.
        loss (nn.Module): Loss function.
        loss_aux (nn.Module): Auxiliary loss function.
        shape_loss (nn.Modulee): Shape loss function.

    Methods:
        __init__(self, config, device="cuda"): Constructor.
        prepare(self, pred, y): Prepares the predictions and targets for loss computation.
        forward(self, pred, pred_aux, y, y_aux): Computes the loss.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor.

        Args:
            config (dict): Configuration parameters.
            device (str, optional): Device to use for computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "ce":
            self.loss = nn.CrossEntropyLoss(reduction="none")  # SmoothCrossEntropyLoss(eps=self.eps)
        else:
            raise NotImplementedError

        if config["name_aux"] == "bce":
            self.loss_aux = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name_aux"] == "ce":
            self.loss_aux = nn.CrossEntropyLoss(reduction="none")  # SmoothCrossEntropyLoss(eps=self.eps)
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
            y = F.one_hot(
               y.squeeze(1).long(), num_classes=self.config["num_classes"] + 1
            ).permute(0, 3, 1, 2)[:, 1:].float()
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
        Main predictions are masks for the segmentation task.
        They are of size [BS x C x H x W] where C=7 if the shape loss is used else 1
        Auxiliary predictions are for the (optional) classification task.

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
