import torch
import numpy as np
import torch.nn as nn


class Mixup(nn.Module):
    def __init__(self, alpha, additive=False, num_classes=1):
        """
        Mixup augmentation module.

        Args:
            alpha (float): Mixup interpolation parameter.
            additive (bool, optional): Whether to use additive mixup. Defaults to False.
            num_classes (int, optional): Number of classes. Defaults to 1.
        """
        super(Mixup, self).__init__()
        self.beta_distribution = torch.distributions.Beta(alpha, alpha)
        self.additive = additive
        self.num_classes = num_classes

    def forward(self, x, y, y_aux=None):
        """
        Forward pass of the Mixup module.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels for the main task.
            y_aux (torch.Tensor, optional): Target labels for the auxiliary task. Defaults to None.

        Returns:
            torch.Tensor: Mixed input data.
            torch.Tensor: Mixed target labels for the main task.
            torch.Tensor: Mixed target labels for the auxiliary task.
        """
        bs = x.shape[0]
        n_dims = len(x.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(x.device)

        if self.num_classes > y.size(-1):  # One-hot
            y = (
                torch.zeros(y.size(0), self.num_classes)
                .to(y.device)
                .scatter(1, y.view(-1, 1).long(), 1)
            )

        if n_dims == 2:
            x = coeffs.view(-1, 1) * x + (1 - coeffs.view(-1, 1)) * x[perm]
        elif n_dims == 3:
            x = coeffs.view(-1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1)) * x[perm]
        elif n_dims == 4:
            x = coeffs.view(-1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1)) * x[perm]
        else:
            x = (
                coeffs.view(-1, 1, 1, 1, 1) * x
                + (1 - coeffs.view(-1, 1, 1, 1, 1)) * x[perm]
            )

        if self.additive:
            y = torch.cat([y.unsqueeze(0), y[perm].unsqueeze(0)], 0).amax(0)
            y[:, 3] *= 1 - y[:, 4]
            y[:, 2] = 1 - y[:, 4] - y[:, 3]
            y[:, 6] *= 1 - y[:, 7]
            y[:, 5] = 1 - y[:, 7] - y[:, 6]
            y[:, 9] *= 1 - y[:, 10]
            y[:, 8] = 1 - y[:, 10] - y[:, 9]

            if y_aux is not None:
                #                 raise NotImplementedError
                y_aux = (y_aux + y_aux[perm]).clip(0, 1)
        else:
            if len(y.shape) == 1:
                y = coeffs * y + (1 - coeffs) * y[perm]
                if y_aux is not None:
                    y_aux = coeffs * y_aux + (1 - coeffs) * y_aux[perm]
            else:
                y = coeffs.view(-1, 1) * y + (1 - coeffs.view(-1, 1)) * y[perm]
                if y_aux is not None:
                    y_aux = (
                        coeffs.view(-1, 1) * y_aux
                        + (1 - coeffs.view(-1, 1)) * y_aux[perm]
                    )

        return x, y, y_aux


class Cutmix(nn.Module):
    def __init__(self, alpha, additive=False, num_classes=1):
        """
        Cutmix augmentation module.

        Args:
            alpha (float): Cutmix interpolation parameter.
            additive (bool, optional): Whether to use additive cutmix. Defaults to False.
            num_classes (int, optional): Number of classes. Defaults to 1.
        """
        super(Cutmix, self).__init__()
        self.beta_distribution = torch.distributions.Beta(alpha, alpha)
        self.additive = additive
        self.num_classes = num_classes

    @staticmethod
    def rand_bbox(size, lam):
        """
        Returns the coordinates of a random rectangle in the image for cutmix.

        Args:
            size (torch.Tensor): Input size [batch_size x c x w x h].
            lam (float): Lambda sampled by the beta distribution. Controls the size of the rectangle.

        Returns:
            Tuple[int]: 4 coordinates of the rectangle (bbx1, bby1, bbx2, bby2).
            float: Proportion of the unmasked image.
        """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        return bbx1, bby1, bbx2, bby2, lam

    @staticmethod
    def rand_bbox_3d(size, lam):
        """
        Returns the coordinates of a random rectangle in the image for cutmix.

        Args:
            size (torch.Tensor): Input size [batch_size x c x w x h].
            lam (float): Lambda sampled by the beta distribution. Controls the size of the rectangle.

        Returns:
            Tuple[int]: 4 coordinates of the rectangle (bbx1, bby1, bbx2, bby2).
            float: Proportion of the unmasked image.
        """
        w = size[2]
        h = size[3]
        d = size[4]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cut_d = int(d * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        cz = np.random.randint(d)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)

        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        bbz1 = np.clip(cz - cut_d // 2, 0, d)
        bbz2 = np.clip(cz + cut_d // 2, 0, d)

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbz2 - bbz1) / (w * h * d))
        return bbx1, bby1, bbx2, bby2, bbz1, bbz2, lam

    def forward(self, x, y, y_aux=None, use_3d=False):
        """
        Forward pass of the Cutmix module.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target mask for the main task.
            y_aux (torch.Tensor, optional): Target labels for the auxiliary task. Defaults to None.

        Returns:
            torch.Tensor: Augmented input data.
            torch.Tensor: Augmented target labels for the main task.
            torch.Tensor: Augmented target labels for the auxiliary task.
        """
        n_dims = len(x.shape)
        perm = torch.randperm(x.shape[0])
        coeff = (
            self.beta_distribution.rsample(torch.Size((1,)))
            .to(x.device)
            .view(-1)
            .item()
        )

        if use_3d:
            bbx1, bby1, bbx2, bby2, bbz1, bbz2, coeff = self.rand_bbox_3d(
                x.size(), coeff
            )
        else:
            bbx1, bby1, bbx2, bby2, coeff = self.rand_bbox(x.size(), coeff)

        if n_dims == 3:  # bs x h x w
            x[:, bbx1:bbx2, bby1:bby2] = x[perm, bbx1:bbx2, bby1:bby2]
        elif n_dims == 4:  # bs x channels x h x w
            x[:, :, bbx1:bbx2, bby1:bby2] = x[perm, :, bbx1:bbx2, bby1:bby2]
        #         elif n_dims == 5:  # bs x t x channels x h x w
        #
        elif n_dims == 5:  # bs x channels x h x w x d
            if use_3d:
                x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = x[
                    perm, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2
                ]
            else:
                x[:, :, :, bbx1:bbx2, bby1:bby2] = x[perm, :, :, bbx1:bbx2, bby1:bby2]
        else:
            raise NotImplementedError

        n_dims = len(y.shape)
        if n_dims == 2:  # bs x n_classes
            if self.additive:
                y = torch.cat([y.unsqueeze(0), y[perm].unsqueeze(0)], 0).amax(0)
                y[:, 3] *= 1 - y[:, 4]
                y[:, 2] = 1 - y[:, 4] - y[:, 3]
                y[:, 6] *= 1 - y[:, 7]
                y[:, 5] = 1 - y[:, 7] - y[:, 6]
                y[:, 9] *= 1 - y[:, 10]
                y[:, 8] = 1 - y[:, 10] - y[:, 9]
            else:
                y = coeff * y + (1 - coeff) * y[perm]
        elif n_dims == 3:  # mask - bs x h x w
            y[:, bbx1:bbx2, bby1:bby2] = y[perm, bbx1:bbx2, bby1:bby2]
        elif n_dims == 4:  # mask - bs x classes x h x w
            y[:, :, bbx1:bbx2, bby1:bby2] = y[perm, :, bbx1:bbx2, bby1:bby2]
        elif n_dims == 5:  # mask - bs x classes x h x w x d
            if use_3d:
                y[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = y[
                    perm, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2
                ]
            else:
                y[:, :, :, bbx1:bbx2, bby1:bby2] = y[perm, :, :, bbx1:bbx2, bby1:bby2]

        else:
            raise NotImplementedError

        if self.additive:
            y_aux = (y_aux + y_aux[perm]).clip(0, 1)
        else:
            y_aux = coeff * y_aux + (1 - coeff) * y_aux[perm]

        return x, y, y_aux
