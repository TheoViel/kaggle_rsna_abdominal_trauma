import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Abdominal2DInfDataset(Dataset):
    def __init__(self, df, transforms=None, frames_chanel=0, imgs={}):
        """
        Constructor.

        Args:
            df_img (pandas DataFrame): Metadata containing information about the dataset.
            df_patient (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df = df
        self.info = self.df[["path", "series", "frame"]].values
        self.transforms = transforms
        self.frames_chanel = frames_chanel
        self.max_frames = dict(df[["series", "frame"]].groupby("series").max()["frame"])

        self.imgs = imgs

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [C, H, W].
            torch.Tensor: Mask as a tensor of shape [1 or 7, H, W].
            torch.Tensor: Label as a tensor of shape [1].
        """
        path, series, frame = self.info[idx]

        image = self.imgs[path]
        #         if len(image.shape) == 2:
        #             image = np.concatenate([image[:, :, None]] * 3, -1)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        #         image = torch.cat([image] * 3)

        return image, 0, 0


def predict(
    model,
    dataset,
    loss_config,
    batch_size=64,
    device="cuda",
    use_fp16=False,
    num_workers=8,
):
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
                img = img.cuda()
                if img.size(1) == 1:
                    img = img.repeat(1, 3, 1, 1)

                y_pred, ft = model(img, return_fts=True)
            #                 y_pred, ft = torch.zeros(1), torch.zeros(1)

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
