import torch
import numpy as np
from data.dataset import get_frames
from torch.utils.data import Dataset, DataLoader


class AbdominalInfDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
        frames_chanel=0,
        n_frames=1,
        stride=1,
        use_mask=False,
        imgs={},
        paths=[],
        features=[],
    ):
        """
        Constructor.

        Args:
            df_img (pandas DataFrame): Metadata containing information about the dataset.
            df_patient (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df = df
        self.info = self.df[['path', "patient_id", 'series', 'frame']].values
        self.transforms = transforms

        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride

        self.max_frames = dict(df[["series", "frame"]].groupby("series").max()['frame'])
        self.min_frames = dict(df[["series", "frame"]].groupby("series").min()["frame"])

#         self.use_mask = use_mask
#         self.mask_folder = "../logs/2023-09-24/20/masks/"
        
        self.imgs = imgs
        self.paths = paths
        self.features = features

        if len(features):
            self.features = dict(zip(self.get_keys(), features))
            

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)
    
    def get_keys(self):
        keys = []
        for idx in range(len(self.df)):
            path, patient, series, frame = self.info[idx]
            frames = get_frames(
                frame, 1, self.frames_chanel, stride=1, max_frame=self.max_frames[series]
            )
            key = f'{patient}_{series}_{"-".join(list(frames.astype(str)))}'
            keys.append(key)
        return keys

    def _getitem_feature(self, idx):
        path, patient, series, frame = self.info[idx]
        
        all_frames = get_frames(
            frame, self.n_frames, self.frames_chanel, stride=self.stride, max_frame=self.max_frames[series]
        )
        all_frames = all_frames.reshape(-1, 3)
        
        fts = []
        for frames in all_frames:
            key = f'{patient}_{series}_{"-".join(list(frames.astype(str)))}'
            fts.append(self.features[key])
        fts = np.array(fts)
        return fts, 0, 0

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
        if len(self.features):
            return self._getitem_feature(idx)

        path, patient, series, frame = self.info[idx]

        frames = get_frames(
            frame,
            1,
            self.frames_chanel,
            stride=1,
            max_frame=self.max_frames[series],
            min_frame=self.min_frames[series]
        )

        image = self.imgs[np.array(frames)]
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

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
    preds = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    with torch.no_grad():
        for img, _, _ in loader:
            with torch.cuda.amp.autocast(enabled=use_fp16):
                img = img.cuda()

                y_pred = model(img)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
#                 y_pred = torch.zeros((img.size(0), 11)).cuda()

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
