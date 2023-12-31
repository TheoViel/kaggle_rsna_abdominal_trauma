import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PatientFeatureInfDataset(Dataset):
    """
    Dataset for inferring level 2 models.

    Attributes:
        fts (numpy.ndarray, optional): Features from 2D models. Defaults to None.
        crop_fts (numpy.ndarray, optional): Features from crop models. Defaults to None.
        max_len (int, optional): Maximum length for features. Defaults to None.
        restrict (bool, optional): Flag to restrict the data. Defaults to False.
        resize (int, optional): Resize the data. Defaults to None.
        half (bool, optional): Whether to use half the data. Defaults to False.
    """
    def __init__(
        self,
        series,
        exp_folders,
        crop_fts=None,
        max_len=None,
        restrict=False,
        resize=None,
        save_folder="",
        half=False,
    ):
        """
        Constructor.

        Args:
            series (list): List of data series.
            exp_folders (list): List of experiment folders.
            crop_fts (numpy.ndarray, optional): Cropped features data. Defaults to None.
            max_len (int, optional): Maximum length for features. Defaults to None.
            restrict (bool, optional): Flag to restrict the data. Defaults to False.
            resize (int, optional): Resize the data. Defaults to None.
            save_folder (str, optional): Folder to save data. Defaults to "".
            half (bool, optional): Whether to use half the data. Defaults to False.
        """
        self.fts = self.retrieve_features(series, exp_folders, save_folder=save_folder)
        self.ids = [0]
        self.max_len = max_len
        self.restrict = restrict
        self.resize = resize
        self.crop_fts = crop_fts
        self.half = half

    @staticmethod
    def retrieve_features(series, exp_folders, save_folder=""):
        """
        Retrieve features based on input data series and experiment folders.

        Args:
            series (list): List of data series.
            exp_folders (list): List of experiment folders.
            save_folder (str, optional): Folder to save data. Defaults to "".

        Returns:
            list: List of features data.
        """
        all_fts = []
        exp_names = [
            "_".join(exp_folder.split("/")[-3:-1]) for exp_folder, _ in exp_folders
        ]

        for s in series:
            fts = []
            for exp_name, (exp_folder, mode) in zip(exp_names, exp_folders):
                if mode == "seg":
                    seg = np.load(save_folder + f"{s}_{exp_name}.npy")
                    fts.append(seg)
                elif mode == "crop":
                    pass
                else:  # proba
                    ft = np.load(save_folder + f"{s}_{exp_name}.npy")
                    fts.append(ft)

                    kidney = (
                        seg[:, 2:4].max(-1, keepdims=True)
                        if seg.shape[-1] == 5
                        else seg[:, 2:3]
                    )
                    fts.append(
                        np.concatenate(
                            [
                                ft[:, :1] * seg[:, -1:],  # bowel
                                ft[:, 1:2] * seg.max(-1, keepdims=True),  # extravasation
                                ft[:, 2:5] * kidney,  # kidney
                                ft[:, 5:8] * seg[:, :1],  # liver
                                ft[:, 8:] * seg[:, 1:2],  # spleen
                            ],
                            -1,
                        )
                    )

            fts = np.concatenate(fts, axis=1)
            all_fts.append(fts)
        return all_fts

    def pad(self, x):
        """
        Pad the features data.

        Args:
            x (numpy.ndarray): Features data.

        Returns:
            numpy.ndarray: Padded features data.
        """
        length = x.shape[0]
        if length > self.max_len:
            return x[-self.max_len:]
        else:
            padded = np.zeros([self.max_len] + list(x.shape[1:]))
            padded[-length:] = x
            return padded

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.fts)

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            dict: Data and feature tensors.
            int: Placeholder (not used).
            int: Placeholder (not used).
        """
        fts = self.fts[idx]
        crop_fts = self.crop_fts[idx] if self.crop_fts is not None else None

        if self.restrict:
            if len(fts) > 400:
                fts = fts[len(fts) // 6:]
            else:
                fts = fts[len(fts) // 8:]

        if self.resize:
            if self.max_len is not None:  # crop too long
                fts = fts[-self.max_len:]

            if self.half:
                fts = fts[::2].copy()

            fts = F.interpolate(
                torch.from_numpy(fts.T).float().unsqueeze(0),
                size=self.resize,
                mode="linear",
            )[0].transpose(0, 1)

        else:
            if self.max_len is not None:
                fts = self.pad(fts)

            fts = torch.from_numpy(fts).float()

        if crop_fts is not None:
            crop_fts = torch.from_numpy(crop_fts).float()
        else:
            crop_fts = 0

        return {"x": fts, "ft": crop_fts}, 0, 0


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
    """
    model.eval()
    preds = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for x, _, _ in loader:
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, _ = model(x["x"].cuda(), x["ft"].cuda())

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
    """
    Transform a DataFrame into a submission format suitable for evaluation.

    Args:
        df (pandas.DataFrame): The input DataFrame containing predictions.

    Returns:
        pandas.DataFrame: A new DataFrame with columns formatted for submission.
    """
    new_df = df[["patient"]].copy()
    new_df.columns = ["patient_id"]

    new_df["bowel_healthy"] = 1 - df["pred_0"]
    new_df["bowel_injury"] = df["pred_0"]

    new_df["extravasation_healthy"] = 1 - df["pred_1"]
    new_df["extravasation_injury"] = df["pred_1"]

    new_df["kidney_healthy"] = df["pred_2"]
    new_df["kidney_low"] = df["pred_3"]
    new_df["kidney_high"] = df["pred_4"]

    new_df["liver_healthy"] = df["pred_5"]
    new_df["liver_low"] = df["pred_6"]
    new_df["liver_high"] = df["pred_7"]

    new_df["spleen_healthy"] = df["pred_8"]
    new_df["spleen_low"] = df["pred_9"]
    new_df["spleen_high"] = df["pred_10"]

    return new_df
