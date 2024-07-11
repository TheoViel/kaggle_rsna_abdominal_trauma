import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.preparation import get_df_series
from params import IMG_TARGETS_EXTENDED, PATIENT_TARGETS


def to_one_hot_patient(y):
    """
    Convert a patient target tensor to a one-hot encoded representation.
    Each column with index less than or equal to 1 (bowel, extrav) are unchanged.
    Columns with index greater than 1 are one-hot encoded based on their original class values.

    Args:
        y (torch.Tensor): The input multi-class tensor of shape (N, C), where N is the number
        of samples and C is the number of classes.

    Returns:
        torch.Tensor: A one-hot encoded tensor of shape (N, K), where K is the sum of the number
        of classes in each column of the input tensor.
    """
    new_y = []
    for i in range(y.size(1)):
        if i <= 1:
            new_y.append(y[:, i].unsqueeze(-1))
        else:
            y_ = (
                torch.zeros(y.size(0), 3)
                .to(y.device)
                .scatter(1, y[:, i].view(-1, 1).long(), 1)
            )
            new_y.append(y_)
    return torch.cat(new_y, -1)


def get_frames(frame, n_frames, frames_c, stride=1, max_frame=100):
    """
    Calculate a sequence of frame indices based on the specified parameters.
    If stride is -1, sample n_frames from 0 to max_frame using linear spacing.

    Args:
        frame (int): The central frame index around which the sequence is generated.
        n_frames (int): The number of frames in the sequence.
        frames_c (int): The number of frames to be repeated and offset around each frame.
        stride (int, optional): The step size between frames. Defaults to 1.
        max_frame (int, optional): The maximum frame index allowed. Defaults to 100.

    Returns:
        numpy.ndarray: An array of frame indices representing the calculated sequence.
    """
    if stride == -1:
        frames = np.linspace(0, max_frame, n_frames + 4, endpoint=True, dtype=int)[2:-2]

    else:
        frames = np.arange(n_frames) * stride
    frames = frames - frames[n_frames // 2] + frame

    if frames_c:
        offset = np.tile(np.arange(-1, 2) * frames_c, len(frames))
        frames = np.repeat(frames, 3) + offset

    if frames.min() < 0:
        frames -= frames.min()
    elif frames.max() > max_frame:
        frames += max_frame - frames.max()

    frames = np.clip(frames, 0, max_frame)
    return frames


class AbdominalDataset(Dataset):
    """
    Dataset for training 2D classification models.

    Attributes:
        df_img (pandas DataFrame): Metadata containing image information.
        df_patient (pandas DataFrame): Metadata containing patient information.
        transforms (albu transforms): Transforms to apply to the images.
        frames_chanel (int): The number of frames to consider for channel stacking.
        n_frames (int): The number of frames to use.
        stride (int): The step size between frames.
        train (bool): Flag indicating whether the dataset is for training.
        classes (list): List of target classes.
        targets (numpy.ndarray): Array of patient targets.
        max_frames (dict): Dictionary of maximum frames per series.
    """

    def __init__(
        self,
        df_patient,
        df_img,
        transforms=None,
        frames_chanel=0,
        n_frames=0,
        stride=1,
        train=False,
    ):
        """
        Constructor.

        Args:
            df_patient (pandas DataFrame): Metadata containing patient information.
            df_img (pandas DataFrame): Metadata containing image information.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
            frames_chanel (int, optional): Number of frames to consider for channel stacking. Defaults to 0.
            n_frames (int, optional): The number of frames to use. Defaults to 0.
            stride (int, optional): The step size between frames. Defaults to 1.
            train (bool, optional): Flag indicating whether the dataset is for training. Defaults to False.
        """
        self.df_img = df_img
        self.df_patient = df_patient
        self.transforms = transforms
        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride
        self.train = train

        self.classes = IMG_TARGETS_EXTENDED

        self.targets = df_patient[PATIENT_TARGETS].values
        self.max_frames = dict(
            df_img[["series", "frame"]].groupby("series").max()["frame"]
        )

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df_patient) * len(self.classes)

    def __getitem__(self, idx):
        """
        Item accessor.
        Frames are sampled the following way:
        - kidney / liver / spleen  / negative bowel : Inside the organ.
        - positive bowel / positive extravasation : Using the frame-level labels.
        - Negative extravasation : Anywhere

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [(N) x C, H, W].
            torch.Tensor: Label as a tensor of shape [9].
            torch.Tensor: Aux label as a tensor of shape [9]. Not used.
        """
        tgt_idx = idx % len(self.classes)
        tgt = self.classes[tgt_idx]

        idx = idx // len(self.classes)
        patient = self.df_patient["patient_id"].values[idx]
        y_patient = self.targets[idx]

        df_img = self.df_img[self.df_img["patient_id"] == patient]

        # Restrict to considered class
        if (df_img[self.classes[tgt_idx]] == y_patient[tgt_idx]).max():
            df_img = df_img[df_img[self.classes[tgt_idx]] == y_patient[tgt_idx]]
        else:  # Class has no match, use argmax - should not happen
            raise NotImplementedError

        # Restrict to segmentation > 0.9 for negatives
        if not y_patient[tgt_idx]:
            seg = df_img[f'pred_{tgt.split("_")[0]}'].values
            seg = seg / (seg.max() + 1e-6)
            df_img = df_img[seg > 0.9]

        # Restrict to one series
        series = (
            np.random.choice(df_img["series"].unique())
            if self.train
            else df_img["series"].values[0]
        )
        df_img = df_img[df_img["series"] == series]

        # Sort by frame
        df_img = df_img.sort_values("frame").reset_index(drop=True)

        # Pick a row
        if self.train:
            ps = np.exp(
                -(
                    ((np.arange(len(df_img)) - len(df_img) // 2) / (0.4 * len(df_img)))
                    ** 2
                )
            )  # gaussian
            row_idx = np.random.choice(len(df_img), p=ps / ps.sum())
            row = df_img.iloc[row_idx]
        else:
            row = df_img.iloc[len(df_img) // 2]  # center

        if self.frames_chanel > 0 or self.n_frames > 1:
            frame = row.frame

            if self.n_frames <= 1:
                frame = np.clip(
                    frame,
                    self.frames_chanel,
                    self.max_frames[series] - self.frames_chanel,
                )
                frames = [frame - self.frames_chanel, frame, frame + self.frames_chanel]
            else:
                frames = get_frames(
                    frame,
                    self.n_frames,
                    self.frames_chanel,
                    stride=self.stride,
                    max_frame=self.max_frames[series],
                )

            prefix = row.path.rsplit("_", 1)[0]
            paths = [prefix + f"_{f:04d}.png" for f in frames]
            target_size = (512, 512)

            images = [cv2.resize(cv2.imread(path, 0), target_size) for path in paths]
            image = np.array(images).transpose(1, 2, 0)

            # print("image_shape", image.shape)
            # image = np.array([cv2.imread(path, 0) for path in paths]).transpose(1, 2, 0)

        else:
            frame = row.frame
            image = cv2.imread(row.path)

        image = image.astype(np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
            print("after transform: ", image.shape)
        y_patient = torch.tensor(y_patient, dtype=torch.float)
        y_img = torch.tensor(row[self.classes], dtype=torch.float)
        print("done 238")
        if y_img.size(-1) == 5:  # Patient level - TODO : y_patient ?
            y_img = to_one_hot_patient(y_img.unsqueeze(0))[0]
        print("done 241")
        if self.n_frames > 1:
            if self.frames_chanel:
                image = image.view(self.n_frames, 3, image.size(1), image.size(2))
            else:
                image = (
                    image.view(1, self.n_frames, image.size(1), image.size(2))
                    .repeat(3, 1, 1, 1)
                    .transpose(0, 1)
                )
        else:
            if not self.frames_chanel:
                image = image.repeat(3, 1, 1)
        print("done 254")
        return image, y_img, y_patient


class AbdominalCropDataset(Dataset):
    """
    Dataset for training 2.5D crop classification models.

    Attributes:
        df_img (pandas DataFrame): Metadata containing image information.
        df_patient (pandas DataFrame): Metadata containing patient information.
        df_series (pandas DataFrame): Metadata containing information about image series.
        transforms (albu transforms): Transforms to apply to the images.
        frames_chanel (int): The number of frames to consider for channel stacking.
        n_frames (int): The number of frames to use.
        stride (int): The step size between frames.
        train (bool): Flag indicating whether the dataset is for training.
        sigmas (dict): Dictionary containing Gaussian sigmas for various organs.
    """

    def __init__(
        self,
        df_patient,
        df_img,
        transforms=None,
        frames_chanel=0,
        n_frames=0,
        stride=1,
        train=False,
        use_soft_target=False,
        df_series=None,
    ):
        """
        Constructor for the AbdominalCropDataset class.

        Args:
            df_patient (pandas DataFrame): Metadata containing patient information.
            df_img (pandas DataFrame): Metadata containing image information.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
            frames_chanel (int, optional): Number of frames to consider for channel stacking. Defaults to 0.
            n_frames (int, optional): The number of frames to use. Defaults to 0.
            stride (int, optional): The step size between frames. Defaults to 1.
            train (bool, optional): Flag indicating whether the dataset is for training. Defaults to False.
            use_soft_target (bool, optional): Flag indicating the use of soft targets. Defaults to False.
            df_series (pandas DataFrame, optional): Metadata containing info about series. Defaults to None.
        """
        self.df_img = df_img
        self.df_patient = df_patient
        self.df_series = (
            get_df_series(df_patient, df_img) if df_series is None else df_series
        )
        self.targets = self.df_series["target"].values

        self.transforms = transforms
        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride

        self.train = train

        self.sigmas = {"kidney": 0.15, "spleen": 0.2, "liver": 0.3}

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df_series)

    def __getitem__(self, idx):
        """
        Item accessor. Samples a random frame inside the organ.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [(N,) C, H, W].
            torch.Tensor: Label as a tensor of shape [3].
            int: Dummy value.
        """
        img = np.load(self.df_series["img_path"].values[idx])

        organ = self.df_series["organ"].values[idx]
        if organ == "kidney":
            d = int(img.shape[1] * 3 / 4)
            img = np.concatenate([img[:, :, :d], img[:, :, -d:]], -1)

        # Pick frame(s)
        if self.train:
            ps = np.exp(
                -(
                    (
                        (np.arange(len(img)) - len(img) // 2)
                        / (self.sigmas[organ] * len(img))
                    )
                    ** 2
                )
            )  # gaussian
            m = 5 + self.stride * (self.n_frames - 1) + self.frames_chanel
            ps[:m] = 0  # Stay in bounds
            ps[-m:] = 0  # Stay in bounds
            if ps.max():
                frame = np.random.choice(len(img), p=ps / ps.sum())
            else:
                frame = len(img) // 2 + np.random.choice([-2, -1, 0, 1, 2])
        else:
            frame = len(img) // 2  # center

        frames = get_frames(
            frame,
            self.n_frames,
            self.frames_chanel,
            stride=self.stride,
            max_frame=len(img) - 1,
        )

        # Load
        image = img[np.array(frames)].transpose(1, 2, 0)
        image = image.astype(np.float32) / 255.0

        # Augment
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        y_img = torch.zeros(3, dtype=torch.float)
        y_img[self.targets[idx]] = 1

        # Reshape
        if self.n_frames > 1:
            if self.frames_chanel:
                image = image.view(
                    self.n_frames, 3, image.size(1), image.size(2)
                )  # .transpose(0, 1)
            else:
                image = (
                    image.view(1, self.n_frames, image.size(1), image.size(2))
                    .repeat(2, 1, 1, 1)
                    .transpose(0, 1)
                )

        return image, y_img, 0


class AbdominalInfDataset(Dataset):
    """
    Dataset for infering 2D classification models.
    It is optimized to compute the CNN forward only once when models are 2.5D :
    Trick is to extract CNN features for all images,
    and then compute the sequential head by retrieving the indexed features.

    Attributes:
        df (pandas DataFrame): Metadata containing image information.
        transforms (albu transforms): Transforms to apply to the images.
        frames_chanel (int): The number of frames to consider for channel stacking.
        n_frames (int): The number of frames to use.
        stride (int): The step size between frames.
        imgs (dict): Dictionary for storing loaded images.
        features (list): List of precompted features.
        single_frame (bool): Flag indicating if only a single frame is used for each item.
    """

    def __init__(
        self,
        df,
        transforms=None,
        frames_chanel=0,
        n_frames=1,
        stride=1,
        imgs={},
        features=[],
        single_frame=False,
    ):
        """
        Constructor.
        The single frame flag is used for features precomputation.

        Args:
            df (pandas DataFrame): Metadata containing image information.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
            frames_chanel (int, optional): Number of frames to consider for channel stacking. Defaults to 0.
            n_frames (int, optional): The number of frames to use. Defaults to 1.
            stride (int, optional): The step size between frames. Defaults to 1.
            imgs (dict, optional): Dictionary for storing loaded images. Defaults to an empty dictionary.
            features (list, optional): List of precomputed features. Defaults to an empty list.
            single_frame (bool, optional): Whether a single frame is used for each item. Defaults to False.
        """
        self.df = df
        self.info = self.df[["path", "patient_id", "series", "frame"]].values
        self.transforms = transforms

        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride
        self.single_frame = single_frame

        self.max_frames = dict(df[["series", "frame"]].groupby("series").max()["frame"])

        self.imgs = imgs
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
        """
        Get keys for indexing features.

        Returns:
            list: List of keys.
        """
        keys = []
        for idx in range(len(self.df)):
            path, patient, series, frame = self.info[idx]
            frames = get_frames(
                frame,
                1,
                self.frames_chanel,
                stride=1,
                max_frame=self.max_frames[series],
            )
            key = f'{patient}_{series}_{"-".join(list(frames.astype(str)))}'
            keys.append(key)
        return keys

    def _getitem_feature(self, idx):
        """
        Item accessor for features.

        Args:
            idx (int): Index.

        Returns:
            np.ndarray: Features.
            int: Dummy value.
            int: Dummy value.
        """
        path, patient, series, frame = self.info[idx]

        all_frames = get_frames(
            frame,
            self.n_frames,
            self.frames_chanel,
            stride=self.stride,
            max_frame=self.max_frames[series],
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
        Refer to _getitem_feature if features are precomputed.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor.
            int: Dummy value.
            int: Dummy value.
        """
        if len(self.features):
            return self._getitem_feature(idx)

        path, patient, series, frame = self.info[idx]

        if self.single_frame:
            frames = get_frames(
                frame,
                1,
                self.frames_chanel,
                stride=1,
                max_frame=self.max_frames[series],
            )
        else:
            frames = get_frames(
                frame,
                self.n_frames,
                self.frames_chanel,
                stride=self.stride,
                max_frame=self.max_frames[series],
            )

        paths = [path.rsplit("_", 1)[0] + f"_{f:04d}.png" for f in frames]

        image = []
        for path, frame in zip(paths, frames):
            try:
                img = self.imgs[path]
            except Exception:
                img = cv2.imread(path, 0)
                if not (idx + 1 % 10000):  # clear buffer
                    self.imgs = {}
                self.imgs[path] = img

            image.append(img)

        image = np.array(image).transpose(1, 2, 0)
        image = image.astype(np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        if not self.single_frame:
            if self.n_frames > 1:
                if self.frames_chanel:
                    image = image.view(self.n_frames, 3, image.size(1), image.size(2))
                else:
                    image = (
                        image.view(1, self.n_frames, image.size(1), image.size(2))
                        .repeat(3, 1, 1, 1)
                        .transpose(0, 1)
                    )
        #     else:
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        return image, 0, 0


class SegDataset(Dataset):
    """
    Dataset for training segmentation models.
    Masks are not used in the pipeline here, we only use the classification part.

    Attributes:
        df (pandas DataFrame): Metadata containing image and mask information.
        for_classification (bool): Flag indicating whether the dataset is used for classification.
        use_soft_target (bool): Flag indicating whether soft targets are used.
        transforms (albu transforms): Transforms to apply to images and masks.

    """

    def __init__(
        self,
        df,
        for_classification=True,
        use_soft_target=False,
        transforms=None,
    ):
        """
        Constructor for the SegDataset class.

        Args:
            df (pandas DataFrame): Metadata containing image and mask information.
            for_classification (bool, optional): Whether the dataset is used for classif. Defaults to True.
            use_soft_target (bool, optional): Whether soft targets are used. Defaults to False.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df = df
        self.transforms = transforms
        self.for_classification = for_classification

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values

        targets = [
            "pixel_count_liver",
            "pixel_count_spleen",
            "pixel_count_left-kidney",
            "pixel_count_right-kidney",
            "pixel_count_bowel",
        ]
        if use_soft_target:
            self.img_targets = df[[c + "_norm" for c in targets]].values
        else:
            self.img_targets = df[targets].values > 100

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor.
            torch.Tensor: Mask as a tensor (if not for classification).
            torch.Tensor: Label as a tensor.
        """
        image = cv2.imread(self.img_paths[idx]).astype(np.float32) / 255.0

        y = torch.tensor(self.img_targets[idx], dtype=torch.float)

        if not self.for_classification:
            mask = cv2.imread(self.mask_paths[idx], 0)

            mask = np.where(mask == 4, 3, mask)
            mask = np.where(mask == 5, 4, mask)

            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.unsqueeze(0).float()

            return image, mask, y

        image = self.transforms(image=image)["image"]
        return image, y, 0


class Seg3dDataset(Dataset):
    """
    Dataset for training 3D segmentation models.

    Attributes:
        df (pandas DataFrame): Metadata containing image and mask information.
        train (bool): Flag indicating whether the dataset is used for training.
        test (bool): Flag indicating whether the dataset is used for testing.
    """

    def __init__(
        self,
        df,
        train=False,
        test=False,
    ):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata containing image and mask information.
            train (bool, optional): Whether the dataset is used for training. Defaults to False.
            test (bool, optional): Whether the dataset is used for testing. Defaults to False.
        """
        self.df = df
        self.train = train
        self.test = test

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values

        if train:
            import monai.transforms as transforms

            # https://docs.monai.io/en/0.3.0/transforms.html
            self.transforms = transforms.Compose(
                [
                    transforms.RandAffined(
                        translate_range=[256 * 0.1] * 3,
                        padding_mode="zeros",
                        keys=["image", "mask"],
                        prob=0.5,
                    ),
                    transforms.RandRotated(
                        range_x=(-0.3, 0.3),
                        range_y=(-0.3, 0.3),
                        range_z=(-0.3, 0.3),
                        mode="nearest",
                        keys=["image", "mask"],
                        prob=0.5,
                    ),
                    transforms.RandZoomd(
                        min_zoom=0.666,
                        max_zoom=1.5,
                        mode="nearest",
                        keys=["image", "mask"],
                        prob=0.5,
                    ),
                ]
            )
        else:
            self.transforms = None

        self.imgs = {}
        self.masks = {}
        if not test:
            for idx in range(len(self.img_paths)):
                self.imgs[self.img_paths[idx]] = np.load(self.img_paths[idx])[None]
                self.masks[self.mask_paths[idx]] = np.load(self.mask_paths[idx])[None]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor.
            torch.Tensor: Mask as a tensor (if not for testing).
            int: Dummy value.
        """
        image = self.imgs.get(
            self.img_paths[idx],
            np.load(self.img_paths[idx])[None],
        )

        if not self.test:
            mask = self.masks.get(
                self.mask_paths[idx], np.load(self.mask_paths[idx])[None]
            )
            # Merge both kidneys !
            mask = np.where(mask == 4, 3, mask)
            mask = np.where(mask == 5, 4, mask)
        else:
            mask = 0

        if self.transforms is not None:
            res = self.transforms({"image": image, "mask": mask})
            image = res["image"].as_tensor().float() / 255.0
            mask = res["mask"].as_tensor()
        else:
            image = torch.from_numpy(image).float() / 255.0
            if not self.test:
                mask = torch.from_numpy(mask)

        return image, mask, 0


class PatientFeatureDataset(Dataset):
    """
    Dataset for training RNN models.

    Attributes:
        df_patient (pandas DataFrame): Metadata containing patient information.
        df_img (pandas DataFrame): Metadata containing image information.
        exp_folders (list of tuples): Experiment folders and modes.
        max_len (int, optional): Maximum length for feature sequences. Defaults to None.
        restrict (bool, optional): Flag to restrict feature length. Defaults to False.
        resize (tuple, optional): Tuple specifying the size for resizing features. Defaults to None.
    """

    def __init__(
        self,
        df_patient,
        df_img,
        exp_folders,
        max_len=None,
        restrict=False,
        resize=None,
    ):
        """
        Constructor.

        Args:
            df_patient (pandas DataFrame): Metadata containing patient information.
            df_img (pandas DataFrame): Metadata containing image information.
            exp_folders (list of tuples): Experiment folders and modes.
            max_len (int, optional): Maximum length for feature sequences. Defaults to None.
            restrict (bool, optional): Flag to restrict feature length. Defaults to False.
            resize (tuple, optional): Tuple specifying the size for resizing features. Defaults to None.
        """
        self.df_patient = df_patient
        self.fts, self.crop_fts = self.retrieve_features(df_img, exp_folders)
        self.ids = list(self.fts.keys())
        self.max_len = max_len
        self.restrict = restrict
        self.resize = resize

    def retrieve_features(self, df, exp_folders):
        """
        Retrieve and organize features from experiment folders.

        Args:
            df (pandas DataFrame): Metadata containing image information.
            exp_folders (list of tuples): Experiment folders and modes.

        Returns:
            dict: Features dictionary.
            dict: Crop features dictionary.
        """
        features_dict = {}
        crop_features_dict = {}
        for fold in sorted(df["fold"].unique()):
            df_val = df[df["fold"] == fold].reset_index(drop=True)

            fts = []
            for exp_folder, mode in exp_folders:
                if mode == "seg":
                    seg = np.load(exp_folder + f"pred_val_{fold}.npy")
                    fts.append(seg)
                elif mode == "crop":
                    continue
                else:  # proba
                    ft = np.load(exp_folder + f"pred_val_{fold}.npy")
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
                                ft[:, 1:2]
                                * seg.max(-1, keepdims=True),  # extravasation
                                ft[:, 2:5] * kidney,  # kidney
                                ft[:, 5:8] * seg[:, :1],  # liver
                                ft[:, 8:] * seg[:, 1:2],  # spleen
                            ],
                            -1,
                        )
                    )
            try:
                fts = np.concatenate(fts, axis=1)
            except Exception:
                fts = np.zeros(len(df_val))

            df_val["index"] = np.arange(len(df_val))
            slice_starts = (
                df_val.groupby(["patient_id", "series"])["index"].min().to_dict()
            )
            slice_ends = (
                df_val.groupby(["patient_id", "series"])["index"].max() + 1
            ).to_dict()

            for k in slice_starts.keys():
                start = slice_starts[k]
                end = slice_ends[k]

                if df_val["frame"][start] < df_val["frame"][end - 1]:
                    features_dict[k] = fts[start:end]
                else:
                    features_dict[k] = fts[start:end][::-1]

            crop_fts = []
            for exp_folder, mode in exp_folders:
                if mode == "crop":
                    if not len(df_val):
                        continue

                    preds = np.load(exp_folder + f"pred_val_{fold}.npy")
                    df_series = get_df_series(
                        self.df_patient[self.df_patient["fold"] == fold],
                        df_val,
                    )

                    for i, c in enumerate(["pred_healthy", "pred_low", "pred_high"]):
                        df_series[c] = preds[:, i]
                    df_series = (
                        df_series.groupby(["patient_id", "series"])
                        .agg(list)
                        .reset_index()
                    )

                    i = 2
                    crop_scores = np.array(
                        [
                            np.array(df_series[p].values.tolist())
                            for p in ["pred_healthy", "pred_low", "pred_high"]
                        ]
                    ).transpose(1, 2, 0)
                    crop_fts.append(crop_scores)

            if len(crop_fts):
                crop_scores = np.concatenate(crop_fts, -1)
                for i, (p, s) in enumerate(df_series[["patient_id", "series"]].values):
                    try:
                        _ = features_dict[(p, s)]
                        crop_features_dict[(p, s)] = crop_scores[i]  # cls x score
                    except KeyError:
                        print(p, s)

        return features_dict, crop_features_dict

    def __len__(self):
        return len(self.fts)

    @staticmethod
    def restrict_fts(fts):
        """
        Restrict the length of features.

        Args:
            fts (numpy.ndarray): Features array.

        Returns:
            numpy.ndarray: Restricted features array.
        """
        if len(fts) > 400:
            fts = fts[len(fts) // 6 :]
        else:
            fts = fts[len(fts) // 8 :]
        return fts

    @staticmethod
    def resize_fts(fts, size, max_len=None):
        """
        Resize features.

        Args:
            fts (numpy.ndarray): Features array.
            size (tuple): Size for resizing.
            max_len (int, optional): Maximum length for features. Defaults to None.

        Returns:
            numpy.ndarray: Resized features array.
        """
        if max_len is not None:  # crop too long
            fts = fts[-max_len:]

        fts = fts[::2].copy()

        fts = F.interpolate(
            torch.from_numpy(fts.T).float().unsqueeze(0), size=size, mode="linear"
        )[0].transpose(0, 1)
        return fts

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            dict: Features and crop features (if available).
            torch.Tensor: Label as a tensor.
            int: Dummy value.
        """
        patient_series = self.ids[idx]

        fts = self.fts[patient_series]
        crop_fts = self.crop_fts.get(patient_series, None)

        if self.restrict:
            fts = self.restrict_fts(fts)

        if self.resize:
            fts = self.resize_fts(fts, self.resize, self.max_len)
        else:
            if self.max_len is not None:
                fts = self.pad(fts)
            fts = torch.from_numpy(fts).float()

        if crop_fts is not None:
            crop_fts = torch.from_numpy(crop_fts).float()
        else:
            crop_fts = 0

        y = self.df_patient[self.df_patient["patient_id"] == patient_series[0]][
            PATIENT_TARGETS
        ].values[0]

        y = torch.from_numpy(y).float()  # bowel, extravasion, kidney, liver, spleen

        return {"x": fts, "ft": crop_fts}, y, 0
