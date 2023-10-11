import os
import re
import cv2
import time
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset
from data.preparation import center_crop_pad, get_df_series
from params import PATIENT_TARGETS, IMAGE_TARGETS, SEG_TARGETS, IMG_TARGETS_EXTENDED, CROP_TARGETS


def to_one_hot_patient(y):
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


def get_frames(frame, n_frames, frames_c, stride=1, max_frame=100, extrav=False):
    if stride == -1:
        if extrav:
            frames = np.linspace(
                1, max_frame - 1, n_frames, endpoint=True, dtype=int
            )
        else:
            frames = np.linspace(
                0, max_frame, n_frames + 4, endpoint=True, dtype=int
            )[2: -2]
        
    else:
        frames = np.arange(n_frames) * stride
    frames = frames - frames[n_frames // 2] + frame
    
    if frames_c:
        offset = np.tile(np.arange(-1, 2) * frames_c, len(frames))
        frames = np.repeat(frames, 3) + offset

    if frames.min() < 0:  # BUG ??
        frames -= frames.min() # - 1
    elif frames.max() > max_frame:
        frames += max_frame - frames.max()

    frames = np.clip(frames, 0, max_frame)
    return frames


class AbdominalDataset(Dataset):
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
        use_mask=False,
        use_crops=False,
        bowel_extrav_only=False,
    ):
        """
        Constructor.

        Args:
            df_patient (pandas DataFrame): Metadata containing information about the dataset.
            df_img (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df_img = df_img
        self.df_patient = df_patient
        self.transforms = transforms
        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride
        self.train = train
        self.use_soft_target = use_soft_target
        
        self.bowel_extrav_only = bowel_extrav_only
        self.classes = IMG_TARGETS_EXTENDED[:2] if bowel_extrav_only else IMG_TARGETS_EXTENDED
    
        self.targets = df_patient[PATIENT_TARGETS].values
        self.max_frames = dict(df_img[["series", "frame"]].groupby("series").max()['frame'])
        
        self.use_mask = use_mask
        self.mask_folder = "../logs/2023-09-24/20/masks/"
        
        self.sigmas = {
            "kidney_injury": 0.4,  # 0.15,
            "spleen_injury": 0.4,  # 0.2,
            "liver_injury": 0.4,  # 0.3,
            "bowel_injury": 0.4,
            "extravasation_injury": 0.4, # 0.1,
        }
        
        self.use_crops = use_crops

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

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [C, H, W].
            torch.Tensor: Label as a tensor of shape [5].
            torch.Tensor: Aux label as a tensor of shape [5].
        """
        tgt_idx = idx % len(self.classes)
        tgt = self.classes[tgt_idx]

        idx = idx // len(self.classes)
        patient = self.df_patient['patient_id'].values[idx]
        y_patient = self.targets[idx]

        df_img = self.df_img[self.df_img['patient_id'] == patient]

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
        series = np.random.choice(df_img['series'].unique()) if self.train else df_img['series'].values[0]
        df_img = df_img[df_img['series'] == series]
        
        # Sort by frame
        df_img = df_img.sort_values('frame').reset_index(drop=True)
        
        if self.use_mask:
            if any(df_img['frame'].values > (self.max_frames[series] - 600)):
                df_img = df_img[df_img['frame'] > (self.max_frames[series] - 600)].reset_index(drop=True)
        
        # Pick a row
        if self.train:
            ps = np.exp(-((np.arange(len(df_img)) - len(df_img) // 2) / (self.sigmas[tgt] * len(df_img))) ** 2)  # gaussian
            row_idx = np.random.choice(len(df_img), p=ps / ps.sum())
            
#             if self.frames > 1:
#                 ps[:self.stride + self.frames_chanel] = 0  # Stay in bounds
#                 ps[-self.stride + self.frames_chanel:] = 0  # Stay in bounds

#                 if ps.max():
#                     row_idx = np.random.choice(len(df_img), p=ps / ps.sum())
#                 else:
#                     row_idx = len(df_img) // 2 + np.random.choice([-2, -1, 0, 1, 2])

#                 try:
#                     row = df_img.iloc[row_idx]
#                 except:
#                     # TODO : decrease sigma ?
#                     ps = np.exp(-((np.arange(len(df_img)) - len(df_img) // 2) / (0.4 * len(df_img))) ** 2)
#                     row_idx = np.random.choice(len(df_img), p=ps / ps.sum())
#     #                 row_idx = np.random.choice(len(df_img))
            row = df_img.iloc[row_idx]
                
        else:
            row = df_img.iloc[len(df_img) // 2]  # center
        
        if self.frames_chanel > 0 or self.n_frames > 1:
            frame = row.frame
            
            if self.n_frames <= 1:
                frame = np.clip(frame, self.frames_chanel, self.max_frames[series] - self.frames_chanel)
                frames = [frame - self.frames_chanel, frame, frame + self.frames_chanel]
            else:
                frames = get_frames(
                    frame, self.n_frames, self.frames_chanel, stride=self.stride, max_frame=self.max_frames[series]
                )
    
            prefix = row.path.rsplit('_', 1)[0]
            paths = [prefix + f'_{f:04d}.png' for f in frames]
            image = np.array([cv2.imread(path, 0) for path in paths]).transpose(1, 2, 0)

        else:
            frame = row.frame
            image = cv2.imread(row.path)

        image = image.astype(np.float32) / 255.0

        if self.use_crops:
            ref_size = 512
            if image.shape[1] != ref_size:
                h = int(ref_size / image.shape[1] * image.shape[0])
                image = cv2.resize(image, (ref_size, h))
#             else:
#                 continue
            x_start, x_end, y_start, y_end = row.x_start, row.x_end, row.y_start, row.y_end
            image = image[x_start: x_end, y_start: y_end]

#         print(image.shape)
        
        if self.use_mask:
            raise NotImplementedError
#             if os.path.exists(self.mask_folder + f'mask_{patient}_{series}_{frame:04d}.png'):
#                 mask = cv2.imread(self.mask_folder + f'mask_{patient}_{series}_{frame:04d}.png', 0)
#             else:
#                 mask = np.zeros((384, 384), dtype=np.uint8)
#             image = center_crop_pad(image.transpose(2, 0, 1), mask.shape[0]).transpose(1, 2, 0)
        
        if self.transforms:
#             if self.use_mask:
#                 transformed = self.transforms(image=image, mask=mask)
#                 image = transformed["image"]
#                 mask = transformed["mask"]
#             else:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        y_patient = torch.tensor(y_patient, dtype=torch.float)
        y_img = torch.tensor(row[self.classes], dtype=torch.float)
        
        if y_img.size(-1) == 5:  # Patient level - TODO : y_patient ?
            y_img = to_one_hot_patient(y_img.unsqueeze(0))[0]

            if self.use_soft_target:
                y_img[0] *= row.pred_bowel
                y_img[3:5] *= row.pred_kidney
                y_img[2] = 1 - y_img[3] - y_img[4]
                y_img[6:8] *= row.pred_liver
                y_img[5] = 1 - y_img[6] - y_img[7]
                y_img[9:] *= row.pred_spleen
                y_img[8] = 1 - y_img[9] - y_img[10]
                
#         if self.use_mask:
#             if self.frames_chanel > 0:
#                 image = torch.cat([image, mask.float().unsqueeze(0)], 0)
#             else:
#                 image[-1] = mask.float()

#         print(image.size())
        if self.n_frames > 1:
            if self.frames_chanel:
                image = image.view(
                    self.n_frames, 3, image.size(1), image.size(2)
                )  # .transpose(0, 1)
            else:
                image = image.view(
                    1, self.n_frames, image.size(1), image.size(2)
                ).repeat(3, 1, 1, 1).transpose(0, 1)
        else:
            if not self.frames_chanel:
                image = image.repeat(3, 1, 1)

        return image, y_img, y_patient


class AbdominalCropDataset(Dataset):
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
        use_mask=False,
        df_series=None
    ):
        """
        Constructor.

        Args:
            df_patient (pandas DataFrame): Metadata containing information about the dataset.
            df_img (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df_img = df_img
        self.df_patient = df_patient
        self.df_series = get_df_series(df_patient, df_img) if df_series is None else df_series
        self.targets = self.df_series["target"].values

        self.transforms = transforms
        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride

        self.train = train
        self.use_mask = use_mask
        
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
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [C, H, W].
            torch.Tensor: Label as a tensor of shape [5].
            torch.Tensor: Aux label as a tensor of shape [5].
        """
        img = np.load(self.df_series['img_path'].values[idx])
        
        if self.use_mask:
            mask = np.load(self.df_series['mask_path'].values[idx])
#             if mask.shape != img.shape:
#                 print(f'Mask shape {mask.shape}, img shape {img.shape}')
            
        organ = self.df_series['organ'].values[idx]
        if organ == "kidney":
            d = int(img.shape[1] * 3 / 4)
            img = np.concatenate([img[:, :, :d], img[:, :, -d:]], -1)
            if self.use_mask:
                mask = np.concatenate([mask[:, :, :d], mask[:, :, -d:]], -1)
                
        if self.use_mask:
            classes = ["", 'liver', 'spleen', 'kidney']
            mask = (mask == classes.index(organ)).astype(int)
        
        # Pick frame(s)
        if self.train:
            ps = np.exp(-((np.arange(len(img)) - len(img) // 2) / (self.sigmas[organ] * len(img))) ** 2)  # gaussian
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
            frame, self.n_frames, self.frames_chanel, stride=self.stride, max_frame=len(img) - 1
        )
#         print(frames)
        
        # Load
        image = img[np.array(frames)].transpose(1, 2, 0)
        image = image.astype(np.float32) / 255.0
        if self.use_mask:
            if self.frames_chanel:
                mask = mask[np.array(frames)[1:][::3]].transpose(1, 2, 0)
            else:
                mask = mask[np.array(frames)].transpose(1, 2, 0)
        
        # Augment
        if self.transforms:
            if self.use_mask:
                transformed = self.transforms(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
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
                image = image.view(
                    1, self.n_frames, image.size(1), image.size(2)
                ).repeat(3 if self.use_mask else 2, 1, 1, 1).transpose(0, 1)
            if self.use_mask:
                mask = mask.transpose(1, 2).transpose(0, 1).float()
                mask = mask.view(
                    1, self.n_frames, mask.size(1), mask.size(2)
                ).transpose(0, 1)
                image = torch.cat([image, mask], 1)

        return image, y_img, 0

    
class AbdominalExtravDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
        frames_chanel=0,
        n_frames=0,
        stride=1,
        train=False,
        use_soft_target=False,
        use_mask=False,
        df_series=None
    ):
        """
        Constructor.

        Args:
            df_patient (pandas DataFrame): Metadata containing information about the dataset.
            df_img (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df = df
        self.targets = self.df["target"].values

        self.transforms = transforms
        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride

        self.train = train
        self.use_mask = False

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
            torch.Tensor: Label as a tensor of shape [5].
            torch.Tensor: Aux label as a tensor of shape [5].
        """
        img = np.load(self.df['path'].values[idx])
        
        if img.shape[-1] == 3:
            img = img[..., 0]
                
        if self.use_mask:
            classes = ["", 'liver', 'spleen', 'kidney']
            mask = (mask == classes.index(organ)).astype(int)
            
#         return 0, 0, 0
        
        # Pick frame(s)
        if self.train:
            ps = np.exp(-((np.arange(len(img)) - len(img) // 2) / (0.15 * len(img))) ** 2)  # gaussian
            m = 5 + self.stride * (self.n_frames - 1) + self.frames_chanel
            ps[:m] = 0  # Stay in bounds
            ps[-m:] = 0  # Stay in bounds
            if ps.max():
                frame = np.random.choice(len(img), p=ps / ps.sum())
            else:
                frame = len(img) // 2 + np.random.choice([-2, -1, 0, 1, 2])
        else:
            frame = len(img) // 2  # center

#         print(img.shape, frame)
        frames = get_frames(
            frame,
            self.n_frames,
            self.frames_chanel,
            stride=self.stride,
            max_frame=len(img) - 1,
            extrav=True
        )
#         print(frames)
        
        # Load
        try:
            image = img[np.array(frames)].transpose(1, 2, 0)
        except ValueError:
            print(img.shape, frames)
            raise ValueError
        image = image.astype(np.float32) / 255.0
        
        # Augment
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
    
        y_img = torch.tensor([self.targets[idx]]).float()

        # Reshape
        if self.n_frames > 1:
            if self.frames_chanel:
                image = image.view(
                    self.n_frames, 3, image.size(1), image.size(2)
                )  # .transpose(0, 1)
            else:
                image = image.view(
                    1, self.n_frames, image.size(1), image.size(2)
                ).repeat(3, 1, 1, 1).transpose(0, 1)

        return image, y_img, 0


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
        features=[],
        single_frame=False,
        use_crops=False,
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
        self.single_frame = single_frame
        

        self.max_frames = dict(df[["series", "frame"]].groupby("series").max()['frame'])

        self.use_mask = use_mask
        self.mask_folder = "../logs/2023-09-24/20/masks/"
        
        self.imgs = imgs
        self.features = features

        if len(features):
            self.features = dict(zip(self.get_keys(), features))

        self.use_crops = use_crops
        if use_crops:
            self.coords = self.df[['x_start', "x_end", 'y_start', 'y_end']].values

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

        if self.single_frame:
            frames = get_frames(
                frame, 1, self.frames_chanel, stride=1, max_frame=self.max_frames[series]
            )
        else:
            frames = get_frames(
                frame, self.n_frames, self.frames_chanel, stride=self.stride, max_frame=self.max_frames[series]
            )
            
        paths = [path.rsplit('_', 1)[0] + f'_{f:04d}.png' for f in frames]

        image = []
        for path, frame in zip(paths, frames):
            try:
                img = self.imgs[path]
            except:
                img = cv2.imread(path, 0)
                if not (idx + 1 % 10000):  # clear buffer
                    self.imgs = {}
                self.imgs[path] = img
                
            if self.use_crops:
                ref_size = 512
                if img.shape[1] != ref_size:
                    h = int(ref_size / img.shape[1] * img.shape[0])
                    img = cv2.resize(img, (ref_size, h))
            image.append(img)

        image = np.array(image).transpose(1, 2, 0)
        image = image.astype(np.float32) / 255.
        
#         if self.use_crops:
# #             ref_size = 512
# #             if image.shape[1] != ref_size:
# #                 h = int(ref_size / image.shape[1] * image.shape[0])
# #                 image = cv2.resize(image, (ref_size, h))
#             x_start, x_end, y_start, y_end = row.x_start, row.x_end, row.y_start, row.y_end
#             image = image[x_start: x_end, y_start: y_end]
        if self.use_crops:
            x_start, x_end, y_start, y_end = self.coords[idx]
            image = image[x_start: x_end, y_start: y_end]

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

#         if image.size(0) == 1:
#             image = image.repeat(3, 1, 1)
        
        if not self.single_frame:
            if self.n_frames > 1:
                if self.frames_chanel:
                    image = image.view(
                        self.n_frames, 3, image.size(1), image.size(2)
                    )  # .transpose(0, 1)
                else:
                    image = image.view(
                        1, self.n_frames, image.size(1), image.size(2)
                    ).repeat(3, 1, 1, 1).transpose(0, 1)
#     else:
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        return image, 0, 0


class SegDataset(Dataset):
    """
    Custom dataset for Segmentation data.
    """

    def __init__(
        self,
        df,
        for_classification=False,
        use_soft_target=False,
        transforms=None,
    ):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
            use_soft_mask (bool, optional): Whether to use the soft mask or not. Defaults to False.
            use_shape_descript (bool, optional): Whether to use shape descriptors. Defaults to False.
            use_pl_masks (bool, optional): Whether to use pseudo-label masks. Defaults to False.
            frames (int or list, optional): Frame(s) to use for the false-color image. Defaults to 4.
            use_ext_data (bool, optional): Whether to use external data. Defaults to False.
            aug_strength (int, optional): Augmentation strength for external data. Defaults to 1.
        """
        self.df = df
        self.transforms = transforms
        self.for_classification = for_classification

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values

        if use_soft_target:
            self.img_targets = df[[c + "_norm" for c in SEG_TARGETS]].values
        else:
            self.img_targets = df[SEG_TARGETS].values > 100

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
            torch.Tensor: Image as a tensor of shape [C, H, W].
            torch.Tensor: Mask as a tensor of shape [1 or 7, H, W].
            torch.Tensor: Label as a tensor of shape [1].
        """
        image = cv2.imread(self.img_paths[idx]).astype(np.float32) / 255.  # 3 frames ?
        
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
    Custom dataset for Segmentation data.
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
            df (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
            use_soft_mask (bool, optional): Whether to use the soft mask or not. Defaults to False.
            use_shape_descript (bool, optional): Whether to use shape descriptors. Defaults to False.
            use_pl_masks (bool, optional): Whether to use pseudo-label masks. Defaults to False.
            frames (int or list, optional): Frame(s) to use for the false-color image. Defaults to 4.
            use_ext_data (bool, optional): Whether to use external data. Defaults to False.
            aug_strength (int, optional): Augmentation strength for external data. Defaults to 1.
        """
        self.df = df
        self.train = train
        self.test = test

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values
        
        if train:
            import monai.transforms as transforms
            # https://docs.monai.io/en/0.3.0/transforms.html
            self.transforms = transforms.Compose([
#                 transforms.RandFlipd(spatial_axis=1, keys=["image", "mask"], prob=0.5),
#                 transforms.RandFlipd(spatial_axis=2, keys=["image", "mask"], prob=0.5),
                transforms.RandAffined(
                    translate_range=[256 * 0.1] * 3,
                    padding_mode='zeros',
                    keys=["image", "mask"], 
                    prob=0.5
                ),
                transforms.RandRotated(
                    range_x=(-0.3, 0.3),
                    range_y=(-0.3, 0.3),
                    range_z=(-0.3, 0.3),
                    mode="nearest",
                    keys=["image", "mask"], 
                    prob=0.5
                ),
                transforms.RandZoomd(
                    min_zoom=0.666,
                    max_zoom=1.5,
                    mode="nearest",
                    keys=["image", "mask"], 
                    prob=0.5,
                ),
#                 transforms.RandGridDistortiond(
#                     distort_limit=(-0.01, 0.01),
#                     mode="nearest",
#                     keys=("image", "mask"), 
#                     prob=0.5,
#                 ),    
            ])
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
            torch.Tensor: Image as a tensor of shape [C, H, W].
            torch.Tensor: Mask as a tensor of shape [1 or 7, H, W].
            torch.Tensor: Label as a tensor of shape [1].
        """
        image = self.imgs.get(
            self.img_paths[idx],
            np.load(self.img_paths[idx])[None],
        )

        if not self.test:
            mask = self.masks.get(
                self.mask_paths[idx],
                np.load(self.mask_paths[idx])[None]
            )
            mask = np.where(mask == 4, 3, mask)
            mask = np.where(mask == 5, 4, mask)
        else:
            mask = 0

        if self.transforms is not None:
            res = self.transforms({'image': image, 'mask': mask})
            image = res['image'].as_tensor().float() / 255.
            mask = res['mask'].as_tensor()
        else:
            image = torch.from_numpy(image).float() / 255.
            if not self.test:
                mask = torch.from_numpy(mask)

        return image, mask, 0

    
class PatientFeatureDataset(Dataset):
    def __init__(self, df_patient, df_img, exp_folders, max_len=None, restrict=False, resize=None, use_other_series=False, refine_target=False):
        self.df_patient = df_patient
        self.fts, self.crop_fts = self.retrieve_features(df_img, exp_folders)
        self.ids = list(self.fts.keys())
        self.max_len = max_len
        self.restrict = restrict
        self.resize = resize
        self.use_other_series = use_other_series
        self.refine_target = refine_target
        
        if self.refine_target:
            self.series_tgts = df_img[
                ['patient_id', 'series', "extravasation_injury", "bowel_injury"]
            ].groupby(['patient_id', 'series']).max().to_dict()

    def retrieve_features(self, df, exp_folders):
        conf_extrav = None
        features_dict = {}
        crop_features_dict = {}
        for fold in sorted(df['fold'].unique()):
            df_val = df[df['fold'] == fold].reset_index(drop=True)
            
            fts = []
            for exp_folder, mode in exp_folders:
#                 print(exp_folder, mode)
                if mode == "seg":
                    seg = np.load(exp_folder + f"pred_val_{fold}.npy")
                    fts.append(seg)
                elif mode == "seg3d":
                    pass
                elif mode == "crop":
                    pass
                elif mode == "yolox":
                    file = sorted(glob.glob(exp_folder + ".npy"))[fold]
                    conf_extrav = np.load(file)
#                     fts.append(conf[:, None])
                elif mode == "bowel_extrav":
                    ft = np.load(exp_folder + f"pred_val_{fold}.npy")
                    assert ft.shape[-1] == 2
                    fts.append(np.concatenate([
                        ft[:, :1] * seg[:, -1:],  # bowel
                        ft[:, 1:2] * seg.max(-1, keepdims=True),  # extravasation
                    ], -1))      
#                     ft = ft[:len(df_val)]
                    fts.append(ft)
                else:  # proba
                    ft = np.load(exp_folder + f"pred_val_{fold}.npy")
                
                    if (conf_extrav is not None) and (exp_folder == "../logs/2023-09-20/36_r/"):
#                         print('Use yoloX')
                        ft[:, 1] = (conf_extrav + ft[:, 1]) / 2
#                     ft = ft[:len(df_val)]
                    fts.append(ft)

                    kidney = seg[:, 2: 4].max(-1, keepdims=True) if seg.shape[-1] == 5 else seg[:, 2: 3]
                    fts.append(np.concatenate([
                        ft[:, :1] * seg[:, -1:],  # bowel
                        ft[:, 1:2] * seg.max(-1, keepdims=True),  # extravasation
                        ft[:, 2: 5] * kidney,  # kidney
                        ft[:, 5: 8] * seg[:, :1],  # liver
                        ft[:, 8:] * seg[:, 1:2],  # spleen
                    ], -1))                
            try:
                fts = np.concatenate(fts, axis=1)
            except:
                fts = np.zeros(len(df_val))

            df_val["index"] = np.arange(len(df_val))
            slice_starts = df_val.groupby(['patient_id', 'series'])['index'].min().to_dict()
            slice_ends = (df_val.groupby(['patient_id', 'series'])['index'].max() + 1).to_dict()

            for k in slice_starts.keys():
                start = slice_starts[k]
                end = slice_ends[k]

                if df_val['frame'][start] < df_val['frame'][end - 1]:
                    features_dict[k] = fts[start: end]
                else:
                    features_dict[k] = fts[start: end][::-1]
                    
            crop_fts = []
            for exp_folder, mode in exp_folders:
                if mode == "crop":
                    if not len(df_val):
                        continue
                
                    preds = np.load(exp_folder + f'pred_val_{fold}.npy')
                    df_series = get_df_series(
                        self.df_patient[self.df_patient['fold'] == fold], df_val,
                    )

                    for i, c in enumerate(['pred_healthy', 'pred_low', 'pred_high']):
                        df_series[c] = preds[:, i]
                    df_series = df_series.groupby(["patient_id", "series"]).agg(list).reset_index()

                    i = 2
                    crop_scores = np.array([
                        np.array(df_series[p].values.tolist()) for p in ['pred_healthy', 'pred_low', 'pred_high']
                    ]).transpose(1, 2, 0)
                    crop_fts.append(crop_scores)

            if len(crop_fts):
                crop_scores = np.concatenate(crop_fts, -1)
                for i, (p, s) in enumerate(df_series[['patient_id', 'series']].values):
                    try:
                        _ = features_dict[(p, s)]
                        crop_features_dict[(p, s)] = crop_scores[i]  # cls x score
                    except KeyError:
                        print(p, s)
                    
        return features_dict, crop_features_dict

    @staticmethod
    def restrict_fts(fts):
        if len(fts) > 400:
            fts = fts[len(fts) // 6:]
        else:
            fts = fts[len(fts) // 8:]
        return fts
    
    @staticmethod
    def resize_fts(fts, size, max_len=None):
        if max_len is not None:  # crop too long
            fts = fts[-max_len:]
            
        fts = fts[::2].copy()
            
        fts = F.interpolate(
            torch.from_numpy(fts.T).float().unsqueeze(0),
            size=size,
            mode="linear"
        )[0].transpose(0, 1)
        return fts
        
    def __len__(self):
        return len(self.fts)
    
#     @staticmethod
    def detect_start_end(self, x, margin=20):
        seg = x[:, :5].copy()
        seg[1:] += seg[:-1]
        seg[:-1] += seg[1:]
        seg = seg / 3

        kept = (seg.max(-1) > (seg.max() * 0.99)).astype(int)
        kept[1:] += kept[:-1]
        kept[:-1] += kept[1:]
        kept = ((kept / 3) >= 0.9).astype(int)
        
        start = np.clip(np.argmax(kept) - margin, 0, len(x))
        end = np.clip(len(kept) - np.argmax(kept[::-1]) + margin, 0, len(x))
     
        return start, end

    def __getitem__(self, idx):
        patient_series = self.ids[idx]

        fts = self.fts[patient_series]
        crop_fts = self.crop_fts.get(patient_series, None)
        
#         other_fts = None
#         if self.use_other_series:
#             other_series = [
#                 k for k in self.fts.keys()
#                 if (k[0] == patient_series[0] and k[1] !=  patient_series[1])
#             ]
#             if len(other_series):
#                 other_series = other_series[0]
#                 other_fts = self.fts[other_series]
#             else:
#                 other_fts = fts.copy()
                
#             other_fts = other_fts[::2].copy()
# #             other_seg, other_fts = other_fts[:, :5], other_fts[:, 5:].reshape(other_fts.shape[0], 11, -1)
# #             other_fts = other_fts[:, 1]
# #                 print(other_fts.shape)

# #                 print(other_fts.shape, fts.shape)
#             if self.restrict:
#                 other_fts = self.restrict_fts(other_fts)

#             if self.resize:
#                 other_fts = self.resize_fts(other_fts, self.resize, self.max_len)
#             else:
#                 if self.max_len is not None:
#                     other_fts = self.pad(other_fts)
#                 other_fts = torch.from_numpy(other_fts).float()
#         else:
        other_fts = 0

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
        
        y = self.df_patient[self.df_patient['patient_id'] == patient_series[0]][PATIENT_TARGETS].values[0]
        
#         print(y)
        if self.refine_target:
            y[0] = self.series_tgts["bowel_injury"][patient_series]
            y[1] = self.series_tgts["extravasation_injury"][patient_series]
#         print(y)
        
        y = torch.from_numpy(y).float()  # bowel, extravasion, kidney, liver, spleen

        return {"x": fts, "ft": crop_fts, "other_x": other_fts}, y, 0
