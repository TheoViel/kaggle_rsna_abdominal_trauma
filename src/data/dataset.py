import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from params import PATIENT_TARGETS, IMAGE_TARGETS, SEG_TARGETS, IMG_TARGETS_EXTENDED


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


class AbdominalDataset(Dataset):
    def __init__(
        self,
        df_patient,
        df_img,
        transforms=None,
        frames_chanel=0,
        train=False,
        use_soft_target=False,
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
        self.train = train
        self.use_soft_target = use_soft_target
    
        self.targets = df_patient[PATIENT_TARGETS].values
        self.max_frames = dict(df_img[["series", "frame"]].groupby("series").max()['frame'])

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df_patient) * 5
            
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
        tgt_idx = idx % 5
        tgt = IMG_TARGETS_EXTENDED[tgt_idx]
        
        idx = idx // 5
        patient = self.df_patient['patient_id'].values[idx]
        y_patient = self.targets[idx]

        df_img = self.df_img[self.df_img['patient_id'] == patient]

        # Restrict to considered class
        if (df_img[IMG_TARGETS_EXTENDED[tgt_idx]] == y_patient[tgt_idx]).max():  
            df_img = df_img[df_img[IMG_TARGETS_EXTENDED[tgt_idx]] == y_patient[tgt_idx]]
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
        
        # Pick middle row
        if self.train:
            row = df_img.iloc[np.random.choice(len(df_img))]
        else:
            row = df_img.iloc[len(df_img) // 2]  # center
        
        if self.frames_chanel > 0:
            frame = row.frame
            frame = np.clip(frame, self.frames_chanel, self.max_frames[series] - self.frames_chanel)
            paths = [row.path.rsplit('_', 1)[0] + f'_{f:04d}.png' for f in [frame - self.frames_chanel, frame, frame + self.frames_chanel]]
            image = np.array([cv2.imread(path, 0) for path in paths]).transpose(1, 2, 0)
        else:
            image = cv2.imread(row.path)
        image = image.astype(np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        y_patient = torch.tensor(y_patient, dtype=torch.float)
        y_img = torch.tensor(row[IMG_TARGETS_EXTENDED], dtype=torch.float)
        
#         print(row)
        
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

        if image.size(0) > 3:
            image = image.view(3, -1, image.size(1), image.size(2)).transpose(0, 1)

        return image, y_img, y_patient


class Abdominal2DInfDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
        frames_chanel=0,
        imgs={}
    ):
        """
        Constructor.

        Args:
            df_img (pandas DataFrame): Metadata containing information about the dataset.
            df_patient (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df = df
        self.info = self.df[['path', 'series', 'frame']].values
        self.transforms = transforms
        self.frames_chanel = frames_chanel
        self.max_frames = dict(df[["series", "frame"]].groupby("series").max()['frame'])
        
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

        if self.frames_chanel > 0:
            frame = np.clip(frame, self.frames_chanel, self.max_frames[series] - self.frames_chanel)
            frames = [frame - self.frames_chanel, frame, frame + self.frames_chanel]
            paths = [path.rsplit('_', 1)[0] + f'_{f:04d}.png' for f in frames]
            
            image = []
            for path, frame in zip(paths, frames):
                try:
                    img = self.imgs[path]
#                     print('!!')
                except:
                    img = cv2.imread(path, 0)  # self.imgs.get(frame, cv2.imread(path, 0))

                    if not (idx + 1 % 10000):  # clear buffer
                        self.imgs = {}
                    self.imgs[path] = img

                image.append(img)
            image = np.array(image).transpose(1, 2, 0)

        else:
            try:
                image = self.imgs[path]
                if len(image.shape) == 2:
                    image = np.concatenate([image[:, :, None]] * 3, -1)
            except:
                image = cv2.imread(path)

        image = image.astype(np.float32) / 255.

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image, 0, 0

    
class PatientFeatureDataset(Dataset):
    def __init__(self, df_patient, df_img, exp_folders, max_len=None):
        self.df_patient = df_patient
        self.fts = self.retrieve_features(df_img, exp_folders)
        self.ids = list(self.fts.keys())
        self.max_len = max_len

    @staticmethod
    def retrieve_features(df, exp_folders):
        features_dict = {}
        for fold in sorted(df['fold'].unique()):
            df_val = df[df['fold'] == fold].reset_index(drop=True)
            
            fts = []
            for exp_folder, mode in exp_folders:
#                 print(exp_folder, mode)
                prefix = 'fts' if "ft" in mode else 'pred'
                ft = np.load(exp_folder + f"{prefix}_val_{fold}.npy")
        
#                 if mode == "seg":
                fts.append(ft)
            
                if "proba" in mode:
                    seg = fts[0]
                    if ft.shape[-1] == 11:
                        fts.append(np.concatenate([
                            ft[:, :1] * seg[:, -1:],  # bowel
                            ft[:, 1:2] * seg.max(-1, keepdims=True),  # extravasation
                            ft[:, 2: 5] * seg[:, 2: 4].max(-1, keepdims=True),  # kidney
                            ft[:, 5: 8] * seg[:, :1],  # liver
                            ft[:, 8:] * seg[:, 1:2],  # spleen
                        ], -1))
                
            
            fts = np.concatenate(fts, axis=1)

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
        return features_dict

    def pad(self, x):
        length = x.shape[0]
        if length > self.max_len:
            return x[: self.max_len]
        else:
            padded = np.zeros([self.max_len] + list(x.shape[1:]))
            padded[:length] = x
            return padded
        
    def __len__(self):
        return len(self.fts)

    def __getitem__(self, idx):
        
        patient_study = self.ids[idx]

        fts = self.fts[patient_study]
        if self.max_len is not None:
            fts = self.pad(fts)
        fts = torch.from_numpy(fts).float()
        
        y = self.df_patient[self.df_patient['patient_id'] == patient_study[0]][PATIENT_TARGETS].values[0]
        y = torch.from_numpy(y).float()  # bowel, extravasion, kidney, liver, spleen

        return fts, y, 0


class SegDataset(Dataset):
    """
    Custom dataset for Segmentation data.
    """

    def __init__(
        self,
        df,
        for_classification=False,
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
            
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.unsqueeze(0).float()
            
            return image, mask, y
        
        image = self.transforms(image=image)["image"]
        return image, y, 0
    

class PatientFeatureInfDataset(Dataset):
    def __init__(self, series, exp_folders, max_len=None, save_folder=""):
        self.fts = self.retrieve_features(series, exp_folders, save_folder=save_folder)
        self.ids = [0]
        self.max_len = max_len

    @staticmethod
    def retrieve_features(series, exp_folders, save_folder=""):
        all_fts = []
        exp_names = ["_".join(exp_folder.split('/')[-3:-1]) for exp_folder, _ in exp_folders]

        for s in series:
            fts = []
            for exp_name, (exp_folder, mode) in zip(exp_names, exp_folders):
                prefix = 'fts' if "ft" in mode else 'pred'
                ft = np.load(save_folder + f'{s}_{exp_name}.npy')
                fts.append(ft)

                if "proba" in mode:
                    seg = fts[0]
                    if ft.shape[-1] == 11:
                        fts.append(np.concatenate([
                            ft[:, :1] * seg[:, -1:],  # bowel
                            ft[:, 1:2] * seg.max(-1, keepdims=True),  # extravasation
                            ft[:, 2: 5] * seg[:, 2: 4].max(-1, keepdims=True),  # kidney
                            ft[:, 5: 8] * seg[:, :1],  # liver
                            ft[:, 8:] * seg[:, 1:2],  # spleen
                        ], -1))

            fts = np.concatenate(fts, axis=1)
            all_fts.append(fts)
            
        return all_fts

    def pad(self, x):
        length = x.shape[0]
        if length > self.max_len:
            return x[: self.max_len]
        else:
            padded = np.zeros([self.max_len] + list(x.shape[1:]))
            padded[:length] = x
            return padded
        
    def __len__(self):
        return len(self.fts)

    def __getitem__(self, idx):
        fts = self.fts[idx]
        if self.max_len is not None:
            fts = self.pad(fts)
        fts = torch.from_numpy(fts).float()

        return fts, 0, 0
