import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.transforms import get_transfos
from params import PATIENT_TARGETS, IMAGE_TARGETS

from tqdm import tqdm


class Abdominal2DDataset(Dataset):
    def __init__(
        self,
        df_patient,
        df_img,
        transforms=None,
        train=False,
        pos_prop=0.5,
    ):
        """
        Constructor.

        Args:
            df_img (pandas DataFrame): Metadata containing information about the dataset.
            df_patient (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df_img = df_img
        self.df_patient = df_patient
        self.transforms = transforms
        
        self.train = train
        self.pos_prop = pos_prop
        
        if not self.train:
            self.img_targets = self._get_img_targets()

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df_patient)
    
    def _get_img_targets(self):
        img_targets = []
        for idx in range(len(self.df_patient)):
            patient = self.df_patient['patient_id'].values[idx]
            df_img = self.df_img[self.df_img['patient_id'] == patient]
            df_img = df_img.sort_values('frame').reset_index(drop=True)
            
            if df_img[IMAGE_TARGETS].max().max() and (idx % int(1 / self.pos_prop)):  # positive
                df_img = df_img[df_img[IMAGE_TARGETS].max(1) > 0].reset_index(drop=True)
                df_img = df_img[df_img['series'] == df_img['series'].values[0]]
            else:
                df_img = df_img[df_img[IMAGE_TARGETS].max(1) == 0].reset_index(drop=True)
                df_img = df_img[df_img['series'] == df_img['series'].values[0]]
            row = df_img.iloc[len(df_img) // 2]  # center
            img_targets.append(row[IMAGE_TARGETS].values)
        return np.array(img_targets).astype(int)
            
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
        patient = self.df_patient['patient_id'].values[idx]

        df_img = self.df_img[self.df_img['patient_id'] == patient]
        df_img = df_img.sort_values('frame').reset_index(drop=True)
        
        if self.train:
            df_pos = df_img[df_img[IMAGE_TARGETS].max(1) > 0].reset_index(drop=True)
            if (np.random.random() < self.pos_prop) and len(df_pos):  # random positive
                df_pos = df_pos[
                    df_pos['series'] == np.random.choice(df_pos['series'].unique())
                ]
                row = df_pos.iloc[np.random.choice(len(df_pos))]
            else:
                df_neg = df_img[df_img[IMAGE_TARGETS].max(1) == 0].reset_index(drop=True)
                df_neg = df_neg[
                    df_neg['series'] == np.random.choice(df_neg['series'].unique())
                ]
                row = df_neg.iloc[np.random.choice(len(df_neg))]
        else:  # sample an image with y_patient[:2] == y_img ?? - should probably sample in segmentation
#             print(df_img[IMAGE_TARGETS].max().max())
            if df_img[IMAGE_TARGETS].max().max() and (idx % int(1 / self.pos_prop)):  # positive
                # positive + first series
                df_img = df_img[df_img[IMAGE_TARGETS].max(1) > 0].reset_index(drop=True)
                df_img = df_img[df_img['series'] == df_img['series'].values[0]]
            else:
                df_img = df_img[df_img[IMAGE_TARGETS].max(1) == 0].reset_index(drop=True)
                df_img = df_img[df_img['series'] == df_img['series'].values[0]]

            row = df_img.iloc[len(df_img) // 2]  # center

#         print(row)
        
        image = cv2.imread(row.path)
        image = image.astype(np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        y_patient = torch.tensor(self.df_patient[PATIENT_TARGETS].values[idx], dtype=torch.float)
        y_img = torch.tensor(row[IMAGE_TARGETS], dtype=torch.float)

#         if image.size(0) > 3:
#             image = image.view(3, -1, image.size(1), image.size(2)).transpose(0, 1)

        return image, y_img, y_patient


class Abdominal2DInfDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
    ):
        """
        Constructor.

        Args:
            df_img (pandas DataFrame): Metadata containing information about the dataset.
            df_patient (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albu transforms, optional): Transforms to apply to images and masks. Defaults to None.
        """
        self.df = df
        self.transforms = transforms

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
        image = cv2.imread(self.df['path'].values[idx])
        image = image.astype(np.float32) / 255.0

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
            fts = np.concatenate([
                np.load(exp_folder + f"fts_val_{fold}.npy")  for exp_folder in exp_folders
            ], axis=1)

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
