import os
import glob
import nibabel
import numpy as np
import pandas as pd

from params import SEG_TARGETS


def prepare_folds(data_path="../input/", k=4):
    """
    Prepare data folds for cross-validation.
    MultilabelStratifiedKFold is used.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".
        k (int, optional): Number of cross-validation folds. Defaults to 4.

    Returns:
        pandas DataFrame: DataFrame containing the patient IDs and their respective fold assignments.
    """
    cols = [
        'bowel_injury', 'extravasation_injury', 'kidney_low',
        'kidney_high', 'liver_low', 'liver_high', 'spleen_low', 'spleen_high'
    ]

    df = pd.read_csv(data_path + "train.csv")

    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    mskf = mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    splits = mskf.split(df, y=df[cols])

    df['fold'] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i

    df_folds = df[["patient_id", "fold"]]
    df_folds.to_csv(data_path + f"folds_{k}.csv", index=False)
    return df_folds


def prepare_data(data_path="../input/", with_seg=True):
    """
    Prepare data for training or evaluation.
    TODO

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".

    Returns:
        pandas DataFrame: DataFrame containing the prepared data for training or evaluation.
    """
    df_patient = pd.read_csv(os.path.join(data_path, "df_train.csv"))
    
    if not with_seg:
        df_img = pd.read_csv(os.path.join(data_path, "df_images_train.csv"))
        df_img = df_img.rename(columns={"patient": "patient_id"})

    else:
        df_img = pd.read_csv(os.path.join(data_path, "df_images_train_with_seg.csv"))
        df_img['pred_extravasation'] = 1  # df_img[["pred_liver", "pred_spleen", "pred_bowel", "pred_kidney"]].max(1)
        
        df_img = df_img.merge(df_patient[["patient_id", "kidney", "liver", "spleen", "any_injury"]], how="left")
        #"kidney_low", "kidney_high", "liver_low", "liver_high"]] for soft labels ?
        for col in ["kidney", "liver", "spleen"]:
            df_img[f'{col}_injury'] = (df_img[f'pred_{col}'] > 0.9) * df_img[col]

    return df_patient, df_img


def prepare_seg_data(data_path="", use_3d=False):
    if use_3d:
        return pd.read_csv(data_path + 'df_seg_3d.csv')

    df_seg = pd.read_csv(data_path + 'df_seg.csv')
    
    df_seg['pixel_count_kidney'] = df_seg['pixel_count_left-kidney'] + df_seg['pixel_count_right-kidney']

    # Add pixel prop tgt
    dfg = df_seg[['series', 'pixel_count_liver', 'pixel_count_spleen', 'pixel_count_kidney', 'pixel_count_bowel']].groupby('series').max()
    dfg = dfg * 0.9 + 1
    df_seg = df_seg.merge(dfg.reset_index(), on="series", suffixes=("", "_norm"))
    for col in SEG_TARGETS:
        df_seg[col + "_norm"] = np.clip(df_seg[col] / df_seg[col + "_norm"], 0, 1)

    return df_seg


def load_segmentation(path):
    img = nibabel.load(path).get_fdata()
    img = np.transpose(img, [1, 0, 2])
    img = np.rot90(img, 1, (1, 2))
    img = img[::-1, :, :]
    img = np.transpose(img, [1, 0, 2])
    return img[::-1]


def load_series(patient_id, series, img_path=""):
    files = sorted(glob.glob(img_path + f"{patient_id}_{series}_*"))
    imgs = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])
    return imgs


def center_crop_pad(img, size=384):
    h, w = img.shape[-2:]
    if h >= size:
        margin = (h - size) // 2
        img = img[..., margin : margin + size, :]
    else:
        new_img = np.zeros(list(img.shape[:-2]) + [size, img.shape[-1]])
        margin = (size - h) // 2
        new_img[..., margin: margin + h, :] = img
        img = new_img
    if w >= size:
        margin = (w - size) // 2
        img = img[..., margin : margin + size]
    else:
        new_img = np.zeros(list(img.shape[:-2]) + [size, size])
        margin = (size - w) // 2
        new_img[..., margin: margin + w] = img
        img = new_img
    
    return img


def auto_windowing(img):
    pixels = img[::4, ::4, ::4].flatten()
    
    pixels = np.clip(pixels, -300, 450)
    
    pixels = pixels[pixels < pixels.max()]
    pixels = pixels[pixels > pixels.min()]
    
    start = np.percentile(pixels, 1)
    end = np.percentile(pixels, 99)

    img = np.clip(img, start, end)

    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    return img, (start, end)
