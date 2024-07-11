import os
import cv2
import glob
import nibabel
import numpy as np
import pandas as pd

from params import SEG_TARGETS, CROP_TARGETS, DATA_PATH


def prepare_folds(data_path=DATA_PATH, k=4):
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
        "bowel_injury",
        "extravasation_injury",
        "kidney_low",
        "kidney_high",
        "liver_low",
        "liver_high",
        "spleen_low",
        "spleen_high",
    ]

    df = pd.read_csv(data_path + "train.csv")

    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    mskf = mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    splits = mskf.split(df, y=df[cols])

    df["fold"] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i

    df_folds = df[["patient_id", "fold"]]
    df_folds.to_csv(data_path + f"folds_{k}.csv", index=False)
    return df_folds


def prepare_data(data_path=DATA_PATH, with_seg=True):
    """
    Prepare data for 2D classification.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".
        with_seg (bool, optional): Whether to retrieve organ segmengations. Defaults to True.

    Returns:
        pandas DataFrame: Metadata containing image information.
        pandas DataFrame: Metadata containing patient information.
    """
    df_patient = pd.read_csv(os.path.join(data_path, "df_train.csv"))

    if not with_seg:
        df_img = pd.read_csv(os.path.join(data_path, "df_images_train.csv"))
        df_img = df_img.rename(columns={"patient": "patient_id"})

    else:
        df_img = pd.read_csv(os.path.join(data_path, "df_images_train_with_seg.csv"))
        df_img[
            "pred_extravasation"
        ] = 1  # df_img[["pred_liver", "pred_spleen", "pred_bowel", "pred_kidney"]].max(1)

        df_img = df_img.merge(
            df_patient[["patient_id", "kidney", "liver", "spleen", "any_injury"]],
            how="left",
        )
        # "kidney_low", "kidney_high", "liver_low", "liver_high"]] for soft labels ?
        for col in ["kidney", "liver", "spleen"]:
            df_img[f"{col}_injury"] = (df_img[f"pred_{col}"] > 0.9) * df_img[col]

    return df_patient, df_img


def prepare_seg_data(data_path=DATA_PATH, use_3d=False):
    """
    Prepare data for segmentation.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".
        use_3d (bool, optional): Whether segmentation is 3D. Defaults to False.

    Returns:
        pandas DataFrame: Metadata containing image information.
    """
    if use_3d:
        return pd.read_csv(data_path + "df_seg_3d.csv")

    df_seg = pd.read_csv(data_path + "df_seg.csv")

    df_seg["pixel_count_kidney"] = (
        df_seg["pixel_count_left-kidney"] + df_seg["pixel_count_right-kidney"]
    )

    # Add pixel prop tgt
    dfg = (
        df_seg[
            [
                "series",
                "pixel_count_liver",
                "pixel_count_spleen",
                "pixel_count_left-kidney",
                "pixel_count_right-kidney",
                "pixel_count_kidney",
                "pixel_count_bowel",
            ]
        ]
        .groupby("series")
        .max()
    )
    dfg = dfg * 0.9 + 1
    df_seg = df_seg.merge(dfg.reset_index(), on="series", suffixes=("", "_norm"))
    for col in SEG_TARGETS:
        df_seg[col + "_norm"] = np.clip(df_seg[col] / df_seg[col + "_norm"], 0, 1)

    return df_seg


def load_segmentation(path):
    """
    Load the 3D image segmentation from a NIfTI file.

    Args:
        path (str): Path to the NIfTI file.

    Returns:
        numpy.ndarray: 3D segmentation image.
    """
    img = nibabel.load(path).get_fdata()
    img = np.transpose(img, [1, 0, 2])
    img = np.rot90(img, 1, (1, 2))
    img = img[::-1, :, :]
    img = np.transpose(img, [1, 0, 2])
    return img[::-1]


def load_series(patient_id, series, img_path=""):
    """
    Load a series of 2D medical images for a specific patient and series.

    Args:
        patient_id (str): The patient's identifier.
        series (str): The series identifier.
        img_path (str, optional): The path where the image files are located. Defaults to an empty string.

    Returns:
        numpy.ndarray: A NumPy array containing the loaded 2D medical images.
    """
    files = sorted(glob.glob(img_path + f"{patient_id}_{series}_*"))
    imgs = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])
    return imgs


def center_crop_pad(img, size=384):
    """
    Center crop or pad an image to a specified size.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        size (int): The target size for cropping or padding.

    Returns:
        numpy.ndarray: The cropped or padded image.
    """
    h, w = img.shape[-2:]
    if h >= size:
        margin = (h - size) // 2
        img = img[..., margin: margin + size, :]
    else:
        new_img = np.zeros(list(img.shape[:-2]) + [size, img.shape[-1]])
        margin = (size - h) // 2
        new_img[..., margin: margin + h, :] = img
        img = new_img
    if w >= size:
        margin = (w - size) // 2
        img = img[..., margin: margin + size]
    else:
        new_img = np.zeros(list(img.shape[:-2]) + [size, size])
        margin = (size - w) // 2
        new_img[..., margin: margin + w] = img
        img = new_img

    return img


def get_df_series(df_patient, df_img):
    """
    Construct a DataFrame containing series information based on patient and image DataFrames.

    Args:
        df_patient (pandas DataFrame): Metadata containing patient information.
        df_img (pandas DataFrame): Metadata containing image information.

    Returns:
        pandas DataFrame: A DataFrame containing series information.
    """
    df_series = (
        df_img[["patient_id", "series", "frame"]]
        .groupby(["patient_id", "series"])
        .max()
        .reset_index()
    )
    df_series = df_series.merge(
        df_patient[["patient_id"] + CROP_TARGETS], on="patient_id", how="left"
    )

    df_series["target"] = df_series[CROP_TARGETS].values.tolist()
    df_series = df_series.explode("target")

    df_series["organ"] = ["kidney", "liver", "spleen"] * (len(df_series) // 3)
    df_series.drop(CROP_TARGETS, axis=1, inplace=True)

    df_series["img_path"] = (
        "../input/crops/imgs/"
        + df_series["patient_id"].astype(str)
        + "_"
        + df_series["series"].astype(str)
        + "_"
        + df_series["organ"]
        + ".npy"
    )
    df_series["mask_path"] = (
        "../input/crops/masks/"
        + df_series["patient_id"].astype(str)
        + "_"
        + df_series["series"].astype(str)
        + "_"
        + df_series["organ"]
        + ".npy"
    )

    return df_series


def get_start_end(x):
    """
    Calculate the starting and ending positions within a sequence.

    Args:
        x (numpy.ndarray): Input sequence.

    Returns:
        tuple: A tuple containing two integers, (start, end), representing the starting
        and ending positions within the sequence.
    """
    return np.argmax(x), len(x) - np.argmax(x[::-1])
