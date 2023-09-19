import glob
import torch
import pydicom
import dicomsdl
import numpy as np


def dicomsdl_to_numpy_image(dicom, index=0):
    info = dicom.getPixelDataInfo()
    dtype = info["dtype"]
    if info["SamplesPerPixel"] != 1:
        raise RuntimeError("SamplesPerPixel != 1")
    else:
        shape = [info["Rows"], info["Cols"]]
    outarr = np.empty(shape, dtype=dtype)
    dicom.copyFrameData(index, outarr)
    return outarr


def load_img_dicomsdl(f):
    return dicomsdl_to_numpy_image(dicomsdl.open(f))


def process(patient, study, on_gpu=False, data_path=""):
    all_imgs = {}
    imgs = {}
    for f in sorted(glob.glob(data_path + f"{patient}/{study}/*.dcm")):
        try:
            dicom = pydicom.dcmread(f)

            try:
                pixel_array = load_img_dicomsdl(f)
            except:
                pixel_array = None

            pos_z = dicom[(0x20, 0x32)].value[-1]

            img = standardize_pixel_array(dicom, pixel_array=pixel_array, on_gpu=on_gpu)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            if dicom.PhotometricInterpretation == "MONOCHROME1":
                img = 1 - img

            if on_gpu:
                img = img.cpu().numpy()  # TODO : REMOVE
            imgs[pos_z] = img

        except:
            pass

    for i, k in enumerate(sorted(imgs.keys())):
        all_imgs[f"{patient}_{study}_{i}.png"] = imgs[k].astype(np.float32)

    return all_imgs


def standardize_pixel_array(
    dcm: pydicom.dataset.FileDataset, pixel_array=None, on_gpu=False
) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    if pixel_array is None:
        pixel_array = dcm.pixel_array

    # Correct DICOM pixel_array if PixelRepresentation == 1.
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

    # Windowing
    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2

    pixel_array = pixel_array.astype(np.float32)

    if on_gpu:
        pixel_array = torch.from_numpy(pixel_array).cuda()

    pixel_array = (pixel_array * slope) + intercept

    if on_gpu:
        pixel_array = torch.clamp(pixel_array, low, high)
    else:
        pixel_array = np.clip(pixel_array, low, high)

    return pixel_array
