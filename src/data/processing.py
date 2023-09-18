import cv2
import glob
import pydicom
import numpy as np


def process(patient, study, size=None, save_folder="", data_path=""):
    all_imgs = {}
    imgs = {}
    for f in sorted(glob.glob(data_path + f"{patient}/{study}/*.dcm")):
        try:
            dicom = pydicom.dcmread(f)

            pos_z = dicom[(0x20, 0x32)].value[-1]

            img = standardize_pixel_array(dicom)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            if dicom.PhotometricInterpretation == "MONOCHROME1":
                img = 1 - img

            imgs[pos_z] = img
        except:
            pass

    for i, k in enumerate(sorted(imgs.keys())):
        img = imgs[k]

        if size is not None:
            img = cv2.resize(img, (size, size))

        all_imgs[save_folder + f"{patient}_{study}_{i}.png"] =  (img * 255).astype(np.uint8)
    return all_imgs


def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
#         pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)

    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2    
    
    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    return pixel_array