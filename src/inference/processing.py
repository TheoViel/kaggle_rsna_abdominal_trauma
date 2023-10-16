import glob
import torch
import pydicom
import dicomsdl
import numpy as np
import torch.nn.functional as F


def restrict_imgs(img_paths, max_len=600, margin=10):
    """
    Restrict the list of image paths based on specified conditions.

    Args:
        img_paths (List[str]): List of image file paths.
        max_len (int, optional): Maximum number of image paths to retain. Defaults to 600.
        margin (int, optional): Margin for selecting the range of image paths. Defaults to 10.

    Returns:
        List[str]: Restricted list of image file paths.
        int: Original number of image paths.
    """
    n_imgs = len(img_paths)

    if n_imgs > 400:
        img_paths = img_paths[n_imgs // 6 - margin:]
    else:
        img_paths = img_paths[max(n_imgs // 8 - margin, 0):]

    img_paths = img_paths[-max_len - margin:]

    return img_paths, n_imgs


def center_crop_pad_gpu(img, size=384):
    """
    Perform a center crop and padding operation on a GPU-based image tensor.

    Args:
        img (torch.Tensor): The input image tensor on GPU.
        size (int): The desired size (both height and width) for the output image.

    Returns:
        torch.Tensor: The center-cropped and padded image tensor with the desired size.
    """
    h, w = img.size()
    if h >= size:
        margin = (h - size) // 2
        img = img[margin: margin + size, :]
    else:
        new_img = torch.zeros([size, w]).cuda()
        margin = (size - h) // 2
        new_img[margin: margin + h, :] = img
        img = new_img
    if w >= size:
        margin = (w - size) // 2
        img = img[:, margin: margin + size]
    else:
        new_img = torch.zeros([size, size]).cuda()
        margin = (size - w) // 2
        new_img[:, margin: margin + w] = img
        img = new_img

    return img


def dicomsdl_to_numpy_image(dicom, index=0):
    """
    Convert pixel data from a DICOM file to a NumPy array.

    Args:
        dicom: The DICOM file object or reader.
        index (int): The frame index for multi-frame DICOMs.

    Returns:
        numpy.ndarray: A NumPy array containing pixel data from the DICOM file.
    """
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
    """
    Load an image from a DICOM file using the dicomsdl library and convert it to a NumPy array.

    Args:
        f (str): The path to the DICOM file.

    Returns:
        numpy.ndarray: A NumPy array containing the image data from the DICOM file.
    """
    return dicomsdl_to_numpy_image(dicomsdl.open(f))


def process(patient, study, on_gpu=False, data_path="", crop_size=None, restrict=False):
    """
    Process DICOM images for a specific patient and study.

    Args:
        patient (str): Patient identifier.
        study (str): Study identifier.
        on_gpu (bool): Flag to specify if processing should be done on the GPU.
        data_path (str): Path to the directory containing DICOM image files.
        crop_size (int, optional): Size for cropping images. Default is None (no cropping).
        restrict (bool, optional): Flag to restrict the number of images. Default is False.

    Returns:
        torch.Tensor: A tensor containing processed DICOM images.
        list of str: Paths to the processed image files.
        int: The number of processed images.
    """
    # Retrieve order
    zs = {}
    for f in sorted(glob.glob(data_path + f"{patient}/{study}/*.dcm")):
        try:
            dicom = pydicom.dcmread(f, stop_before_pixels=True)
            pos_z = dicom[(0x20, 0x32)].value[-1]
            zs[pos_z] = f
        except Exception:
            pass
    sorted_imgs = [zs[k] for k in sorted(zs.keys())]

    # Restrict
    if restrict:
        sorted_imgs, n_imgs = restrict_imgs(sorted_imgs)
    else:
        n_imgs = len(sorted_imgs)

    # Load
    all_imgs, paths = [], []
    for i, f in enumerate(
        sorted_imgs
    ):
        try:
            dicom = pydicom.dcmread(f)

            try:
                pixel_array = load_img_dicomsdl(f)
            except Exception:
                pixel_array = None

            img = standardize_pixel_array(dicom, pixel_array=pixel_array, on_gpu=on_gpu)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            ref_size = 512
            if img.size(1) != ref_size:
                h = int(ref_size / img.size(1) * img.size(0))
                img = F.interpolate(
                    img.unsqueeze(0).unsqueeze(0), size=(h, ref_size), mode="bilinear"
                )[0, 0]

            if crop_size is not None:
                if on_gpu:
                    img = center_crop_pad_gpu(img, size=crop_size)
                else:
                    raise NotImplementedError

            if dicom.PhotometricInterpretation == "MONOCHROME1":
                img = 1 - img

            all_imgs.append(img.unsqueeze(0))
            paths.append(f"{patient}_{study}_{i:04d}.png")

        except Exception:
            pass
    all_imgs = torch.cat(all_imgs, 0)
    return all_imgs, paths, n_imgs


def standardize_pixel_array(
    dcm: pydicom.dataset.FileDataset, pixel_array=None, on_gpu=False
) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    Standardize and window the pixel array from a DICOM image.

    Args:
        dcm (pydicom.dataset.FileDataset): The DICOM dataset.
        pixel_array (numpy.ndarray, optional): The pixel array to standardize.
            If not provided, it's extracted from the DICOM dataset.
        on_gpu (bool, optional): Flag indicating if the processing should be performed on a GPU.

    Returns:
        np.ndarray or torch.Tensor: The standardized pixel array.
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
