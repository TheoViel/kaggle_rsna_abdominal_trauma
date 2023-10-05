import glob
import torch
import pydicom
import dicomsdl
import numpy as np
import torch.nn.functional as F


def restrict_imgs(img_paths, max_len=600, margin=10):
    n_imgs = len(img_paths)
    
    if n_imgs > 400:
        img_paths = img_paths[n_imgs // 6 - margin:]
    else:
        img_paths = img_paths[max(n_imgs // 8 - margin, 0):]
            
    img_paths = img_paths[- max_len - margin :]
    
    return img_paths, n_imgs


def center_crop_pad_gpu(img, size=384):
    h, w = img.size()
    if h >= size:
        margin = (h - size) // 2
        img = img[margin : margin + size, :]
    else:
        new_img = torch.zeros([size, w]).cuda()
        margin = (size - h) // 2
        new_img[margin: margin + h, :] = img
        img = new_img
    if w >= size:
        margin = (w - size) // 2
        img = img[:, margin : margin + size]
    else:
        new_img = torch.zeros([size, size]).cuda()
        margin = (size - w) // 2
        new_img[:, margin: margin + w] = img
        img = new_img
    
    return img


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


def process(patient, study, on_gpu=False, data_path="", crop_size=None, restrict=False):
    # Retrieve order
    zs = {}
    for f in sorted(glob.glob(data_path + f"{patient}/{study}/*.dcm")):
        try:
            dicom = pydicom.dcmread(f, stop_before_pixels=True)
            pos_z = dicom[(0x20, 0x32)].value[-1]
            zs[pos_z] = f
        except:
            pass
    sorted_imgs = [zs[k] for k in sorted(zs.keys())]

    # Restrict
    if restrict:
        sorted_imgs, n_imgs = restrict_imgs(sorted_imgs)
    else:
        n_imgs = len(sorted_imgs)
    
    # Load
    all_imgs, paths = [], []
    for i, f in enumerate(sorted_imgs): # glob.glob(data_path + f"{patient}/{study}/*.dcm")):
        try:
            dicom = pydicom.dcmread(f)

            try:
                pixel_array = load_img_dicomsdl(f)
            except:
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

#             if on_gpu:
#                 img = img.cpu().numpy()  # TODO : REMOVE
#             all_imgs[f"{patient}_{study}_{i:04d}.png"] = img.astype(np.float32)

        except:
            pass
    all_imgs = torch.cat(all_imgs, 0)
    return all_imgs, paths, n_imgs


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