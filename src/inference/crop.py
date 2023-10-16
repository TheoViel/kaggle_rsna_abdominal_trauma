import torch
import numpy as np


def get_crops(seg):
    """
    Extract bounding box coordinates for organs in a segmentation mask.
    This function takes a 3D segmentation mask as input and calculates the coordinates of bounding boxes for
    liver, spleen, and kidney.

    Args:
        seg (torch.Tensor): 3D segmentation mask tensor wwith labels
            (background = 0, liver = 1, spleen = 2, and kidney = 3).

    Returns:
        List of List: A list containing the coordinates for different organs in the following format:
        [[x0_liver, x1_liver, y0_liver, y1_liver, z0_liver, z1_liver],
         [x0_spleen, x1_spleen, y0_spleen, y1_spleen, z0_spleen, z1_spleen],
         [x0_kidney, x1_kidney, y0_kidney, y1_kidney, z0_kidney, z1_kidney]]
    """
    # Imp
    def get_start_end(x):
        x = x.int()
        return torch.argmax(x).item(), len(x) - torch.argmax(x.flip(0)).item()

    liver = (seg == 1).int()
    spleen = (seg == 2).int()
    kidney = (seg == 3).int()

    x0_liver, x1_liver = get_start_end(liver.sum((1, 2)) > 400)
    y0_liver, y1_liver = get_start_end(liver.sum((0, 2)) > 400)
    z0_liver, z1_liver = get_start_end(liver.sum((0, 1)) > 400)

    x0_spleen, x1_spleen = get_start_end(spleen.sum((1, 2)) > 100)
    y0_spleen, y1_spleen = get_start_end(spleen.sum((0, 2)) > 100)
    z0_spleen, z1_spleen = get_start_end(spleen.sum((0, 1)) > 100)

    x0_kidney, x1_kidney = get_start_end(kidney.sum((1, 2)) > 100)
    y0_kidney, y1_kidney = get_start_end(kidney.sum((0, 2)) > 100)
    z0_kidney, z1_kidney = get_start_end(kidney.sum((0, 1)) > 100)

    x0s = [x0_liver, x0_spleen, x0_kidney]
    x1s = [x1_liver, x1_spleen, x1_kidney]
    y0s = [y0_liver, y0_spleen, y0_kidney]
    y1s = [y1_liver, y1_spleen, y1_kidney]
    z0s = [z0_liver, z0_spleen, z0_kidney]
    z1s = [z1_liver, z1_spleen, z1_kidney]
    return np.array([x0s, x1s, y0s, y1s, z0s, z1s]).T.tolist()
