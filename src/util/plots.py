import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_contours(img, mask=None, preds=None, w=1):
    """
    Plot contours on the input image.

    Args:
        img (numpy array): The input image as a NumPy array.
        mask (numpy array, optional): The mask to plot the contours for. Defaults to None.
        preds (numpy array, optional): The predictions to plot the contours for. Defaults to None.
        w (int, optional): Width of the contour lines. Defaults to 1.

    Returns:
        None: This function only displays the image with the plotted contours.
    """
    contours, contours_preds = None, None
    img = img.copy()
    if img.max() > 1:
        img = (img / 255).astype(float)

    if mask is not None:
        if mask.max() > 1:
            mask = (mask / 255).astype(float)
        mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if preds is not None:
        if preds.max() > 1:
            preds = (preds / 255).astype(float)
        preds = (preds * 255).astype(np.uint8)
        contours_preds, _ = cv2.findContours(preds, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if contours_preds is not None:
        cv2.polylines(img, contours_preds, True, (0.0, 1.0, 0.0), w)
    if contours is not None:
        cv2.polylines(img, contours, True, (1.0, 0.0, 0.0), w)

    plt.imshow(img)
    plt.axis(False)


def plot_mask(img, mask):
    """
    Plot the mask overlay on the input image.

    Args:
        img (numpy array): The input image as a NumPy array.
        mask (numpy array): The mask to overlay on the image.

    Returns:
        None: This function only displays the image with the mask overlay.
    """
    mask = mask.copy()
    mask = np.where(mask, mask, img)

    plt.imshow(img)
    plt.imshow(mask, cmap='Reds', alpha=.4, interpolation='none')
    plt.axis(False)


def plot_sample(img, mask, figsize=(18, 6), n=3):
    """
    Plot a sample with the original image, the mask, and the contours.

    Args:
        img (numpy array): The original image as a NumPy array.
        mask (numpy array): The mask to plot.
        figsize (tuple, optional): Figure size for the subplots. Defaults to (18, 6).

    Returns:
        None: This function only displays the sample with the image, mask, and contours.
    """
    if len(mask.shape) < len(img.shape):
        mask = mask[:, :, None]

    plt.figure(figsize=figsize)

    plt.subplot(1, n, 1)
    plt.imshow(img)
    plt.axis(False)

    if n == 3:
        plt.subplot(1, n, 2)
        plot_mask(img, mask)

        plt.subplot(1, n, 3)
        plot_contours(img, mask)
    else:
        assert n == 2
        plt.subplot(1, n, 2)
        plot_contours(img, mask)

    plt.show()
