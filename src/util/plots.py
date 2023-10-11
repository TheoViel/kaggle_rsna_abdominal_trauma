import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_pred,
    y_true,
    cm=None,
    normalize="true",
    display_labels=None,
    cmap="viridis",
    show_label=False,
):
    """
    Computes and plots a confusion matrix.

    Args:
        y_pred (numpy array): Predictions.
        y_true (numpy array): Truths.
        cm (numpy array or None, optional): Precomputed onfusion matrix. Defaults to None.
        normalize (bool or None, optional): Whether to normalize the matrix. Defaults to None.
        display_labels (list of strings or None, optional): Axis labels. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "viridis".
    """
    if cm is None:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Display colormap
    n_classes = cm.shape[0]
    im_ = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    # Display values
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    thresh = (cm.max() + cm.min()) / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0.1:
                color = cmap_max if cm[i, j] < thresh else cmap_min
                text = f"{cm[i, j]:.0f}" if normalize is None else f"{cm[i, j]:.1f}"
                plt.text(j, i, text, ha="center", va="center", color=color)

    # Display legend
    plt.xlim(-0.5, n_classes - 0.5)
    plt.ylim(-0.5, n_classes - 0.5)
    if display_labels is not None:
        plt.xticks(np.arange(n_classes), display_labels)
        plt.yticks(np.arange(n_classes), display_labels)

    if show_label:
        plt.ylabel("True label", fontsize=12)
        plt.xlabel("Predicted label", fontsize=12)

    
def plot_mask(img, mask):
    mask = np.copy(mask)
    for j in range(5):
        mask[j, 0] = j + 1

    plt.imshow(img, cmap='gray')
    mask = np.where(mask, mask, np.nan)
    plt.imshow(mask, cmap='Set3', alpha=0.3)        
    plt.axis(False)


def show_cmap(kidney=False):
    if kidney:
        labels = {1: "liver", 2: "spleen", 3: "kidney", 4: "bowel", 5:""}
        plt.imshow(np.arange(1, 6)[None], cmap='Set3', alpha=0.5)  
    else:
        labels = {1: "liver", 2: "spleen", 3: "left-kidney", 4: "right-kidney", 5: "bowel"}
        plt.imshow(np.arange(1, 6)[None], cmap='Set3', alpha=0.5)  

    for i in range(len(labels)):
        plt.text(
            i, 0, labels[i + 1],
            horizontalalignment='center',
            verticalalignment='center',
        )

    plt.axis(False)
    plt.show()

    
def plot_boxes(img, boxes, bbox_format="yolo", merged_boxes=None):
    """
    Plots an image with bounding boxes.
    Coordinates are expected in format [x_center, y_center, width, height].

    Args:
        img (numpy.ndarray): The input image to be plotted.
        boxes_list (list): A list of lists containing the bounding box coordinates.
    """
    boxes = np.array(boxes)
#     if len(boxes.shape) == 1:
#         boxes = boxes[None]
        
    plt.imshow(img, cmap="gray")
    plt.axis(False)

    for box in boxes:
        if bbox_format == "yolo":
            h, w = img.shape[:2]
            rect = Rectangle(
                ((box[0] - box[2] / 2) * w, (box[1] - box[3] / 2) * h),
                box[2] * w,
                box[3] * h,
                linewidth=2,
                facecolor="none",
                edgecolor="salmon",
            )
        else:
            rect = Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                facecolor="none",
                edgecolor="salmon",
            )

        plt.gca().add_patch(rect)
        
    if merged_boxes is not None:
        for box in merged_boxes:
            if bbox_format == "yolo":
                h, w = img.shape[:2]
                rect = Rectangle(
                    ((box[0] - box[2] / 2) * w, (box[1] - box[3] / 2) * h),
                    box[2] * w,
                    box[3] * h,
                    linewidth=1.5,
                    facecolor="none",
                    edgecolor="skyblue",
                )
            else:
                rect = Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=1.5,
                    facecolor="none",
                    edgecolor="skyblue",
                )

            plt.gca().add_patch(rect)
