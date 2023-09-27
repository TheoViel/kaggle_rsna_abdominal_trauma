import os
import gc
import sys
import cv2
import torch
import importlib
import numpy as np
import torch.nn as nn
import albumentations as albu

from tqdm import tqdm
from yolox.utils import postprocess
from albumentations import pytorch as AT
from torch.utils.data import Dataset, DataLoader

from util.boxes import Boxes


class InferenceDataset(Dataset):
    """
    Dataset for inference in a detection task.

    Attributes:
        df (DataFrame): The DataFrame containing the dataset information.
        paths (numpy.ndarray): The paths to the images in the dataset.
        transforms: Augmentations to apply to the images.
        gts (list): Ground truth boxes for each image.
        classes (list): Ground truth classes for each image.

    Methods:
        __init__(self, df, transforms=None, pad=False, pad_advanced=False): Constructor
        __len__(self): Returns the length of the dataset.
        __getitem__(self, idx): Returns the item at the specified index.
    """
    def __init__(self, df, transforms=None):
        """
        Constructor

        Args:
            df (DataFrame): The DataFrame containing the dataset information.
            transforms (albu transforms, optional): Augmentations. Defaults to None.
        """
        self.df = df
        self.paths = df["path"].values
        self.transforms = transforms

        self.gts, self.classes = [], []
        for i in range(len(df)):
            try:
                with open(df["gt_path"][i], "r") as f:
                    bboxes = np.array([line[:-1].split() for line in f.readlines()]).astype(float)
                    labels, bboxes = bboxes[:, 0], bboxes[:, 1:]
                    self.gts.append(bboxes)
                    self.classes.append(labels)
            except Exception:
                self.gts.append([])
                self.classes.append([])

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        Args:
            idx (int): Index.

        Returns:
            tuple: A tuple containing the image, ground truth, and image shape.
        """
        image = cv2.imread(self.paths[idx])
        
        shape = image.shape
#         shape = (384, 384)

        if self.transforms is not None:
            try:
                image = self.transforms(image=image, bboxes=[], class_labels=[])["image"]
            except ValueError:
                image = self.transforms(image=image)["image"]

        return image, self.gts[idx], shape


def get_transfos(size):
    """
    Returns a composition of image transformations for preprocessing.

    Args:
        size (tuple): The desired size of the transformed image (height, width).

    Returns:
        albumentations.Compose: The composition of image transformations.
    """
    normalizer = albu.Compose(
        [
            albu.Normalize(mean=0, std=1),
            AT.transforms.ToTensorV2(),
        ],
        p=1,
    )

    return albu.Compose(
        [
#             albu.Resize(size[0], size[1]),
            albu.PadIfNeeded(size[0], size[1]),
            albu.CenterCrop(size[0], size[1]),
            normalizer,
        ],
        bbox_params=albu.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


class DetectionMeter:
    """
    Detection meter for evaluating object detection performance.

    Methods:
        __init__(pred_format, truth_format): Constructor
        update(y_batch, preds, shape): Update ground truths and predictions
        reset(): Resets all values

    Attributes:
        truth_format (str): Format of ground truth bounding box coordinates
        pred_format (str): Format of predicted bounding box coordinates
        preds (list): List of predicted bounding boxes (Boxes instances)
        labels (list): List of labels corresponding to predicted bounding boxes
        confidences (list): List of confidence scores for predicted bounding boxes
        truths (list): List of ground truth bounding boxes (Boxes instances)
        metrics (dict): Dictionary storing evaluation metrics (tp, fp, fn, precision, recall, f1_score)
    """

    def __init__(self, pred_format="coco", truth_format="yolo"):
        """
        Constructor

        Args:
            pred_format (str, optional): Format of predicted bounding box coordinates. Defaults to "coco".
            truth_format (str, optional): Format of ground truth bounding box coordinates. Defaults to "yolo".
        """
        self.truth_format = truth_format
        self.pred_format = pred_format
        self.reset()

    def update(self, y_batch, preds, shape):
        """
        Update ground truths and predictions.

        Args:
            y_batch (list of np arrays): Truths.
            preds (list of torch tensors): Predictions.
            shape (list or tuple): Image shape.

        Raises:
            NotImplementedError: Mode not implemented.
        """
        n, c, h, w = shape  # TODO : verif h & w

        self.truths += [
            Boxes(box, (h, w), bbox_format=self.truth_format) for box in y_batch
        ]

        for pred in preds:
            pred = pred.cpu().numpy()

            if pred.shape[1] >= 5:
                label = pred[:, 5].astype(int)
                self.labels.append(label)

            pred, confidences = pred[:, :4], pred[:, 4]

            self.preds.append(Boxes(pred, (h, w), bbox_format=self.pred_format))
            self.confidences.append(confidences)

    def reset(self):
        """
        Resets everything.
        """
        self.preds = []
        self.labels = []
        self.confidences = []
        self.truths = []

        self.metrics = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        }


def collate_fn_val_yolo(batch):
    """
    Validation batch collating function for yolo-v5.

    Args:
        batch (tuple): Input batch.

    Returns:
        torch tensor: Images.
        list: Boxes.
        list: Image shapes.
    """
    img, boxes, shapes = zip(*batch)
    return torch.stack(list(img), 0), boxes, shapes


def predict(model, dataset, config, disable_tqdm=True):
    """
    Performs prediction on a dataset using a model.

    Args:
        model (nn.Module): The model to use for prediction.
        dataset (Dataset): The dataset to perform prediction on.
        config: Configuration object or dictionary.
        disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to True.

    Returns:
        DetectionMeter: Evaluation meter.
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=config.val_bs,
        shuffle=False,
        collate_fn=collate_fn_val_yolo,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    meter = DetectionMeter(pred_format=config.pred_format, truth_format=config.bbox_format)
    meter.reset()

    with torch.no_grad():
        for batch in tqdm(loader, disable=disable_tqdm):
            x = batch[0].to(config.device)
            pred_boxes = model(x)
            meter.update(batch[1], pred_boxes, x.size())

    gc.collect()
    torch.cuda.empty_cache()
    return meter


def retrieve_yolox_model(exp_file, ckpt_file, size=(1024, 1024), verbose=1):
    """
    Retrieves and configures a YOLOX model for inference.

    Args:
        exp_file (str): The path to the experiment file.
        ckpt_file (str): The path to the checkpoint file containing the model weights.
        size (tuple, optional): The input size of the model. Defaults to (1024, 1024).
        verbose (int, optional): Verbosity level. If 1, it prints the loading message. Defaults to 1.

    Returns:
        nn.Module: The configured YOLOX model for inference.
    """
    sys.path.append(os.path.dirname(exp_file))
    current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])

    exp = current_exp.Exp()

    exp.test_conf = 0.0
    exp.test_size = size
    exp.nmsthre = 0.75

    model_roi_ = exp.get_model()

    if verbose:
        print(" -> Loading weights from", ckpt_file)

    ckpt = torch.load(ckpt_file, map_location="cpu")
    model_roi_.load_state_dict(ckpt["model"], strict=True)

    model_roi_.max_det = 100
    model_roi_.nmsthre = 0.75
    model_roi_.test_conf = 0.1
    model_roi_.test_size = exp.test_size
    model_roi_.num_classes = 1
    model_roi_.stride = 64
    model_roi_.amp = False  # FP16

    return model_roi_.eval().cuda()


class YoloXWrapper(nn.Module):
    """
    Wrapper for YoloX models.

    Methods:
        __init__(model, config): Constructor
        forward(x): Forward function

    Attributes:
        model (torch model): Yolo-v5 model.
        config (Config): Config.
        conf_thresh (float): Confidence threshold.
        iou_thresh (float): IoU threshold.
        max_per_img (int): Maximum number of detections per image.
    """
    def __init__(self, model, config):
        """
        Constructor

        Args:
            model (torch model): Yolo model.
            config (Config): Config.
        """
        super().__init__()
        self.model = model
        self.config = config

        self.conf_thresh = config.conf_thresh
        self.iou_thresh = config.iou_thresh
        self.max_per_img = config.max_per_img

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [BS x C x H x W]): Input images.

        Returns:
            torch tensor: Predictions.
        """
        pred_boxes = self.model(x * 255)

        boxes = []
        for i, b in enumerate(pred_boxes):
            conf = min(
                self.conf_thresh,
                (b[:, 4] * b[:, 5:].amax(-1)).max()  # most confident box
            )
            b = postprocess(
                b.unsqueeze(0),
                1,
                conf,
                self.iou_thresh,
                class_agnostic=False,
            )[0][:self.max_per_img]
            boxes.append(b)

        return boxes
