import numpy as np


def pascal_to_yolo(boxes, h=None, w=None):
    """
    x0, y0, x1, y1 -> xc, yc, w, h
    Normalized in [0, 1].
    Args:
        boxes (np array): Boxes in the pascal format.
        h (int, optional): Image height. Defaults to None.
        w (int, optional): Image width. Defaults to None.

    Returns:
        np array: Boxes in the yolo format.
    """
    if not len(boxes):
        return boxes

    if h is not None and w is not None:
        boxes = boxes.astype(float)
        boxes[:, 0] = np.clip(boxes[:, 0] / w, 0, 1)
        boxes[:, 1] = np.clip(boxes[:, 1] / h, 0, 1)
        boxes[:, 2] = np.clip(boxes[:, 2] / w, 0, 1)
        boxes[:, 3] = np.clip(boxes[:, 3] / h, 0, 1)

    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 2] - boxes[:, 0]
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 3] - boxes[:, 1]

    return boxes


def pascal_to_albu(boxes, h, w):
    """
    x0, y0, x1, y1 -> x0, y0, x1, y1 normalized in [0, 1].
    Args:
        boxes (np array): Boxes in the pascal format.
        h (int): Image height.
        w (int,): Image width.

    Returns:
        np array: Boxes in the albu format.
    """
    if not len(boxes):
        return boxes

    boxes = boxes.astype(float)
    boxes[:, 0] = boxes[:, 0] / w
    boxes[:, 1] = boxes[:, 1] / h
    boxes[:, 2] = boxes[:, 2] / w
    boxes[:, 3] = boxes[:, 3] / h

    boxes = np.clip(boxes, 0, 1)

    return boxes


def albu_to_pascal(boxes, h, w):
    """
    x0, y0, x1, y1 normalized in [0, 1] -> x0, y0, x1, y1.
    Args:
        boxes (np array): Boxes in the albu format.
        h (int): Image height.
        w (int): Image width.

    Returns:
        np array: Boxes in the pascal format.
    """
    if not len(boxes):
        return boxes

    boxes = np.clip(boxes, 0, 1).astype(float)

    boxes[:, 0] = boxes[:, 0] * w
    boxes[:, 1] = boxes[:, 1] * h
    boxes[:, 2] = boxes[:, 2] * w
    boxes[:, 3] = boxes[:, 3] * h

    return np.round(boxes).astype(int)


def pascal_to_coco(boxes):
    """
    x0, y0, x1, y1 -> x0, y0, w, h
    Args:
        boxes (np array): Boxes in the pascal format.

    Returns:
        np array: Boxes in the yolo format.
    """
    if not len(boxes):
        return boxes
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    return boxes


def coco_to_pascal(boxes):
    """
    x0, y0, w, h -> x0, y0, x1, y1
    Args:
        boxes (np array): Boxes in the coco format.

    Returns:
        np array: Boxes in the pascal format.
    """
    if not len(boxes):
        return boxes
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    return boxes


def yolo_to_pascal(boxes, h=None, w=None):
    """
    xc, yc, w, h -> x0, y0, x1, y1
    Normalized in [0, 1]

    Args:
        boxes (np array): Boxes in the yolo format
        h (int, optional): Image height. Defaults to None.
        w (int, optional): Image width. Defaults to None.

    Returns:
        np array: Boxes in the pascal format.
    """
    if not len(boxes):
        return boxes

    boxes[:, 0], boxes[:, 2] = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 0] + boxes[:, 2] / 2
    boxes[:, 1], boxes[:, 3] = boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 1] + boxes[:, 3] / 2

    if h is not None and w is not None:
        boxes[:, 0] = boxes[:, 0] * w
        boxes[:, 1] = boxes[:, 1] * h
        boxes[:, 2] = boxes[:, 2] * w
        boxes[:, 3] = boxes[:, 3] * h

        boxes = np.round(boxes).astype(int)

    return boxes


def expand_boxes(boxes, r=1, min_size=0, max_size=None):
    """
    Expands boxes. Handled in the coco format which is perhaps to the easiest.

    Args:
        boxes (Boxes): Boxes.
        r (int, optional): Exansion ratio. Defaults to 1.

    Returns:
        Boxes: Expanded boxes.
    """
    if isinstance(boxes, list):
        return boxes

    shape = boxes.shape
    boxes = boxes["yolo"]

    boxes[:, 2] = np.clip(boxes[:, 2] * r, min_size / shape[1], max_size / shape[1])
    boxes[:, 3] = np.clip(boxes[:, 3] * r, min_size / shape[0], max_size / shape[0])


    for b in boxes:  # shift boxes out of bounds
        if b[0] - b[2] / 2 < 0:
#             b[2] += b[0] - b[2] / 2
            b[0] = b[2] / 2

        if b[0] + b[2] / 2 > 1:
#             b[2] -= b[0] + b[2] / 2 - 1
            b[0] = 1 - (b[2] / 2)

        if b[1] - b[3] / 2 < 0:
#             b[3] += b[1] - b[3] / 2
            b[1] = b[3] / 2

        if b[1] + b[3] / 2 > 1:
#             b[3] -= b[1] + b[3] / 2 - 1
            b[1] = 1 - (b[3] / 2)

    return Boxes(boxes, shape, bbox_format="yolo")


class Boxes:
    """
    Class to handle different format of bounding boxes easily.
    """
    def __init__(self, data, shape, bbox_format="yolo"):
        h, w = shape[:2]
        self.shape = shape[:2]
        self.h = h
        self.w = w

        if bbox_format == "yolo":
            self.boxes_yolo = data
            self.boxes_pascal = yolo_to_pascal(self.boxes_yolo.copy(), h, w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), h, w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())
        elif bbox_format == "pascal_voc":
            self.boxes_pascal = data
            self.boxes_yolo = pascal_to_yolo(self.boxes_pascal.copy(), h, w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), h, w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())
        elif bbox_format == "albu":
            self.boxes_albu = data
            self.boxes_pascal = albu_to_pascal(self.boxes_albu.copy(), h, w)
            self.boxes_yolo = pascal_to_yolo(self.boxes_pascal.copy(), h, w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())
        elif bbox_format == "coco":
            self.boxes_coco = data
            self.boxes_pascal = coco_to_pascal(self.boxes_coco.copy())
            self.boxes_yolo = pascal_to_yolo(self.boxes_pascal.copy(), h, w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), h, w)
        else:
            raise NotImplementedError

    def __getitem__(self, bbox_format):
        if bbox_format == "yolo":
            return self.boxes_yolo
        elif bbox_format == "pascal_voc":
            return self.boxes_pascal
        elif bbox_format == "albu":
            return self.boxes_albu
        elif bbox_format == "coco":
            return self.boxes_coco
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.boxes_yolo)

    def update_shape(self, shape):
        self.__init__(self.boxes_yolo, shape, bbox_format="yolo")

    def filter_(self, to_keep):
        self.__init__(self.boxes_yolo[to_keep], self.shape, bbox_format="yolo")

    def get_ratios(self):
        return self.boxes_yolo[:, 3].astype(float) / self.boxes_yolo[:, 2].astype(float)

    def is_side(self, tol=5):
        is_bottom = np.abs(self.boxes_pascal[:, 3] - self.h) < tol
        is_top = np.abs(self.boxes_pascal[:, 2]) < tol
        is_left = np.abs(self.boxes_pascal[:, 1] - self.w) < tol
        is_right = np.abs(self.boxes_pascal[:, 0]) < tol
        return is_bottom | is_top | is_left | is_right
