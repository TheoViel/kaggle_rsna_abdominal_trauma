# Adapted from https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py

import warnings
import numpy as np
from util.boxes import Boxes


def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes in albu or pascal_voc format.

    Args:
        box1 (list or tuple): The coordinates of the first bounding box in [x_min, y_min, x_max, y_max] format.
        box2 (list or tuple): The coordinates of the second bounding box in [x_min, y_min, x_max, y_max] format.

    Returns:
        float: The IoU between the two bounding boxes, a value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def get_combined_box(box1, box2):
    """
    Get the bounding box that contains two input bounding boxes in albu or pascal_voc format.

    Args:
        box1 (list or tuple): The coordinates of the first bounding box in [x_min, y_min, x_max, y_max] format.
        box2 (list or tuple): The coordinates of the second bounding box in [x_min, y_min, x_max, y_max] format.

    Returns:
        list: The coordinates of the bounding box that contains both input bounding boxes.
    """
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    combined_box = [x_min, y_min, x_max, y_max]
    return np.array(combined_box)


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > 1:
                warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x1 = 1
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > 1:
                warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x2 = 1
            if y1 < 0:
                warnings.warn('Y1 < 0 in box. Set it to 0.')
                y1 = 0
            if y1 > 1:
                warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y1 = 1
            if y2 < 0:
                warnings.warn('Y2 < 0 in box. Set it to 0.')
                y2 = 0
            if y2 > 1:
                warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
#                 warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, n_hits, x1, y1, x2, y2)
    """

    box = np.zeros(10, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:8] += (b[1] * b[4:8])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = np.median([b[3] for b in boxes])  # Median frame
    box[8] = np.min([b[3] for b in boxes])
    box[9] = np.max([b[3] for b in boxes])
    box[4:8] /= conf
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """
    def bb_iou_array(boxes, new_box):
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

#     ious = bb_iou_array_frame(boxes[:, 3:], new_box[3:])
    ious = bb_iou_array(boxes[:, 4:8], new_box[4:8])

    ious[boxes[:, 0] != new_box[0]] = -1
    ious[
        (np.abs(boxes[:, 9] - new_box[3]) > 5) &
        (np.abs(boxes[:, 8] - new_box[3]) > 5)
    ] = -1
#     ious[
#         (np.abs(boxes[:, 9] - new_box[3]) > 5) &
#         (np.abs(boxes[:, 8] - new_box[3]) > 5)
#     ] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]
    
    
    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1


    return best_idx, best_iou


def weighted_boxes_fusion(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type='avg',
    allows_overflow=False
):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), {}

    boxes = filtered_boxes[0]

    new_boxes = []
    mapping = {}
    weighted_boxes = np.empty((0, 10))

    # Clusterize boxes
    for j in range(0, len(boxes)):
        index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)
        
        if index != -1:
            new_boxes[index].append(boxes[j])
            weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            mapping[index].append(boxes[j, 3])
        else:
            new_boxes.append([boxes[j].copy()])
            
            new_box = np.zeros(10)
            new_box[:8] = boxes[j].copy()
            new_box[8] = boxes[j, 3]
            new_box[9] = boxes[j, 3]
            weighted_boxes = np.vstack((weighted_boxes, new_box))
            mapping[len(weighted_boxes) - 1] = [boxes[j, 3]]

    overall_boxes = weighted_boxes.copy()
    boxes = overall_boxes[:, 4:8]
    scores = overall_boxes[:, 1]
    hits = overall_boxes[:, 2]

    mapping = np.array([np.sort(mapping[k]) for k in range(len(overall_boxes))], dtype=object)
    return boxes, scores, hits, mapping


def fusion(
    boxes,
    confidences,
    iou_threshold=0.2,
    conf_threshold=0.1,
    skip_box_thr=0.,
    hits_threshold=0,
    merge=False,
    max_det=-1
):
    """
    Merges detections at different depths using wbf.

    Args:
        iou_threshold (float, optional): IoU threshold for wbf. Defaults to 0.2.

    Returns:
        list of Boxes: Ground truths.
        list of Boxes: Predictions.
        list of Boxes: Prediction confidences.
    """
    fused_boxes = []

    preds = [pred["albu"].copy() for pred in boxes]
    labels = [[0] * len(p) for p in  preds]

    pred_wbf, confidences_wbf, hits_wbf, mapping = weighted_boxes_fusion(
        preds,
        confidences,
        labels,
        iou_thr=iou_threshold,
        skip_box_thr=skip_box_thr,
    )
    
    if not len(pred_wbf):
        return [], [], [], []

    # Filter low conf
    pred_wbf = pred_wbf[confidences_wbf >= conf_threshold]
    hits_wbf = hits_wbf[confidences_wbf >= conf_threshold]
    mapping = mapping[confidences_wbf >= conf_threshold]
    confidences_wbf = confidences_wbf[confidences_wbf >= conf_threshold]
    
    if not len(pred_wbf):
        return [], [], [], []
    
    # Filter low hits
    pred_wbf = pred_wbf[hits_wbf >= hits_threshold]
    confidences_wbf = confidences_wbf[hits_wbf >= hits_threshold]
    mapping = mapping[hits_wbf >= hits_threshold]
    hits_wbf = hits_wbf[hits_wbf >= hits_threshold]
    
    if not len(pred_wbf):
        return [], [], [], []

    zs = np.array([[np.min(c), np.max(c)] for c in mapping]).astype(int)
    
#     print()
    if not merge:
        pred_wbf = Boxes(
            pred_wbf,
            shape=(boxes[0].h, boxes[0].w),
            bbox_format="albu"
        )
        return pred_wbf, confidences_wbf, hits_wbf, zs
    
    # Re-merge
    ids = np.argsort(zs[:, 0])
    
    pred_merged, confidences_merged, hits_merged, zs_merged = [], [], [], []
    for i in range(len(ids)):
        idx = ids[i]
        if i == 0:
            pred_merged.append(pred_wbf[idx])
            confidences_merged.append(confidences_wbf[idx])
            hits_merged.append(hits_wbf[idx])
            zs_merged.append(zs[idx])
        
        elif (
            (np.abs(zs_merged[-1][1] - zs[idx, 0]) < 3) or (np.abs(zs_merged[-1][0] - zs[idx, 1]) < 3)
            and (iou(pred_wbf[idx], pred_merged[-1]) > 0.2)
        ):  # End prev - start next
#             print('Merging', zs_merged[-1],  zs[idx])

            pred_merged[-1] = get_combined_box(pred_wbf[idx], pred_merged[-1])
            confidences_merged[-1] = max(confidences_wbf[idx], confidences_merged[-1])
            hits_merged[-1] += hits_wbf[idx]
            zs_merged[-1] = [zs_merged[-1][0], zs[idx, 1]]
            
        else:
            pred_merged.append(pred_wbf[idx])
            confidences_merged.append(confidences_wbf[idx])
            hits_merged.append(hits_wbf[idx])
            zs_merged.append(zs[idx])

    zs_merged = np.array(zs_merged)
    confidences_merged = np.array(confidences_merged)
    hits_merged = np.array(hits_merged)
    pred_merged = np.array(pred_merged)
    
    if max_det > 0:
        ids = np.argsort(confidences_merged)[::-1][:max_det]
        zs_merged = zs_merged[ids]
        confidences_merged = confidences_merged[ids]
        hits_merged = hits_merged[ids]
        pred_merged = pred_merged[ids]
        
    pred_merged = Boxes(
        pred_merged,
        shape=(boxes[0].h, boxes[0].w),
        bbox_format="albu"
    )
    
    return pred_merged, confidences_merged, hits_merged, zs_merged
