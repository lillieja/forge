"""
MediaPipe-compatible palm detection post-processing (SSD anchors, decode, NMS, ROI).

Constants match ``palm_detection_cpu.pbtxt`` / GPU graphs plus **two** aspect ratios
``[1.0, 0.5]`` so anchor count is **2016** (matches ``hand_detector.tflite`` output).

References:
  https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
  https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/tensor/tensors_to_detections_calculator.cc
  https://github.com/VimalMollyn/GenMediaPipePalmDectionSSDAnchors (anchor count check)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

NUM_BOXES = 2016
NUM_COORDS = 18
NUM_CLASSES = 1
KEYPOINT_OFFSET = 4
NUM_KEYPOINTS = 7
DETECTOR_SIZE = 192
LANDMARK_SIZE = 224

# TensorsToDetectionsCalculator (palm)
SIGMOID_SCORE = True
SCORE_CLIP = 100.0
REVERSE_OUTPUT_ORDER = True  # XYWH raw box layout
X_SCALE = Y_SCALE = W_SCALE = H_SCALE = 192.0
MIN_SCORE_THRESH = 0.5
NMS_MIN_SUPPRESSION_THRESHOLD = 0.3

# SsdAnchorsCalculator — second ratio yields 2016 anchors (see module docstring).
ANCHOR_OPTIONS = dict(
    num_layers=4,
    strides=(8, 16, 16, 16),
    min_scale=0.1484375,
    max_scale=0.75,
    input_size_width=DETECTOR_SIZE,
    input_size_height=DETECTOR_SIZE,
    anchor_offset_x=0.5,
    anchor_offset_y=0.5,
    aspect_ratios=(1.0, 0.5),
    fixed_anchor_size=True,
    reduce_boxes_in_lowest_layer=False,
    interpolated_scale_aspect_ratio=0.0,
)


@dataclass
class Anchor:
    y_center: float
    x_center: float
    h: float
    w: float


@dataclass
class NormalizedRect:
    """``mediapipe.NormalizedRect``-compatible (rotation clockwise radians)."""

    x_center: float
    y_center: float
    height: float
    width: float
    rotation: float = 0.0


def calculate_scale(min_scale: float, max_scale: float, stride_index: int, num_strides: int) -> float:
    if num_strides == 1:
        return (min_scale + max_scale) * 0.5
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def generate_ssd_anchors(options: dict[str, Any] | None = None) -> list[Anchor]:
    """Port of MediaPipe ``SsdAnchorsCalculator::GenerateAnchors`` (fixed_anchor_size)."""
    opt = {**ANCHOR_OPTIONS, **(options or {})}
    num_layers = opt["num_layers"]
    strides = opt["strides"]
    num_strides = len(strides)
    anchors: list[Anchor] = []
    layer_id = 0
    while layer_id < num_layers:
        aspect_ratios: list[float] = []
        scales: list[float] = []
        last_same = layer_id
        while last_same < num_strides and strides[last_same] == strides[layer_id]:
            scale = calculate_scale(opt["min_scale"], opt["max_scale"], last_same, num_strides)
            if last_same == 0 and opt["reduce_boxes_in_lowest_layer"]:
                aspect_ratios.extend([1.0, 2.0, 0.5])
                scales.extend([0.1, scale, scale])
            else:
                for ar in opt["aspect_ratios"]:
                    aspect_ratios.append(ar)
                    scales.append(scale)
                if opt["interpolated_scale_aspect_ratio"] > 0.0:
                    scale_next = (
                        1.0
                        if last_same == num_strides - 1
                        else calculate_scale(
                            opt["min_scale"], opt["max_scale"], last_same + 1, num_strides
                        )
                    )
                    scales.append(math.sqrt(scale * scale_next))
                    aspect_ratios.append(opt["interpolated_scale_aspect_ratio"])
            last_same += 1

        anchor_heights: list[float] = []
        anchor_widths: list[float] = []
        for i in range(len(aspect_ratios)):
            r = math.sqrt(aspect_ratios[i])
            anchor_heights.append(scales[i] / r)
            anchor_widths.append(scales[i] * r)

        stride = strides[layer_id]
        fh = int(math.ceil(opt["input_size_height"] / stride))
        fw = int(math.ceil(opt["input_size_width"] / stride))
        for y in range(fh):
            for x in range(fw):
                for _aid in range(len(anchor_heights)):
                    xc = (x + opt["anchor_offset_x"]) / fw
                    yc = (y + opt["anchor_offset_y"]) / fh
                    if opt["fixed_anchor_size"]:
                        ww, hh = 1.0, 1.0
                    else:
                        ww = anchor_widths[_aid]
                        hh = anchor_heights[_aid]
                    anchors.append(Anchor(y_center=yc, x_center=xc, h=hh, w=ww))
        layer_id = last_same
    return anchors


_ANCHORS_CACHE: list[Anchor] | None = None


def palm_anchors() -> list[Anchor]:
    global _ANCHORS_CACHE
    if _ANCHORS_CACHE is None:
        _ANCHORS_CACHE = generate_ssd_anchors()
        if len(_ANCHORS_CACHE) != NUM_BOXES:
            raise RuntimeError(
                f"Anchor mismatch: got {len(_ANCHORS_CACHE)}, expected {NUM_BOXES}"
            )
    return _ANCHORS_CACHE


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -SCORE_CLIP, SCORE_CLIP)))


def decode_boxes(raw_boxes: np.ndarray, anchors: list[Anchor]) -> np.ndarray:
    """``TensorsToDetectionsCalculator::DecodeBoxes`` (XYWH + keypoints). Returns [N, 18]."""
    raw_boxes = np.asarray(raw_boxes).reshape(NUM_BOXES, NUM_COORDS)
    boxes = np.zeros((NUM_BOXES, NUM_COORDS), dtype=np.float32)
    for i in range(NUM_BOXES):
        row = raw_boxes[i]
        # XYWH order when reverse_output_order is true
        x_center = float(row[0])
        y_center = float(row[1])
        w = float(row[2])
        h = float(row[3])

        ax = anchors[i].x_center
        ay = anchors[i].y_center
        ah = anchors[i].h
        aw = anchors[i].w

        x_center = x_center / X_SCALE * aw + ax
        y_center = y_center / Y_SCALE * ah + ay
        h = h / H_SCALE * ah
        w = w / W_SCALE * aw

        ymin = y_center - h / 2.0
        xmin = x_center - w / 2.0
        ymax = y_center + h / 2.0
        xmax = x_center + w / 2.0
        boxes[i, 0] = ymin
        boxes[i, 1] = xmin
        boxes[i, 2] = ymax
        boxes[i, 3] = xmax

        for k in range(NUM_KEYPOINTS):
            ko = KEYPOINT_OFFSET + k * 2
            kx = float(row[ko + 0])
            ky = float(row[ko + 1])
            boxes[i, ko + 0] = kx / X_SCALE * aw + ax
            boxes[i, ko + 1] = ky / Y_SCALE * ah + ay
    return boxes


def raw_detections_to_structs(
    boxes: np.ndarray, raw_scores: np.ndarray
) -> list[dict[str, Any]]:
    """Build detection dicts with score and relative bbox (letterboxed [0,1])."""
    scores = raw_scores.reshape(NUM_BOXES, NUM_CLASSES)
    out: list[dict[str, Any]] = []
    for i in range(NUM_BOXES):
        row = scores[i]
        class_id = 0
        score = float(row[0])
        if SIGMOID_SCORE:
            score = float(_sigmoid(np.array([score]))[0])
        if score < MIN_SCORE_THRESH:
            continue
        ymin, xmin, ymax, xmax = boxes[i, 0:4].tolist()
        kps = boxes[i, KEYPOINT_OFFSET : KEYPOINT_OFFSET + NUM_KEYPOINTS * 2].reshape(
            NUM_KEYPOINTS, 2
        )
        out.append(
            {
                "index": i,
                "score": score,
                "class_id": class_id,
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax,
                "keypoints": kps.astype(np.float32),
            }
        )
    return out


def detection_letterbox_removal(
    detections: list[dict[str, Any]], padding: np.ndarray
) -> None:
    """In-place: map coords from letterboxed 192 tensor space to uncropped image [0,1]."""
    left, top, right, bottom = [float(p) for p in padding]
    lr = left + right
    tb = top + bottom
    for d in detections:
        d["xmin"] = (d["xmin"] - left) / (1.0 - lr)
        d["ymin"] = (d["ymin"] - top) / (1.0 - tb)
        d["xmax"] = (d["xmax"] - left) / (1.0 - lr)
        d["ymax"] = (d["ymax"] - top) / (1.0 - tb)
        w = d["xmax"] - d["xmin"]
        h = d["ymax"] - d["ymin"]
        d["width"] = w
        d["height"] = h
        kps = d["keypoints"]
        kps[:, 0] = (kps[:, 0] - left) / (1.0 - lr)
        kps[:, 1] = (kps[:, 1] - top) / (1.0 - tb)


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    xa1, ya1, xa2, ya2 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
    xb1, yb1, xb2, yb2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)


def non_max_suppression(
    detections: list[dict[str, Any]], min_suppression_threshold: float = NMS_MIN_SUPPRESSION_THRESHOLD
) -> list[dict[str, Any]]:
    """Greedy IoU NMS (approximates MediaPipe WEIGHTED NMS for single-class)."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep: list[dict[str, Any]] = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [
            d
            for d in dets
            if _iou(best, d) < min_suppression_threshold
        ]
    return keep


def palm_detection_to_normalized_rect(det: dict[str, Any]) -> NormalizedRect:
    """
    Maps a palm ``detection`` (letterbox-removed, normalized) to ``NormalizedRect``.

    Uses SSD box for scale/center and wrist + middle MCP for rotation (BlazePalm-style).
    Sizes are clipped so ``NormalizedRect`` stays in valid normalized image bounds.
    """
    kps = det["keypoints"]
    wrist = kps[0]
    middle_mcp = kps[2]
    dx = float(middle_mcp[0] - wrist[0])
    dy = float(middle_mcp[1] - wrist[1])
    rotation = math.atan2(dy, dx)
    bw = float(det["xmax"] - det["xmin"])
    bh = float(det["ymax"] - det["ymin"])
    cx = float((det["xmin"] + det["xmax"]) * 0.5)
    cy = float((det["ymin"] + det["ymax"]) * 0.5)
    size = max(bw, bh) * 2.3
    size = float(np.clip(size, 0.08, 0.95))
    return NormalizedRect(
        x_center=float(np.clip(cx, 0.0, 1.0)),
        y_center=float(np.clip(cy, 0.0, 1.0)),
        height=size,
        width=size,
        rotation=-rotation,
    )


def letterbox_to_square(
    rgb: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MediaPipe ``ImageToTensorCalculator`` FIT letterbox → ``[1,size,size,3]`` float32 [0,1].

    Returns:
      tensor: NHWC float32
      padding: ``[left, top, right, bottom]`` normalized to letterboxed canvas (as MP).
    """
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = size - nw
    pad_h = size - nh
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = resized
    t = canvas.astype(np.float32) / 255.0
    out = np.expand_dims(t, axis=0)
    # Padding relative to letterboxed image (same convention as DetectionLetterboxRemoval).
    pad = np.array(
        [
            left / size,
            top / size,
            right / size,
            bottom / size,
        ],
        dtype=np.float32,
    )
    return out, pad


def preprocess_detector_input(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-frame RGB uint8 → detector tensor + letterbox padding."""
    return letterbox_to_square(rgb, DETECTOR_SIZE)


def warp_normalized_roi_to_landmark_input(
    rgb: np.ndarray,
    rect: NormalizedRect,
    out_size: int = LANDMARK_SIZE,
) -> np.ndarray:
    """
    Crop ``rect`` from ``rgb`` (uint8 RGB full image) and letterbox to ``out_size``.

    Mirrors MediaPipe ``ImageToTensorCalculator`` on ROI (FIT, float [0,1], NHWC).
    """
    h, w = rgb.shape[:2]
    cx = rect.x_center * w
    cy = rect.y_center * h
    rw = rect.width * w
    rh = rect.height * h
    angle_deg = math.degrees(rect.rotation)
    rrect = ((cx, cy), (rw, rh), angle_deg)
    box = cv2.boxPoints(rrect).astype(np.float32)
    dst = np.array(
        [[0, out_size - 1], [0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(rgb, m, (out_size, out_size), flags=cv2.INTER_LINEAR)
    t = crop.astype(np.float32) / 255.0
    return np.expand_dims(t, axis=0)


def postprocess_detector_outputs(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    letterbox_padding: np.ndarray,
    anchors: list[Anchor] | None = None,
) -> list[dict[str, Any]]:
    """Decode TRT/TFLite palm outputs → list of filtered detections (full-image normalized)."""
    if anchors is None:
        anchors = palm_anchors()
    rb = np.asarray(raw_boxes).reshape(NUM_BOXES, NUM_COORDS)
    rs = np.asarray(raw_scores).reshape(NUM_BOXES, NUM_CLASSES)
    boxes = decode_boxes(rb, anchors)
    dets = raw_detections_to_structs(boxes, rs)
    detection_letterbox_removal(dets, letterbox_padding)
    return non_max_suppression(dets)


def split_palm_detector_outputs(
    outs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pick raw box tensor [2016,18] and score tensor [2016,1] from TRT outputs (any order)."""
    raw_boxes = None
    raw_scores = None
    for arr in outs:
        a = np.asarray(arr).reshape(-1)
        if a.size == NUM_BOXES * NUM_COORDS:
            raw_boxes = np.asarray(arr).reshape(NUM_BOXES, NUM_COORDS)
        elif a.size == NUM_BOXES * NUM_CLASSES:
            raw_scores = np.asarray(arr).reshape(NUM_BOXES, NUM_CLASSES)
    if raw_boxes is None or raw_scores is None:
        raise ValueError(
            f"Palm outputs: need tensors sized {NUM_BOXES * NUM_COORDS} and {NUM_BOXES * NUM_CLASSES}"
        )
    return raw_boxes, raw_scores


def palm_roi_from_detector_outputs(
    outs: list[np.ndarray],
    letterbox_padding: np.ndarray,
) -> NormalizedRect | None:
    """Decode palm TRT outputs + letterbox padding used for inference → best ``NormalizedRect``."""
    rb, rs = split_palm_detector_outputs(outs)
    dets = postprocess_detector_outputs(rb, rs, letterbox_padding)
    if not dets:
        return None
    return palm_detection_to_normalized_rect(dets[0])
