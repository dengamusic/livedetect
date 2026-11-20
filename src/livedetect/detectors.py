"""Detectors: image in, list of detections out.

`MotionDetector` needs no model and no GPU. `TorchvisionDetector` and
`OnnxYoloDetector` wrap trained networks. `TiledDetector` wraps any of those
to find targets that are too small for the network at full-frame scale. The
pipeline only knows the `Detector` interface, so a different model is one new
class.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from livedetect.types import Detection


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class MotionDetector:
    """Finds small things that move against a background that mostly does not.

    Keeps a running average of the scene and reports connected regions that
    differ from it. The first `warmup_frames` only learn the background.
    Assumes a fixed camera.
    """

    def __init__(
        self,
        threshold: float = 20.0,
        learning_rate: float = 0.02,
        warmup_frames: int = 30,
        min_area: int = 3,
        max_area: int = 4000,
        max_detections: int = 30,
    ) -> None:
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.warmup_frames = warmup_frames
        self.min_area, self.max_area = min_area, max_area
        self.max_detections = max_detections
        self._background: np.ndarray | None = None
        self._seen = 0
        self._kernel = np.ones((3, 3), np.uint8)

    def detect(self, image: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)  # average out pixel noise
        if self._background is None or self._background.shape != gray.shape:
            self._background, self._seen = gray.copy(), 1
            return []
        self._seen += 1
        difference = cv2.absdiff(gray, self._background)
        # During warm-up this is a plain mean, so an object that happens to be
        # in the first frame is averaged away instead of leaving a ghost.
        rate = max(self.learning_rate, 1.0 / self._seen)
        cv2.accumulateWeighted(gray, self._background, rate)
        if self._seen <= self.warmup_frames:
            return []

        mask = (difference > self.threshold).astype(np.uint8)
        mask = cv2.dilate(mask, self._kernel)  # join the fragments of one object
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections = []
        for i in range(1, count):  # component 0 is the background
            x, y, w, h, area = stats[i]
            if not self.min_area <= area <= self.max_area:
                continue
            peak = float(difference[y : y + h, x : x + w].max())
            score = min(1.0, peak / (4 * self.threshold))
            cx, cy = centroids[i]
            detections.append(Detection(float(cx), float(cy), float(w), float(h), score, "moving"))
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[: self.max_detections]


class TorchvisionDetector:
    """A pretrained torchvision detection network.

    The stock weights are trained on COCO, which has no drone class; treat this
    as the slot where task-specific weights go.
    """

    _MODELS = {
        "ssdlite": ("ssdlite320_mobilenet_v3_large", "SSDLite320_MobileNet_V3_Large_Weights"),
        "fasterrcnn": (
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
        ),
    }

    def __init__(
        self,
        model: str = "ssdlite",
        score_threshold: float = 0.4,
        device: str | None = None,
        pretrained: bool = True,
    ) -> None:
        import torch
        from torchvision.models import detection

        builder_name, weights_name = self._MODELS[model]
        weights = getattr(detection, weights_name).DEFAULT
        self._labels = weights.meta["categories"]
        builder = getattr(detection, builder_name)
        if pretrained:
            network = builder(weights=weights)
        else:  # random weights: no download, used by the tests
            network = builder(weights=None, weights_backbone=None)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._network = network.eval().to(self.device)
        self.score_threshold = score_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        torch = self._torch
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            output = self._network([tensor])[0]
        detections = []
        for box, score, label in zip(
            output["boxes"].tolist(),
            output["scores"].tolist(),
            output["labels"].tolist(),
            strict=True,
        ):
            if score < self.score_threshold:
                continue
            x0, y0, x1, y1 = box
            detections.append(
                Detection(
                    (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, score, self._labels[label]
                )
            )
        return detections


class OnnxYoloDetector:
    """A YOLOv5-format network exported to ONNX, run with onnxruntime.

    ONNX is the hand-off format between training (PyTorch) and deployment: the
    file holds the network and its weights, and the runtime needs neither
    PyTorch nor the training code.
    """

    def __init__(
        self,
        path: str | Path,
        score_threshold: float = 0.4,
        overlap_threshold: float = 0.45,
        threads: int = 0,  # 0 lets the runtime use every core
    ) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        network_input = self._session.get_inputs()[0]
        self._input_name = network_input.name
        self._size = int(network_input.shape[2])  # square input, e.g. 416
        # The exporter stores the class names as text: "{0: 'drone'}".
        names = self._session.get_modelmeta().custom_metadata_map.get("names", "{}")
        self._labels: dict[int, str] = ast.literal_eval(names)
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        canvas, scale, left, top = letterbox(image, self._size)
        rgb = canvas[:, :, ::-1].transpose(2, 0, 1)  # BGR, HWC -> RGB, CHW
        batch = np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0
        (output,) = self._session.run(None, {self._input_name: batch})
        return decode_yolo(
            output[0], scale, left, top, self.score_threshold, self.overlap_threshold, self._labels
        )


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Fit `image` into a size x size square without distorting it.

    Returns the padded square, the scale applied, and the left and top padding,
    which is what is needed to map boxes back onto the original image.
    """
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)  # grey, as in YOLO training
    canvas[top : top + new_height, left : left + new_width] = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, left, top


def decode_yolo(
    rows: np.ndarray,
    scale: float,
    left: int,
    top: int,
    score_threshold: float,
    overlap_threshold: float,
    labels: dict[int, str],
) -> list[Detection]:
    """Turn raw YOLOv5 output into detections in original-image pixels.

    Each row is one candidate box: centre x, centre y, width, height,
    objectness, then one score per class, in letterboxed coordinates. The
    network emits thousands of them, many on the same object, so the last step
    keeps only the best box of each overlapping group.
    """
    class_ids = rows[:, 5:].argmax(axis=1)
    scores = rows[:, 4] * rows[np.arange(len(rows)), 5 + class_ids]
    keep = scores >= score_threshold
    rows, scores, class_ids = rows[keep], scores[keep], class_ids[keep]
    if len(rows) == 0:
        return []
    cx, cy = (rows[:, 0] - left) / scale, (rows[:, 1] - top) / scale
    w, h = rows[:, 2] / scale, rows[:, 3] / scale
    corners = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    best = cv2.dnn.NMSBoxes(corners.tolist(), scores.tolist(), score_threshold, overlap_threshold)
    return [
        Detection(
            float(cx[i]),
            float(cy[i]),
            float(w[i]),
            float(h[i]),
            float(scores[i]),
            labels.get(int(class_ids[i]), "object"),
        )  # fmt: skip
        for i in np.asarray(best).flatten()
    ]


class TiledDetector:
    """Runs another detector on overlapping tiles of the frame, to find small targets.

    A network looks at its input at a fixed size, 416 pixels for the drone
    model. A 960-pixel frame is shrunk to less than half for it, so a drone 12
    pixels wide arrives 5 pixels wide. A tile is a part of the frame, shown to
    the network at closer to its real resolution.

    The whole frame is still run once, for targets too large to fit in a tile,
    so a grid of `cols` x `rows` costs `cols * rows + 1` network runs per frame.
    `grid` may be changed while the detector is in use; (1, 1) turns tiling off.
    """

    def __init__(self, inner: Detector, cols: int = 1, rows: int = 1, overlap: float = 0.2) -> None:
        self.inner = inner
        self.grid = (cols, rows)
        self.overlap = overlap  # fraction of a tile shared with its neighbour

    def detect(self, image: np.ndarray) -> list[Detection]:
        cols, rows = self.grid  # read once: another thread may change it
        detections = list(self.inner.detect(image))
        if (cols, rows) == (1, 1):
            return detections
        height, width = image.shape[:2]
        for x0, y0, x1, y1 in tile_boxes(width, height, cols, rows, self.overlap):
            for d in self.inner.detect(image[y0:y1, x0:x1]):
                d = replace(d, cx=d.cx + x0, cy=d.cy + y0)  # tile -> frame coordinates
                if not _cut_by_tile(d, (x0, y0, x1, y1), width, height):
                    detections.append(d)
        return merge_detections(detections)


def tile_boxes(
    width: int, height: int, cols: int, rows: int, overlap: float = 0.2
) -> list[tuple[int, int, int, int]]:
    """Corners (x0, y0, x1, y1) of a grid of equal tiles that covers the frame.

    Neighbouring tiles share `overlap` of their size, so a target on the edge
    of one tile is whole in the next.
    """

    def spans(length: int, count: int) -> list[tuple[int, int]]:
        size = length / (count - (count - 1) * overlap)
        starts = [round(i * size * (1 - overlap)) for i in range(count)]
        return [(start, min(length, round(start + size))) for start in starts]

    return [(x0, y0, x1, y1) for y0, y1 in spans(height, rows) for x0, x1 in spans(width, cols)]


def _cut_by_tile(
    d: Detection, tile: tuple[int, int, int, int], width: int, height: int, margin: float = 2.0
) -> bool:
    """True if the box touches a tile edge that is not also an edge of the frame.

    Such a box is probably part of a target that continues outside the tile.
    The neighbouring tile or the whole-frame run sees it complete.
    """
    x0, y0, x1, y1 = tile
    return (
        (x0 > 0 and d.cx - d.w / 2 <= x0 + margin)
        or (y0 > 0 and d.cy - d.h / 2 <= y0 + margin)
        or (x1 < width and d.cx + d.w / 2 >= x1 - margin)
        or (y1 < height and d.cy + d.h / 2 >= y1 - margin)
    )


def merge_detections(detections: list[Detection]) -> list[Detection]:
    """Keep the best-scoring box of each target that was found more than once.

    Two boxes count as the same target when the centre of either lies inside
    the other. Overlap ratios, the usual test, are unreliable for boxes a few
    pixels wide: the same drone seen at two scales gives boxes of different
    sizes that share little area.
    """
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda d: d.score, reverse=True):
        if not any(_centre_inside(d, k) or _centre_inside(k, d) for k in kept):
            kept.append(d)
    return kept


def _centre_inside(a: Detection, b: Detection) -> bool:
    return abs(a.cx - b.cx) <= b.w / 2 and abs(a.cy - b.cy) <= b.h / 2


def parse_grid(text: str) -> tuple[int, int]:
    """'3x2' -> (3, 2): columns, then rows."""
    try:
        cols, rows = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"tiles must look like 3x2 (columns x rows), not {text!r}") from None
    if not (1 <= cols <= 8 and 1 <= rows <= 8):
        raise ValueError(f"tiles must be between 1x1 and 8x8, not {text!r}")
    return cols, rows


def make_detector(
    name: str, score_threshold: float = 0.4, tiles: tuple[int, int] = (1, 1)
) -> Detector:
    if name == "motion":
        if tiles != (1, 1):
            # It compares each frame with a remembered background of the same size.
            raise ValueError("the motion detector cannot be tiled")
        return MotionDetector()
    if name == "drone":
        from livedetect.models import fetch

        network: Detector = OnnxYoloDetector(fetch("drone"), score_threshold=score_threshold)
    elif name in TorchvisionDetector._MODELS:
        network = TorchvisionDetector(model=name, score_threshold=score_threshold)
    else:
        raise ValueError(f"unknown detector {name!r}; choose motion, drone, ssdlite or fasterrcnn")
    # Always wrapped, so tiling can be switched on while the service runs.
    return TiledDetector(network, *tiles)
