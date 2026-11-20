"""Detectors: image in, list of detections out.

`MotionDetector` needs no model and no GPU. `TorchvisionDetector` wraps a
pretrained network. The pipeline only knows the `Detector` interface, so a
different model is one new class.
"""

from __future__ import annotations

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


def make_detector(name: str, score_threshold: float = 0.4) -> Detector:
    if name == "motion":
        return MotionDetector()
    if name in TorchvisionDetector._MODELS:
        return TorchvisionDetector(model=name, score_threshold=score_threshold)
    raise ValueError(f"unknown detector {name!r}; choose motion, ssdlite or fasterrcnn")
