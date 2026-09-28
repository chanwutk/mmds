"""MobileNetV3-small vehicle-color classifier (inference + model builder).

This module provides a learned replacement for the whole-crop mean-RGB color
heuristic in :mod:`udfs.detection_ops`. Averaging a whole bounding box (which
includes windows, shadows, and road) washes saturated colors toward gray; a
small CNN classifier on the crop is far more accurate.

Design:

- The model is a ``torchvision`` MobileNetV3-small with a fresh classification
  head sized to the training vocabulary (the 8-color *Vehicle Color Recognition*
  dataset by Chen et al.).
- Trained weights are an **external artifact** (not committed); train them with
  ``scripts/train_vehicle_color_model.py`` and point at them via the
  ``MMDS_VEHICLE_COLOR_WEIGHTS`` env var or the default ``models/`` path.
- Inference **degrades gracefully**: :func:`predict_color_from_crop` returns
  ``None`` (never raises) whenever the weights file is missing, the ML stack is
  unavailable, the crop is unusable, or inference fails — so the caller falls
  back to the RGB heuristic. When weights are absent it does not even import
  ``torch``, keeping the no-model path cheap.
- The loaded model is cached per weights path behind a lock, mirroring the YOLOE
  model cache in ``mmds.execution.ops.detect`` so it is safe under the executor's
  thread pool.

``torch``/``torchvision``/``cv2`` are imported lazily inside functions so that
importing this module (and ``udfs.detection_ops``) never pulls the heavy ML
stack.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# Chen "Vehicle Color Recognition" dataset classes, in canonical (alphabetical)
# order. The training script MUST assign label indices in this exact order so
# that a checkpoint's output logits line up with these names at inference time.
CHEN_COLOR_CLASSES: tuple[str, ...] = (
    "black",
    "blue",
    "cyan",
    "gray",
    "green",
    "red",
    "white",
    "yellow",
)

# Map each Chen class onto the project's color vocabulary
# (``VEHICLE_COLOR_VOCAB`` in ``udfs.detection_ops``). "cyan" has no vocabulary
# entry, so it maps to the nearest available color ("blue"); every other Chen
# class is already in the vocabulary.
_CHEN_TO_VOCAB: dict[str, str] = {
    "black": "black",
    "blue": "blue",
    "cyan": "blue",
    "gray": "gray",
    "green": "green",
    "red": "red",
    "white": "white",
    "yellow": "yellow",
}

# Default (uncommitted) location of the trained checkpoint, overridable via env.
_WEIGHTS_ENV = "MMDS_VEHICLE_COLOR_WEIGHTS"
DEFAULT_WEIGHTS_PATH = os.path.join("models", "vehicle_color_mobilenet.pt")

# Preprocessing constants (standard ImageNet normalization at 224x224).
_INPUT_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-path model cache + a lock that serialises load and inference so the
# classifier is safe to call from the executor's thread pool.
_model_lock = threading.Lock()
_classifier_cache: dict[str, Any] = {}
_load_failed: set[str] = set()


def vocab_color_for_chen_label(label: str) -> str:
    """Map a Chen dataset color label onto the project color vocabulary.

    Unknown labels fall back to ``"gray"`` (the vocabulary's neutral default).
    """
    return _CHEN_TO_VOCAB.get(label, "gray")


def weights_path() -> str:
    """Return the configured checkpoint path (env override or default)."""
    return os.environ.get(_WEIGHTS_ENV, DEFAULT_WEIGHTS_PATH)


def build_color_model(num_classes: int = len(CHEN_COLOR_CLASSES)) -> Any:
    """Construct a MobileNetV3-small with a fresh ``num_classes`` head.

    Shared by the training script and the inference loader so both agree on the
    architecture. Weights are **not** loaded here.
    """
    import torch.nn as nn
    from torchvision import models

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _preprocess_crop(crop: Any) -> Any:
    """Convert a BGR ``HxWx3`` uint8 crop to a normalized ``(1,3,224,224)`` tensor.

    Uses only ``numpy`` + ``torch`` (no OpenCV) so the color hot path stays light.
    Returns ``None`` when the crop is empty/unusable.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    if getattr(crop, "ndim", 0) != 3 or crop.shape[-1] != 3:
        return None

    # BGR -> RGB via a channel-reversed contiguous copy, then to a CHW batch.
    rgb = np.ascontiguousarray(crop[:, :, ::-1]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(_INPUT_SIZE, _INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _load_classifier(path: str) -> Any:
    """Load and cache the trained model for *path*; return ``None`` on failure.

    Callers must hold ``_model_lock``.
    """
    if path in _classifier_cache:
        return _classifier_cache[path]
    if path in _load_failed:
        return None
    try:
        import torch

        model = build_color_model()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _classifier_cache[path] = model
        return model
    except Exception:
        # Corrupt checkpoint, arch mismatch, missing ML stack, etc. Remember the
        # failure so we do not retry the load on every crop.
        _load_failed.add(path)
        return None


def predict_colors_from_crops(
    crops: list[Any],
    *,
    path: str | None = None,
    batch_size: int = 32,
) -> list[str | None]:
    """Classify BGR vehicle crops in bounded batches.

    Each result is ``None`` when its crop is unusable or model inference is
    unavailable, signalling the caller to use its HSV fallback. Model loading
    and inference remain serialized behind ``_model_lock``.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not crops:
        return []

    resolved = path or weights_path()
    # Cheap no-model path: if there is no checkpoint, skip importing torch.
    if resolved not in _classifier_cache and not os.path.exists(resolved):
        return [None] * len(crops)

    results: list[str | None] = [None] * len(crops)
    with _model_lock:
        model = _load_classifier(resolved)
        if model is None:
            return results

        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            try:
                import torch

                valid = [
                    (offset, tensor)
                    for offset, crop in enumerate(batch)
                    if (tensor := _preprocess_crop(crop)) is not None
                ]
                if not valid:
                    continue
                tensors = torch.cat([tensor for _, tensor in valid], dim=0)
                with torch.no_grad():
                    logits = model(tensors)
                indices = logits.argmax(dim=1).tolist()
                for (offset, _), index in zip(valid, indices):
                    index = int(index)
                    if 0 <= index < len(CHEN_COLOR_CLASSES):
                        results[start + offset] = vocab_color_for_chen_label(
                            CHEN_COLOR_CLASSES[index]
                        )
            except Exception:
                # Leave this batch as ``None`` so callers can use HSV.
                continue
    return results


def predict_color_from_crop(crop: Any, *, path: str | None = None) -> str | None:
    """Classify one crop, preserving the original single-item API."""
    return predict_colors_from_crops([crop], path=path, batch_size=1)[0]


def _reset_cache_for_tests() -> None:
    """Clear the module-level model/failure caches (test helper)."""
    with _model_lock:
        _classifier_cache.clear()
        _load_failed.clear()
