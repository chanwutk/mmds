"""Appearance embeddings for cross-camera vehicle re-identification.

Turns a vehicle crop into a fixed-length appearance vector so tracks on
different cameras can be matched by embedding **cosine similarity** instead of
brittle exact categorical keys (class/color/subtype), which disagree across
viewpoints.

Two backends, chosen automatically with graceful degradation:

- **Primary — frozen ImageNet CNN** (``torchvision`` ResNet50 with the classifier
  head removed) → an L2-normalized penultimate feature vector. Real learned
  appearance features (shape + texture + color), no training. The weights are
  downloaded once by ``torchvision`` and cached; if ``torch``/weights are
  unavailable (offline / CI), we fall back to:
- **Fallback — coarse RGB color histogram** (pure ``numpy``, fully offline,
  deterministic). Lower discriminative power but keeps the pipeline runnable.

Within a single run the backend is consistent (the model is cached), so every
track's embedding has the same dimensionality and cosine similarity is
well-defined. ``torch``/``torchvision`` are imported lazily so importing this
module stays cheap.
"""

from __future__ import annotations

import threading
from typing import Any

_INPUT_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Fallback descriptor: a per-channel-binned RGB histogram (bins**3 dims).
_HIST_BINS = 4

# Cached backbone + a lock; ``_backbone_failed`` records that the CNN path is
# unavailable so we do not retry the (possibly network) load on every crop.
_model_lock = threading.Lock()
_backbone: Any = None
_backbone_failed = False


def _load_backbone() -> Any:
    """Load and cache the frozen ResNet50 feature extractor, or ``None``.

    Caller must hold ``_model_lock``.
    """
    global _backbone, _backbone_failed
    if _backbone is not None:
        return _backbone
    if _backbone_failed:
        return None
    try:
        import torch
        from torchvision import models

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()  # expose the 2048-d penultimate features
        model.eval()
        _backbone = model
        return model
    except Exception:
        # No torch/torchvision, or weights could not be downloaded (offline).
        _backbone_failed = True
        return None


def _preprocess(crop: Any) -> Any:
    """BGR ``HxWx3`` uint8 crop -> normalized ``(1,3,224,224)`` tensor (or None)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    if getattr(crop, "ndim", 0) != 3 or crop.shape[-1] < 3:
        return None
    rgb = np.ascontiguousarray(crop[:, :, ::-1]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(
        tensor, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def _cnn_embed(crop: Any) -> list[float] | None:
    """CNN appearance embedding, or ``None`` if the CNN path is unavailable."""
    with _model_lock:
        model = _load_backbone()
        if model is None:
            return None
        try:
            import torch

            tensor = _preprocess(crop)
            if tensor is None:
                return None
            with torch.no_grad():
                features = model(tensor)
            return _l2_normalize([float(v) for v in features.reshape(-1).tolist()])
        except Exception:
            return None


def _histogram_embed(crop: Any) -> list[float] | None:
    """Offline fallback: L2-normalized coarse RGB color histogram."""
    import numpy as np

    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    if getattr(crop, "ndim", 0) != 3 or crop.shape[-1] < 3:
        return None
    pixels = np.ascontiguousarray(crop[:, :, ::-1]).reshape(-1, 3).astype(np.float32)
    # Bin each channel into _HIST_BINS and accumulate a joint histogram.
    idx = np.minimum((pixels / (256.0 / _HIST_BINS)).astype(np.int64), _HIST_BINS - 1)
    flat = (idx[:, 0] * _HIST_BINS + idx[:, 1]) * _HIST_BINS + idx[:, 2]
    hist = np.bincount(flat, minlength=_HIST_BINS**3).astype(np.float32)
    return _l2_normalize(hist.tolist())


def embed_crop(crop: Any) -> list[float] | None:
    """Return an appearance embedding for a BGR crop, or ``None`` if unusable.

    Prefers the CNN backbone; falls back to the color histogram when the CNN is
    unavailable. Never raises.
    """
    vector = _cnn_embed(crop)
    if vector is not None:
        return vector
    return _histogram_embed(crop)


def cosine_similarity(left: Any, right: Any) -> float:
    """Cosine similarity of two embedding vectors, clamped to ``[0, 1]``.

    Returns ``0.0`` when either vector is missing, empty, or a different length
    (e.g. a CNN vector compared against a histogram vector).
    """
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return 0.0
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    similarity = dot / (left_norm * right_norm)
    return max(0.0, min(1.0, similarity))


def _reset_cache_for_tests() -> None:
    """Clear the cached backbone / failure flag (test helper)."""
    global _backbone, _backbone_failed
    with _model_lock:
        _backbone = None
        _backbone_failed = False
