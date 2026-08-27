from __future__ import annotations

import threading
from typing import Any

from ...model import DatasetExpr, DetectSpec, MMDSValidationError, Row
from ...utilities.media import resolve_video_source
from ...utilities.video import Video, VideoView, open_video

# ---------------------------------------------------------------------------
# Model cache — one YOLOE instance per model name, shared across rows.
# A lock serialises set_classes+predict so concurrent row processing is safe.
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}
_model_lock = threading.Lock()


def _select_device() -> str:
    """Return ``"cuda"`` only if CUDA is genuinely usable, else ``"cpu"``.

    ``torch.cuda.is_available()`` may return ``True`` even when the installed
    PyTorch build lacks kernels for the current GPU (e.g. compute capability
    6.1 on a build that requires ≥ 7.0).  We try a small convolution—the same
    op type that YOLOE uses—and fall back to CPU if it fails.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        # A minimal Conv2d forward pass exercises the same CUDA kernels that
        # YOLOE will need.  If this fails, CUDA isn't really usable.
        _probe = torch.nn.Conv2d(1, 1, 1).cuda()
        _probe(torch.zeros(1, 1, 1, 1, device="cuda"))
        return "cuda"
    except Exception:
        return "cpu"


_device: str | None = None


def _get_device() -> str:
    """Return the device string, computing and caching it once."""
    global _device
    if _device is None:
        _device = _select_device()
    return _device


def _get_model(model_name: str) -> Any:
    with _model_lock:
        if model_name not in _model_cache:
            from ultralytics import YOLOE  # deferred: heavy import

            model = YOLOE(model_name)
            model.to(_get_device())
            _model_cache[model_name] = model
        return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Core apply function
# ---------------------------------------------------------------------------


def _apply_detect(node: DatasetExpr, row: Row) -> Row:
    """Run YOLOE detection on every frame of the video field and merge results."""
    spec = node.spec
    assert isinstance(spec, DetectSpec)

    raw = row.get(spec.video_field)
    if raw is None:
        raise MMDSValidationError(
            f"Detect: field {spec.video_field!r} is missing from the row."
        )

    source = resolve_video_source(raw)
    video = open_video(source)
    if isinstance(video, list):
        raise MMDSValidationError(
            f"Detect: video field {spec.video_field!r} resolved to a directory; "
            "it must point to a single video file."
        )

    # Wrap in a VideoView if the raw field dict specifies a time range.
    iterable: Video | VideoView = video
    if isinstance(raw, dict):
        start = raw.get("start")
        end = raw.get("end")
        if start is not None and end is not None:
            iterable = VideoView(video, float(start), float(end))

    detections = _detect_in_video(
        iterable,
        list(spec.classes),
        spec.model,
        frame_stride=spec.frame_stride,
        conf=spec.conf,
        imgsz=spec.imgsz,
    )

    result = dict(row)
    result[spec.output_field] = detections
    return result


def _detect_in_video(
    video: Video | VideoView,
    classes: list[str],
    model_name: str,
    *,
    frame_stride: int = 1,
    conf: float | None = None,
    imgsz: int | None = None,
) -> list[dict[str, Any]]:
    """Run detection on sampled frames and group bboxes by detected class.

    *video* may be a :class:`Video` (processes all frames) or a
    :class:`VideoView` (processes only the view's frame range). Detection
    records always use absolute frame indices from the underlying source
    video, even when iterating a :class:`VideoView`.

    *frame_stride* controls how densely frames are sampled for inference:
    ``1`` (the default) runs every frame through the model; ``N > 1`` only
    runs inference on every ``N``-th frame (frame ``0``, ``N``, ``2N``, ...),
    skipping the model call entirely on the rest. This trades temporal
    resolution for lower inference cost and is most appropriate when the
    detected objects (and their attributes) do not change meaningfully
    between consecutive frames.

    Returns a list of::

        {"type": <class_name>, "bboxes": [{"frame_idx": int, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]}
    """
    if frame_stride < 1:
        raise MMDSValidationError("_detect_in_video frame_stride must be >= 1.")

    # Only forward predict kwargs the caller actually set, so unset values keep
    # the model's own defaults rather than being pinned to None.
    predict_kwargs: dict[str, Any] = {}
    if conf is not None:
        predict_kwargs["conf"] = conf
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    model = _get_model(model_name)

    with _model_lock:
        text_pe = model.get_text_pe(classes)
        model.set_classes(classes, text_pe)

    base_frame_idx = video.start_frame if isinstance(video, VideoView) else 0

    by_class: dict[str, list[dict[str, Any]]] = {}
    for relative_frame_idx, frame in enumerate(video):
        if relative_frame_idx % frame_stride != 0:
            continue
        frame_idx = base_frame_idx + relative_frame_idx
        with _model_lock:
            results = model.predict(
                frame, verbose=False, device=_get_device(), **predict_kwargs
            )
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                class_name: str = result.names[int(boxes.cls[i])]
                by_class.setdefault(class_name, []).append(
                    {
                        "frame_idx": frame_idx,
                        "bbox": boxes.xyxy[i].tolist(),
                        "confidence": float(boxes.conf[i]),
                    }
                )

    return [{"type": cls, "bboxes": bboxes} for cls, bboxes in by_class.items()]
