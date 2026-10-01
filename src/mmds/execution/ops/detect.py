from __future__ import annotations

import threading
from typing import Any

from ...model import DatasetExpr, DetectSpec, MMDSValidationError, Row
from ...utilities.video import Video, VideoView, open_video

# ---------------------------------------------------------------------------
# Model cache — one YOLOE instance per model name, shared across rows.
# A lock serialises set_classes+predict so concurrent row processing is safe.
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}
_model_lock = threading.Lock()

# A new box joins an existing track of the same class when it overlaps the
# track's last box and arrives within this many source frames. Otherwise it
# starts another instance. The gap is in source-frame units, so a large
# frame_stride can split one animal into two tracks.
_TRACK_IOU_THRESHOLD = 0.3
_MAX_TRACK_FRAME_GAP = 30


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
# Video source resolution
# ---------------------------------------------------------------------------


def _resolve_video_source(value: Any) -> str:
    """Extract a path/URL string from a row field value.

    Accepted forms:
    - a plain string (file path or URL)
    - a dict with a ``"source"``, ``"path"``, or ``"uri"`` key
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("source", "path", "uri"):
            if key in value and isinstance(value[key], str):
                return value[key]
        raise MMDSValidationError(
            "Detect: video dict must contain a 'source', 'path', or 'uri' string key."
        )
    raise MMDSValidationError(
        f"Detect: video field must be a string path/URL or a dict, got {type(value).__name__!r}."
    )


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

    source = _resolve_video_source(raw)
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
        stop_after_n=spec.stop_after_n,
    )

    result = dict(row)
    result[spec.output_field] = detections
    fps = float(getattr(iterable, "fps", 0.0) or 0.0)
    if fps > 0:
        result["_mmds_video_fps"] = fps
    return result


def _bbox_iou(left: list[float], right: list[float]) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


class _Track:
    """One animal: the last box that was associated to it."""

    def __init__(self, class_name: str, frame_idx: int, bbox: list[float]) -> None:
        self.class_name = class_name
        self.last_frame_idx = frame_idx
        self.last_bbox = bbox


def _match_track(
    tracks: list[_Track],
    *,
    class_name: str,
    frame_idx: int,
    bbox: list[float],
    claimed: set[int],
) -> _Track | None:
    """Return the same-class track this box continues, if one overlaps it."""
    best: _Track | None = None
    best_iou = _TRACK_IOU_THRESHOLD
    for track in tracks:
        if track.class_name != class_name or id(track) in claimed:
            continue
        if frame_idx - track.last_frame_idx > _MAX_TRACK_FRAME_GAP:
            continue
        overlap = _bbox_iou(track.last_bbox, bbox)
        if overlap >= best_iou:
            best = track
            best_iou = overlap
    return best


def _detect_in_video(
    video: Video | VideoView,
    classes: list[str],
    model_name: str,
    *,
    frame_stride: int = 1,
    conf: float | None = None,
    imgsz: int | None = None,
    stop_after_n: int | None = None,
) -> list[dict[str, Any]]:
    """Run detection on sampled frames and group bboxes by detected class.

    When ``stop_after_n`` is set, scanning ends once that many distinct tracks
    exist. A track is one animal: overlapping boxes of the same class across
    nearby frames stay on the same track, and a non-overlapping box starts a
    new one.

    *video* may be a :class:`Video` (processes all frames) or a
    :class:`VideoView` (processes only the view's frame range). Detection
    records always use absolute frame indices from the underlying source
    video, even when iterating a :class:`VideoView`.

    Returns a list of::

        {"type": <class_name>, "bboxes": [{"frame_idx": int, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]}
    """
    if frame_stride < 1:
        raise MMDSValidationError("_detect_in_video frame_stride must be >= 1.")
    if stop_after_n is not None and stop_after_n < 1:
        raise MMDSValidationError(
            "_detect_in_video stop_after_n must be an integer >= 1 or None."
        )

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
    tracks: list[_Track] = []
    for relative_frame_idx, frame in enumerate(video):
        if relative_frame_idx % frame_stride != 0:
            continue
        frame_idx = base_frame_idx + relative_frame_idx
        with _model_lock:
            results = model.predict(
                frame, verbose=False, device=_get_device(), **predict_kwargs
            )
        hits: list[tuple[str, list[float], float]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                class_name: str = result.names[int(boxes.cls[i])]
                hits.append(
                    (
                        class_name,
                        boxes.xyxy[i].tolist(),
                        float(boxes.conf[i]),
                    )
                )
        ordered = sorted(hits, key=lambda hit: hit[2], reverse=True)
        prior_tracks = list(tracks)
        claimed: set[int] = set()
        unmatched: list[tuple[str, list[float], float]] = []
        for class_name, bbox, confidence in ordered:
            track = _match_track(
                prior_tracks,
                class_name=class_name,
                frame_idx=frame_idx,
                bbox=bbox,
                claimed=claimed,
            )
            if track is None:
                unmatched.append((class_name, bbox, confidence))
                continue
            track.last_frame_idx = frame_idx
            track.last_bbox = bbox
            claimed.add(id(track))
            by_class.setdefault(class_name, []).append(
                {
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )
        opened: list[_Track] = []
        for class_name, bbox, confidence in unmatched:
            track = _match_track(
                opened,
                class_name=class_name,
                frame_idx=frame_idx,
                bbox=bbox,
                claimed=set(),
            )
            if track is None:
                track = _Track(class_name, frame_idx, bbox)
                opened.append(track)
                tracks.append(track)
            else:
                track.last_frame_idx = frame_idx
                track.last_bbox = bbox
            by_class.setdefault(class_name, []).append(
                {
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )
        if stop_after_n is not None and len(tracks) >= stop_after_n:
            break

    return [{"type": cls, "bboxes": bboxes} for cls, bboxes in by_class.items()]
