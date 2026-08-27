"""Deprecated compatibility imports for the UCA retrieval example."""

from warnings import warn

warn(
    "mmds.join.text_retrieval is deprecated; import UCA helpers from "
    "mmds.case_studies.text_retrieval instead.",
    DeprecationWarning,
    stacklevel=2,
)

from mmds.case_studies.text_retrieval import (  # noqa: E402,F401
    format_gt_caption_clip,
    group_gt_clips_by_video,
    tag_gallery_video,
)

__all__ = [
    "format_gt_caption_clip",
    "group_gt_clips_by_video",
    "tag_gallery_video",
]
