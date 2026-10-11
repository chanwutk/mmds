from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....model import PromptSpec, RecordPath
from ..errors import MMDSRewriteError
from ._prompt import record_paths


DEFAULT_DETECT_MODEL = "yoloe-11s-seg.pt"
DETECTIONS_FIELD = "detections"


class DetectClassesParams(BaseModel):
    """Shared parameters for directives that answer a Map from YOLOE tracks."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    video_field: str = Field(
        min_length=1,
        description="Input field containing the video whose objects answer the Map.",
    )
    classes: list[str] = Field(
        min_length=1,
        description="Open-vocabulary YOLOE class names whose tracks answer the Map.",
    )
    model: str = Field(
        default=DEFAULT_DETECT_MODEL,
        min_length=1,
        description="YOLOE weights file for the Detect stage.",
    )

    @model_validator(mode="after")
    def validate_detect_fields(self) -> DetectClassesParams:
        if not self.video_field.strip():
            raise ValueError("video_field cannot be blank.")
        if not self.model.strip():
            raise ValueError("model cannot be blank.")
        if any(not isinstance(name, str) or not name.strip() for name in self.classes):
            raise ValueError("classes must contain non-empty strings.")
        normalized = tuple(name.strip() for name in self.classes)
        if len(set(normalized)) != len(normalized):
            raise ValueError("classes must be unique.")
        object.__setattr__(self, "video_field", self.video_field.strip())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "classes", list(normalized))
        return self


def single_schema_field(spec: PromptSpec, type_name: str) -> str | None:
    """Return the only schema field when its type is ``type_name``.

    Accepts the shorthand ``{"field": "integer"}`` and the expanded
    ``{"field": {"type": "integer"}}``.
    """

    schema = spec.output_schema or {}
    if len(schema) != 1:
        return None
    field, value = next(iter(schema.items()))
    if isinstance(value, Mapping):
        value = value.get("type")
    if value != type_name:
        return None
    return field


def require_video_reference(spec: PromptSpec, video_field: str) -> None:
    if RecordPath((video_field,)) not in record_paths(spec.parts):
        raise MMDSRewriteError(
            "Matched Map does not directly reference video field "
            f"{video_field!r}."
        )
