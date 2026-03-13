from __future__ import annotations

from typing import Any


class Detect:
    """YOLOE-based object detector UDF.

    Input row fields:
        image (str | np.ndarray | PIL.Image): image to run detection on.
        classes (list[str], optional): open-vocabulary class names.  When
            omitted the model uses its default COCO classes.

    Output row fields:
        detections (list[dict]): list of detected objects, each with keys
            ``bbox`` ([x1, y1, x2, y2] in pixels), ``confidence`` (float),
            ``class_id`` (int), and ``class_name`` (str).
    """

    def __init__(self) -> None:
        from ultralytics import YOLOE

        self._model = YOLOE("yoloe-11x.pt")

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        image = row["image"]
        classes: list[str] | None = row.get("classes")

        if classes is not None:
            text_pe = self._model.get_text_pe(classes)
            self._model.set_classes(classes, text_pe)

        results = self._model.predict(image, verbose=False)

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                detections.append(
                    {
                        "bbox": boxes.xyxy[i].tolist(),
                        "confidence": float(boxes.conf[i]),
                        "class_id": int(boxes.cls[i]),
                        "class_name": result.names[int(boxes.cls[i])],
                    }
                )

        return {"detections": detections}


detect = Detect()
detect.__name__ = "detect"  # type: ignore[attr-defined]
detect.__module__ = "udfs.detect"  # type: ignore[attr-defined]
detect.__qualname__ = "detect"  # type: ignore[attr-defined]
