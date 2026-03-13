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

    def __init__(self) -> None: ...
    def __call__(self, row: dict[str, Any]) -> dict[str, Any]: ...

detect: Detect
