def add_bucket(row: dict[str, int]) -> dict[str, int]:
    return {"bucket": row["value"] // 2}


def keep_large(row: dict[str, int]) -> bool:
    return row["value"] >= 2


def summarize_group(rows: list[dict[str, int]]) -> dict[str, int]:
    return {"total": sum(row["value"] for row in rows), "count": len(rows)}


def annotate(row: dict[str, int]) -> dict[str, str]:
    return {"label": f"v{row['value']}"}


def empty_update(row: dict) -> dict:
    """Test Map UDF that intentionally replaces a row with an empty mapping."""
    return {}


def join_rows_share_incident_id(left: dict, right: dict) -> bool:
    """Test join predicate: both rows share the same string ``incident_id``."""
    return left.get("incident_id") == right.get("incident_id")


def join_cross_camera_test_bucket(left: dict, right: dict) -> bool:
    """Test join predicate: different cameras sharing the same appearance bucket."""
    return left.get("camera_id") != right.get("camera_id")


def join_test_match_score(left: dict, right: dict) -> float:
    """Test join score UDF: mean of left/right confidence fields."""
    return (float(left.get("confidence", 0)) + float(right.get("confidence", 0))) / 2.0
