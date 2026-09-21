def add_bucket(row: dict[str, int]) -> dict[str, int]:
    return {"bucket": row["value"] // 2}


def keep_large(row: dict[str, int]) -> bool:
    return row["value"] >= 2


def summarize_group(rows: list[dict[str, int]]) -> dict[str, int]:
    return {"total": sum(row["value"] for row in rows), "count": len(rows)}


def annotate(row: dict[str, int]) -> dict[str, str]:
    return {"label": f"v{row['value']}"}


def empty_update(row: dict) -> dict:
    return {}


def join_rows_share_incident_id(left: dict, right: dict) -> bool:
    return left.get("incident_id") == right.get("incident_id")


def join_cross_camera_test_bucket(left: dict, right: dict) -> bool:
    return left.get("camera_id") != right.get("camera_id")


def join_test_match_score(left: dict, right: dict) -> float:
    return (
        float(left.get("confidence", 0))
        + float(right.get("confidence", 0))
    ) / 2.0
