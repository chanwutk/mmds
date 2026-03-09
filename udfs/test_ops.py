def add_bucket(row: dict[str, int]) -> dict[str, int]:
    return {"bucket": row["value"] // 2}


def keep_large(row: dict[str, int]) -> bool:
    return row["value"] >= 2


def summarize_group(rows: list[dict[str, int]]) -> dict[str, int]:
    return {"total": sum(row["value"] for row in rows), "count": len(rows)}


def annotate(row: dict[str, int]) -> dict[str, str]:
    return {"label": f"v{row['value']}"}
