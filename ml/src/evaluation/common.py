from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from ml.src.data.split_manager import SplitConfig
from ml.src.models.common import load_target_frames


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def get_test_frames(
    target_column: str,
    split_config: SplitConfig | None = None,
):
    frames = load_target_frames(
        target_column=target_column,
        force_rebuild_split=False,
        split_config=split_config,
    )
    return frames.x_test, frames.y_test


def write_markdown_report(path: Path, title: str, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "| Metric | Value |", "|---|---:|"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines), encoding="utf-8")
