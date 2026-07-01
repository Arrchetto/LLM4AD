from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BPInstance:
    instance_id: str
    num_items: int
    bin_capacity: int
    items: tuple[int, ...]
    family: str
    subseries: str


@dataclass(frozen=True)
class ManifestEntry:
    instance_id: str
    relative_path: str
    family: str
    subseries: str
    size: int
    split: str


COMMON_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = COMMON_DIR / "split_manifest.json"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[5] / "data/benchmarks/bp_1d"


def parse_instance(
    path: str | Path,
    instance_id: str,
    family: str = "",
    subseries: str = "",
) -> BPInstance:
    path = Path(path)
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except OSError as error:
        raise FileNotFoundError(f"cannot read BPPLIB instance: {path}") from error
    if len(lines) < 2:
        raise ValueError(f"{path}: expected item count and bin capacity")
    try:
        num_items = int(lines[0])
        bin_capacity = int(lines[1])
        items = tuple(int(value) for value in lines[2:])
    except ValueError as error:
        raise ValueError(f"{path}: BPPLIB fields must be integers") from error
    if num_items <= 0 or bin_capacity <= 0:
        raise ValueError(f"{path}: item count and capacity must be positive")
    if len(items) != num_items:
        raise ValueError(
            f"{path}: expected {num_items} item sizes, found {len(items)}"
        )
    if any(size <= 0 or size > bin_capacity for size in items):
        raise ValueError(f"{path}: every item must be in (0, bin_capacity]")
    return BPInstance(
        instance_id=instance_id,
        num_items=num_items,
        bin_capacity=bin_capacity,
        items=items,
        family=family,
        subseries=subseries,
    )


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> tuple[ManifestEntry, ...]:
    path = Path(path)
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"cannot read split manifest: {path}") from error
    entries = tuple(ManifestEntry(**entry) for entry in payload.get("instances", []))
    if not entries:
        raise ValueError(f"split manifest contains no instances: {path}")
    paths = [entry.relative_path for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("split manifest contains duplicate instance paths")
    return entries


def load_split(
    split: str,
    data_root: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> tuple[BPInstance, ...]:
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    extracted = root / "extracted"
    if not extracted.is_dir():
        raise FileNotFoundError(f"BPPLIB extracted directory not found: {extracted}")
    selected = [entry for entry in load_manifest(manifest_path) if entry.split == split]
    if not selected:
        raise ValueError(f"unknown or empty dataset split: {split}")
    return tuple(
        parse_instance(
            extracted / entry.relative_path,
            entry.instance_id,
            entry.family,
            entry.subseries,
        )
        for entry in selected
    )

