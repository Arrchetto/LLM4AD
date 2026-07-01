from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FSSPInstance:
    """One flow-shop scheduling instance."""

    instance_id: str
    n: int
    m: int
    matrix: tuple[tuple[int, ...], ...]
    family: str
    subseries: str

    @property
    def size(self) -> int:
        """Number of jobs; convenience alias used by manifests."""
        return self.n


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
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[5] / "data/benchmarks/fssp"


def parse_instance(
    path: str | Path,
    instance_id: str,
    family: str = "",
    subseries: str = "",
) -> FSSPInstance:
    """Read a single CO-Bench/Taillard FSSP instance file.

    The file format is the per-instance CO-Bench layout:
      - header label line
      - "n m seed upper_bound lower_bound"
      - "processing times :"
      - m lines, each containing n integer processing times for that machine
    The returned matrix stores one row per job (n x m).
    """
    path = Path(path)
    try:
        raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise FileNotFoundError(f"cannot read FSSP instance: {path}") from error

    lines = [line.strip() for line in raw_lines if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"{path}: expected header, size line, and processing-times label")

    # Size line is the first non-label, non-empty line after the header.
    size_tokens = lines[1].split()
    if len(size_tokens) < 5:
        raise ValueError(f"{path}: expected at least 5 numbers in size line")
    try:
        n = int(size_tokens[0])
        m = int(size_tokens[1])
        # seed is ignored
        upper_bound = int(size_tokens[3])
        lower_bound = int(size_tokens[4])
    except ValueError as error:
        raise ValueError(f"{path}: size line fields must be integers") from error

    if n <= 0 or m <= 0:
        raise ValueError(f"{path}: job and machine counts must be positive")
    if upper_bound < 0 or lower_bound < 0:
        raise ValueError(f"{path}: bounds must be non-negative")

    label_index = 2
    if label_index >= len(lines):
        raise ValueError(f"{path}: unexpected end of file before processing-times label")
    if not lines[label_index].lower().startswith("processing times"):
        raise ValueError(f"{path}: expected 'processing times' label")

    data_start = label_index + 1
    if len(lines) - data_start < m:
        raise ValueError(f"{path}: expected {m} processing-time rows, found {len(lines) - data_start}")

    machine_times: list[list[int]] = []
    for idx in range(data_start, data_start + m):
        tokens = lines[idx].split()
        if len(tokens) != n:
            raise ValueError(
                f"{path}: expected {n} processing times in row, found {len(tokens)}"
            )
        try:
            row = [int(token) for token in tokens]
        except ValueError as error:
            raise ValueError(f"{path}: processing times must be integers") from error
        if any(time < 0 for time in row):
            raise ValueError(f"{path}: processing times must be non-negative")
        machine_times.append(row)

    # Transpose to job-major matrix.
    matrix = tuple(
        tuple(machine_times[machine][job] for machine in range(m))
        for job in range(n)
    )

    return FSSPInstance(
        instance_id=instance_id,
        n=n,
        m=m,
        matrix=matrix,
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
) -> tuple[FSSPInstance, ...]:
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    extracted = root / "extracted"
    if not extracted.is_dir():
        raise FileNotFoundError(f"FSSP extracted directory not found: {extracted}")
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
