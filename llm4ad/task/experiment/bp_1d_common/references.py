from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .dataset import BPInstance


@dataclass(frozen=True)
class ReferenceValue:
    bins: int
    status: str
    source: str


def load_references(path: str | Path) -> dict[str, ReferenceValue]:
    path = Path(path)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise FileNotFoundError(f"cannot read best-known metadata: {path}") from error

    references: dict[str, ReferenceValue] = {}
    for row in rows:
        instance_id = (row.get("instance_id") or "").strip()
        value = (row.get("best_known_bins") or "").strip()
        status = (row.get("status") or "").strip().upper()
        source = (row.get("source") or "").strip()
        if not instance_id:
            raise ValueError("best-known metadata contains an empty instance_id")
        if instance_id in references:
            raise ValueError(f"duplicate best-known row: {instance_id}")
        if status not in {"OPTIMAL", "BEST_KNOWN", "LOWER_BOUND", "UNVERIFIED"}:
            raise ValueError(f"invalid reference status for {instance_id}: {status}")
        if status in {"OPTIMAL", "BEST_KNOWN", "LOWER_BOUND"}:
            try:
                bins = int(value)
            except ValueError as error:
                raise ValueError(f"invalid reference value for {instance_id}") from error
            if bins <= 0:
                raise ValueError(f"reference value must be positive: {instance_id}")
        else:
            bins = 0
        references[instance_id] = ReferenceValue(bins, status, source)
    return references


def validate_reference_coverage(
    instances: Sequence[BPInstance],
    references: Mapping[str, ReferenceValue],
    split: str,
) -> None:
    missing = [
        instance.instance_id
        for instance in instances
        if instance.instance_id not in references
    ]
    if missing:
        raise ValueError(f"missing best-known metadata for: {', '.join(missing[:5])}")
    if split == "train":
        invalid = [
            instance.instance_id
            for instance in instances
            if references[instance.instance_id].status
            not in {"OPTIMAL", "BEST_KNOWN"}
            or not references[instance.instance_id].source
        ]
        if invalid:
            raise ValueError(
                "training references must be sourced OPTIMAL or BEST_KNOWN values: "
                + ", ".join(invalid[:5])
            )
