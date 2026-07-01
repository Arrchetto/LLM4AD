"""Generate the frozen BPPLIB split manifest and blank reference ledger."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data/benchmarks/bp_1d/extracted"
OUTPUT = REPO_ROOT / "llm4ad/task/experiment/bp_1d_common"
SEED = 20260701


def size(path: Path) -> int:
    return int(path.read_text(encoding="utf-8-sig").splitlines()[0].strip())


def identity(path: Path) -> tuple[str, str]:
    text = path.as_posix()
    if "Falkenauer U" in text:
        return "Falkenauer", "Falkenauer U"
    if "Falkenauer_T" in text:
        return "Falkenauer", "Falkenauer T"
    if "Scholl_1" in text:
        return "Scholl", "Scholl 1"
    if "Scholl_2" in text:
        return "Scholl", "Scholl 2"
    if "Scholl_3" in text:
        return "Scholl", "Scholl 3"
    if "Hard28" in text:
        return "Hard28", "Hard28"
    if "Schwerin_1" in text:
        return "Schwerin", "Schwerin 1"
    if "Schwerin_2" in text:
        return "Schwerin", "Schwerin 2"
    return "Waescher", "Waescher"


def entry(path: Path, split: str) -> dict[str, object]:
    family, subseries = identity(path)
    return {
        "instance_id": path.stem,
        "relative_path": path.relative_to(DATA_ROOT).as_posix(),
        "family": family,
        "subseries": subseries,
        "size": size(path),
        "split": split,
    }


def choose(paths: list[Path], count: int, rng: random.Random) -> list[Path]:
    ordered = sorted(paths, key=lambda path: path.relative_to(DATA_ROOT).as_posix())
    rng.shuffle(ordered)
    return sorted(ordered[:count], key=lambda path: path.relative_to(DATA_ROOT).as_posix())


def main() -> None:
    rng = random.Random(SEED)
    paths = sorted(DATA_ROOT.rglob("*.txt"))
    selected: list[dict[str, object]] = []
    used: set[Path] = set()

    training_strata = [
        ("Falkenauer U", 120, 8),
        ("Falkenauer T", 120, 8),
        ("Scholl 1", 100, 7),
        ("Scholl 2", 100, 7),
    ]
    for subseries, target_size, count in training_strata:
        candidates = [
            path
            for path in paths
            if identity(path)[1] == subseries and size(path) == target_size
        ]
        train = choose(candidates, count, rng)
        validation = choose([path for path in candidates if path not in train], count, rng)
        selected.extend(entry(path, "train") for path in train)
        selected.extend(entry(path, "validation") for path in validation)
        used.update(train)
        used.update(validation)

    cross_family = [
        path for path in paths if identity(path)[0] in {"Waescher", "Hard28", "Schwerin"}
    ]
    selected.extend(entry(path, "test_cross_family") for path in cross_family)
    used.update(cross_family)

    scale_strata = [
        ("Falkenauer U", 250),
        ("Falkenauer U", 500),
        ("Falkenauer U", 1000),
        ("Falkenauer T", 249),
        ("Falkenauer T", 501),
        ("Scholl 1", 200),
        ("Scholl 1", 500),
        ("Scholl 2", 200),
        ("Scholl 2", 500),
    ]
    for subseries, target_size in scale_strata:
        candidates = [
            path
            for path in paths
            if path not in used
            and identity(path)[1] == subseries
            and size(path) == target_size
        ]
        sample = choose(candidates, 4, rng)
        selected.extend(entry(path, "test_cross_scale") for path in sample)
        used.update(sample)

    selected.sort(key=lambda item: (str(item["split"]), str(item["relative_path"])))
    payload = {
        "protocol": "bp-1d-complete-optimizer-v1",
        "selection_seed": SEED,
        "instances": selected,
    }
    (OUTPUT / "split_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT / "best_known.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "best_known_bins", "status", "source", "checked_date"])
        for item in selected:
            writer.writerow([item["instance_id"], "", "UNVERIFIED", "", "2026-07-01"])


if __name__ == "__main__":
    main()
