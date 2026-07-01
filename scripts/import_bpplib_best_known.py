"""Import BPPLIB LB/UB metadata from the official Solutions.xlsx workbook."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SHEETS = {
    "sheet1.xml": "Falkenauer",
    "sheet2.xml": "Scholl",
    "sheet3.xml": "Waescher",
    "sheet4.xml": "Schwerin",
    "sheet5.xml": "Schoenfield Hard28",
}
SOURCE_URL = (
    "https://github.com/mdelorme2/BPPLIB/blob/master/"
    "Instances/Solutions/Solutions.xlsx"
)


def column(reference: str) -> str:
    match = re.match(r"[A-Z]+", reference)
    return match.group(0) if match else ""


def workbook_rows(path: Path) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.findall(".//m:t", NS))
            for item in shared_root.findall("m:si", NS)
        ]
        for sheet_file, sheet_name in SHEETS.items():
            root = ET.fromstring(archive.read(f"xl/worksheets/{sheet_file}"))
            rows = root.findall(".//m:row", NS)
            for row in rows[1:]:
                values: dict[str, str] = {}
                for cell in row.findall("m:c", NS):
                    value = cell.find("m:v", NS)
                    if value is None:
                        continue
                    text = value.text or ""
                    if cell.attrib.get("t") == "s":
                        text = shared[int(text)]
                    values[column(cell.attrib.get("r", ""))] = text.strip()
                name = values.get("A", "")
                if name:
                    values["sheet"] = sheet_name
                    results[Path(name).stem] = values
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("llm4ad/task/experiment/bp_1d_common/split_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("llm4ad/task/experiment/bp_1d_common/best_known.csv"),
    )
    args = parser.parse_args()
    official = workbook_rows(args.workbook)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["instances"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "best_known_bins", "status", "source", "checked_date"])
        for item in manifest:
            instance_id = item["instance_id"]
            row = official.get(instance_id)
            if row is None:
                writer.writerow([instance_id, "", "UNVERIFIED", "", "2026-07-01"])
                continue
            lower = int(row["B"])
            upper = int(row["C"])
            solved = row.get("D", "").casefold() == "solved" and lower == upper
            status = "OPTIMAL" if solved else "BEST_KNOWN"
            source = f"{SOURCE_URL} ({row['sheet']}; LB={lower}; UB={upper})"
            writer.writerow([instance_id, upper, status, source, "2026-07-01"])


if __name__ == "__main__":
    main()
