"""Export the recorded sweep for the static project page (standard library only)."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/assets/spec_decode_sweep/results.csv"
OUTPUT = ROOT / "docs/project/results.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the export is stale")
    args = parser.parse_args()
    with SOURCE.open(newline="") as source:
        rows = [
            {
                key: (
                    None
                    if value == ""
                    else value if key == "verify_attention_model" else float(value)
                )
                for key, value in row.items()
            }
            for row in csv.DictReader(source)
        ]
    exported = json.dumps(rows, indent=2, allow_nan=False) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != exported:
            raise SystemExit("Project data is stale. Run scripts/export_project_results.py")
        print(f"Project page matches all {len(rows)} recorded CSV rows.")
    else:
        OUTPUT.write_text(exported)
        print(f"Exported {len(rows)} recorded CSV rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
