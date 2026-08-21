#!/usr/bin/env python
"""Fetch the NASA/IMS Test 2 bearing dataset into data/raw/ims_test2/.

Authoritative source: NASA Prognostics Center of Excellence Data Set Repository,
"Bearing Data Set" (IMS, University of Cincinnati, sponsored by Rexnord Corp.).
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

NASA's own hosting for this file has moved multiple times over the years, so this
script uses a stable, widely-cited GitHub mirror of the same Test 2 files instead of
a single fragile URL. If that mirror ever disappears, get the "Bearing Data Set"
zip directly from NASA's PCoE repository above and place its Test 2 files under
data/raw/ims_test2/ (filenames like 2004.02.12.10.32.39, no extension).

Usage:
    python scripts/download_data.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MIRROR_URL = "https://github.com/RicardoPSLopes/IMS-DATASET.git"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ims_test2"


def main() -> None:
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"{RAW_DIR} already has files in it, skipping download.")
        print("Delete it first if you want to re-fetch.")
        return

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "ims_mirror"
        print(f"Cloning {MIRROR_URL} (about 130MB, this can take a few minutes)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", MIRROR_URL, str(tmp_path)],
            check=True,
        )
        source_data_dir = tmp_path / "data"
        if not source_data_dir.exists():
            print("ERROR: mirror layout changed, expected a data/ folder in it.", file=sys.stderr)
            sys.exit(1)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in source_data_dir.iterdir():
            if f.is_file():
                shutil.move(str(f), RAW_DIR / f.name)
                n += 1
        print(f"Moved {n} snapshot files into {RAW_DIR}")

    print("Done. Run `python src/train_bearing.py` to extract features and train.")


if __name__ == "__main__":
    main()
