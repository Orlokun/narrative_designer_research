"""
Reset script — wipe all harvested pipeline data.

Usage:
    python scripts/reset_data.py          # shows what will be deleted, does nothing
    python scripts/reset_data.py --yes    # deletes everything
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

TARGETS = [
    (ROOT / "data/archivo/archivo.sqlite", "file"),
    (ROOT / "data/missions.json",          "file"),
    (ROOT / "data/documents.json",         "file"),
    (ROOT / "data/logs",                   "dir"),
    (ROOT / "data/gatekeeper_cache",       "dir"),
]


def main() -> None:
    confirmed = "--yes" in sys.argv

    if not confirmed:
        print()
        print("  Will delete:")
        for path, _ in TARGETS:
            exists = "exists" if path.exists() else "absent"
            print(f"    {path.relative_to(ROOT)}  ({exists})")
        print()
        print("  Run with --yes to confirm:  make reset CONFIRM=yes")
        print()
        sys.exit(1)

    deleted = 0
    for path, kind in TARGETS:
        if not path.exists():
            continue
        if kind == "file":
            path.unlink()
        else:
            shutil.rmtree(path)
        print(f"  Deleted  {path.relative_to(ROOT)}")
        deleted += 1

    if deleted == 0:
        print("  Nothing to delete — already clean.")
    else:
        print(f"\n  Reset complete ({deleted} items deleted).")


if __name__ == "__main__":
    main()
