#!/usr/bin/env python3
"""CLI: for a given Earth date, ensure MCS + EMARS, then prepare v2 I/O + limbs.

Examples:
  python scripts/run_workflow.py --earth-date 2012-11-19
  python scripts/run_workflow.py --earth-date 2012-11-19 --hours 0 1 --force-prep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emars_mcs_prep.config import EMARS_ROOT, OBS_ROOT, TRAINING_DATA_ROOT  # noqa: E402
from emars_mcs_prep.workflow import run_workflow  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--earth-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--hours", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--mars-year", default=None, help="e.g. MY31 (auto if local EMARS covers date)")
    p.add_argument("--ls-bin", default=None, help="e.g. Ls210-240")
    p.add_argument("--emars-root", type=Path, default=EMARS_ROOT)
    p.add_argument("--mcs-root", type=Path, default=OBS_ROOT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--mcs-workers", type=int, default=8)
    p.add_argument("--no-download-mcs", action="store_true")
    p.add_argument("--no-download-emars", action="store_true")
    p.add_argument(
        "--force-prep",
        action="store_true",
        help="Re-run prepare even if output NetCDFs already exist",
    )
    args = p.parse_args()

    summary = run_workflow(
        earth_date=args.earth_date,
        hours=args.hours,
        mars_year=args.mars_year,
        ls_bin=args.ls_bin,
        emars_root=args.emars_root,
        mcs_root=args.mcs_root,
        out_dir=args.out_dir,
        download_mcs=not args.no_download_mcs,
        download_emars=not args.no_download_emars,
        skip_prep_if_exists=not args.force_prep,
        mcs_workers=args.mcs_workers,
    )
    print(json.dumps(summary, indent=2, default=str))
    out = Path(summary["out_dir"])
    print(f"\nOutputs: {out}")
    print(f"Default outputs root: {TRAINING_DATA_ROOT}")


if __name__ == "__main__":
    main()
