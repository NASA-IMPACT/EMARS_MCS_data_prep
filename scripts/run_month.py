#!/usr/bin/env python3
"""Prepare EMARS–MCS v2 triples for every Earth day in a calendar month.

Uses local EMARS coverage + MCS DDR. Days without EMARS or MCS are skipped.

Example:
  python scripts/run_month.py --year 2012 --month 11 --mars-year MY31 --ls-bin Ls210-240
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emars_mcs_prep.config import EMARS_ROOT, OBS_ROOT, TRAINING_DATA_ROOT  # noqa: E402
from emars_mcs_prep.emars_io import file_covers_earth_date, resolve_emars_bin  # noqa: E402
from emars_mcs_prep.mcs_io import day_has_mcs_coverage  # noqa: E402
from emars_mcs_prep.pairs import _product_path  # noqa: E402
from emars_mcs_prep.workflow import run_workflow  # noqa: E402


def days_in_month(year: int, month: int) -> list[str]:
    n = calendar.monthrange(year, month)[1]
    return [f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, n + 1)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--hours", nargs="+", type=int, default=list(range(24)))
    p.add_argument("--mars-year", default=None)
    p.add_argument("--ls-bin", default=None)
    p.add_argument("--emars-root", type=Path, default=EMARS_ROOT)
    p.add_argument("--mcs-root", type=Path, default=OBS_ROOT)
    p.add_argument("--no-download-mcs", action="store_true")
    p.add_argument("--no-download-emars", action="store_true")
    p.add_argument("--force-prep", action="store_true")
    p.add_argument("--mcs-workers", type=int, default=4)
    p.add_argument(
        "--day-index",
        type=int,
        default=None,
        help="1-based index into eligible days (for Slurm array tasks).",
    )
    p.add_argument(
        "--list-days",
        action="store_true",
        help="Only print eligible days and exit.",
    )
    args = p.parse_args()

    candidates = days_in_month(args.year, args.month)
    eligible: list[tuple[str, str, str]] = []
    for earth_date in candidates:
        try:
            my, ls = resolve_emars_bin(
                earth_date,
                mars_year=args.mars_year,
                ls_bin=args.ls_bin,
                emars_root=args.emars_root,
            )
            bm = _product_path(args.emars_root, "back_mean", my, ls)
            y, m, d = [int(x) for x in earth_date.split("-")]
            if not file_covers_earth_date(bm, y, m, d):
                continue
            if not day_has_mcs_coverage(args.mcs_root, earth_date, args.hours):
                # still eligible if we allow download; mark for attempt
                if args.no_download_mcs:
                    continue
            eligible.append((earth_date, my, ls))
        except Exception as e:
            print(f"[skip] {earth_date}: {e}")

    print(f"Eligible days in {args.year}-{args.month:02d}: {len(eligible)}")
    for i, (d, my, ls) in enumerate(eligible, 1):
        print(f"  {i:02d} {d}  {my} {ls}")

    if args.list_days:
        TRAINING_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        manifest = TRAINING_DATA_ROOT / f"month_{args.year}{args.month:02d}_days.json"
        manifest.write_text(json.dumps([{"earth_date": d, "mars_year": my, "ls_bin": ls} for d, my, ls in eligible], indent=2))
        print(f"Wrote {manifest}")
        return

    if not eligible:
        raise SystemExit("No eligible days found.")

    if args.day_index is not None:
        if args.day_index < 1 or args.day_index > len(eligible):
            raise SystemExit(f"--day-index {args.day_index} out of range 1..{len(eligible)}")
        todo = [eligible[args.day_index - 1]]
    else:
        todo = eligible

    summaries = []
    for earth_date, my, ls in todo:
        print(f"\n======== {earth_date} ({my} {ls}) ========")
        summary = run_workflow(
            earth_date=earth_date,
            hours=args.hours,
            mars_year=my,
            ls_bin=ls,
            emars_root=args.emars_root,
            mcs_root=args.mcs_root,
            download_mcs=not args.no_download_mcs,
            download_emars=not args.no_download_emars,
            skip_prep_if_exists=not args.force_prep,
            mcs_workers=args.mcs_workers,
        )
        summaries.append(summary)

    print(json.dumps({"month": f"{args.year}-{args.month:02d}", "n": len(summaries), "days": summaries}, indent=2, default=str))


if __name__ == "__main__":
    main()
