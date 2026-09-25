"""End-to-end: ensure MCS + EMARS for an Earth date, then prepare v2 triples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import EMARS_ROOT, EMARS_V2_PRODUCTS, OBS_ROOT, TRAINING_DATA_ROOT
from .emars_io import ensure_emars_products, resolve_emars_bin
from .mcs_io import ensure_mcs_for_date
from .pairs import _product_path, emars_hours_on_day, load_emars_time_table
from .prepare_da_hourly import EPS_DEFAULT, prepare_da_hourly


def day_output_dir(earth_date: str, outputs_root: Path = TRAINING_DATA_ROOT) -> Path:
    y, m, d = earth_date.split("-")
    return outputs_root / f"EMARS_MCS_DA_{y}{m}{d}"


def available_hours_for_day(
    earth_date: str,
    requested: list[int],
    *,
    mars_year: str,
    ls_bin: str,
    emars_root: Path = EMARS_ROOT,
) -> list[int]:
    """Intersect requested hours with EMARS timesteps present that day."""
    y, m, d = [int(x) for x in earth_date.split("-")]
    bm = _product_path(emars_root, "back_mean", mars_year, ls_bin)
    table = load_emars_time_table(bm)
    have = set(emars_hours_on_day(table, y, m, d))
    return [h for h in requested if h in have]


def outputs_complete(
    out_dir: Path,
    mars_year: str,
    earth_date: str,
    hours: list[int],
) -> bool:
    """True if every requested hour has input/output/limb NetCDFs."""
    if not hours:
        return False
    y, m, d = [int(x) for x in earth_date.split("-")]
    for hour in hours:
        stamp = f"{y:04d}{m:02d}{d:02d}{hour:02d}"
        needed = [
            out_dir / "input" / f"input_{mars_year}_{stamp}.nc",
            out_dir / "output" / f"output_{mars_year}_{stamp}.nc",
            out_dir / "mcs_limbs" / f"mcs_limb_{mars_year}_{stamp}.nc",
        ]
        if not all(p.exists() and p.stat().st_size > 0 for p in needed):
            return False
    return True


def run_workflow(
    earth_date: str,
    hours: list[int] | None = None,
    *,
    mars_year: str | None = None,
    ls_bin: str | None = None,
    emars_root: Path = EMARS_ROOT,
    mcs_root: Path = OBS_ROOT,
    out_dir: Path | None = None,
    download_mcs: bool = True,
    download_emars: bool = True,
    skip_prep_if_exists: bool = True,
    mcs_workers: int = 8,
    eps: float = EPS_DEFAULT,
) -> dict[str, Any]:
    """Download (if needed) MCS + EMARS, then write input/output/mcs_limbs."""
    hours = list(hours if hours is not None else [0, 1, 2, 3, 4, 5])
    summary: dict[str, Any] = {"earth_date": earth_date, "hours_requested": hours}

    my, ls = resolve_emars_bin(
        earth_date, mars_year=mars_year, ls_bin=ls_bin, emars_root=emars_root
    )
    summary["mars_year"] = my
    summary["ls_bin"] = ls

    emars_paths = ensure_emars_products(
        my,
        ls,
        products=EMARS_V2_PRODUCTS,
        emars_root=emars_root,
        download=download_emars,
    )
    summary["emars"] = {k: str(v) for k, v in emars_paths.items()}

    hours_use = available_hours_for_day(
        earth_date, hours, mars_year=my, ls_bin=ls, emars_root=emars_root
    )
    summary["hours"] = hours_use
    if not hours_use:
        summary["prep"] = {"status": "skipped", "reason": "no_emars_hours"}
        print(f"[prep] skip {earth_date}: no EMARS hours in requested set")
        return summary

    mcs_info = ensure_mcs_for_date(
        earth_date,
        hours_use,
        mcs_root=mcs_root,
        download=download_mcs,
        workers=mcs_workers,
    )
    summary["mcs"] = mcs_info

    out = out_dir or day_output_dir(earth_date)
    summary["out_dir"] = str(out)

    if skip_prep_if_exists and outputs_complete(out, my, earth_date, hours_use):
        print(f"[prep] skip; outputs already complete under {out} ({len(hours_use)} hours)")
        summary["prep"] = {"status": "skipped", "reason": "outputs_exist"}
        return summary

    print(f"[prep] writing v2 triples → {out}  hours={hours_use}")
    results = prepare_da_hourly(
        mars_year=my,
        ls_bin=ls,
        hours=hours_use,
        earth_date=earth_date,
        emars_root=emars_root,
        mcs_root=mcs_root,
        out_dir=out,
        eps=eps,
    )
    summary["prep"] = {"status": "ok", "pairs": results}
    return summary
