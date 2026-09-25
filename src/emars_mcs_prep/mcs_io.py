"""Ensure MCS DDR volumes covering an Earth date are present under obs/."""

from __future__ import annotations

import json
from pathlib import Path

from .config import OBS_ROOT
from .emars_io import load_openmars_download


def mcs_volume_id_for_month(year: int, month: int) -> int:
    """Map Earth calendar month → MCS PDS volume id (MROM_####)."""
    # MCS_VOLUME_EPOCH = 2006-09-01 → volume 2001
    months = (year - 2006) * 12 + (month - 9)
    return 2001 + months


def mcs_volume_name_for_date(earth_date: str) -> str:
    y, m, _ = [int(x) for x in earth_date.split("-")]
    return f"MROM_{mcs_volume_id_for_month(y, m)}"


def mcs_volume_roots(mcs_root: Path, volume: str) -> list[Path]:
    """Local roots for a monthly volume (usually one MY*/MCS/MROM_####)."""
    return sorted(mcs_root.glob(f"MY*/MCS/{volume}"))


def day_has_mcs_coverage(mcs_root: Path, earth_date: str, hours: list[int]) -> bool:
    y, m, d = [int(x) for x in earth_date.split("-")]
    chunks = sorted({(h // 4) * 4 for h in hours})
    volume = mcs_volume_name_for_date(earth_date)
    roots = mcs_volume_roots(mcs_root, volume)
    if not roots:
        return False
    stamp = f"{y:04d}{m:02d}{d:02d}"
    present: set[int] = set()
    for root in roots:
        for chunk in chunks:
            name = f"{stamp}{chunk:02d}_DDR.TAB"
            if any(root.rglob(name)):
                present.add(chunk)
    return all(c in present for c in chunks)


def ensure_mcs_for_date(
    earth_date: str,
    hours: list[int],
    *,
    mcs_root: Path = OBS_ROOT,
    download: bool = True,
    workers: int = 8,
) -> dict:
    """Skip if MCS DDR for the date/hours exists; else download the monthly volume."""
    volume = mcs_volume_name_for_date(earth_date)
    if day_has_mcs_coverage(mcs_root, earth_date, hours):
        print(f"[mcs] skip download; DDR present for {earth_date} hours={hours}")
        return {"status": "skipped", "volume": volume, "reason": "day_coverage"}

    if not download:
        raise FileNotFoundError(
            f"MCS DDR missing for {earth_date} hours={hours} under {mcs_root}"
        )

    dl = load_openmars_download()
    from openmars_config import DownloadTask, assign_mcs_volume, my_dest  # type: ignore

    vol_id = int(volume.split("_", 1)[1])
    mars_year = assign_mcs_volume(vol_id)
    if mars_year is None:
        raise RuntimeError(f"No OpenMARS MY assignment for {volume}")

    dest = my_dest(mars_year, "MCS", volume)
    cache = dl.load_mcs_volume_cache()
    tab_count = int(cache.get(volume, {}).get("tab_count") or 0)
    task = DownloadTask(
        task_id=0,
        kind="mcs_volume",
        instrument="MCS",
        mars_year=mars_year,
        label=volume,
        source=f"{dl.ATMOS_PDS}/{volume}/DATA",
        dest_dir=str(dest),
        extra=json.dumps({"tab_count": tab_count}),
    )
    print(f"[mcs] downloading {volume} → {dest} (MY={mars_year})")
    rec = dl.run_mcs_volume(task, workers=workers)
    print(f"[mcs] status={rec.status} files={rec.files} error={rec.error!r}")

    if not day_has_mcs_coverage(mcs_root, earth_date, hours):
        raise FileNotFoundError(
            f"After download, still no MCS DDR for {earth_date} hours={hours}"
        )
    return {
        "status": rec.status,
        "volume": volume,
        "mars_year": mars_year,
        "dest_dir": str(dest),
        "files": rec.files,
    }
