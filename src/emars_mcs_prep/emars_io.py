"""EMARS product path helpers, date→(MY, Ls) resolution, and HTTP download."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from .config import EMARS_BASE_URL, EMARS_ROOT, EMARS_V2_PRODUCTS, REPO_SCRIPTS
from .pairs import _product_path, load_emars_time_table


def emars_filename(product: str, mars_year: str, ls_bin: str) -> str:
    return f"emars_v1.0_{product}_{mars_year}_{ls_bin}.nc"


def emars_url(product: str, mars_year: str, ls_bin: str) -> str:
    return f"{EMARS_BASE_URL}/{emars_filename(product, mars_year, ls_bin)}"


def emars_local_path(
    product: str,
    mars_year: str,
    ls_bin: str,
    emars_root: Path = EMARS_ROOT,
) -> Path:
    return _product_path(emars_root, product, mars_year, ls_bin)


def http_download(url: str, dest: Path, timeout: int = 3600) -> int:
    """Download url → dest; skip if dest already exists with size > 0."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    req = urllib.request.Request(url, headers={"User-Agent": "EMARS_MCS_data_prep/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest.stat().st_size


def ensure_emars_products(
    mars_year: str,
    ls_bin: str,
    products: Iterable[str] = EMARS_V2_PRODUCTS,
    emars_root: Path = EMARS_ROOT,
    *,
    download: bool = True,
) -> dict[str, Path]:
    """Ensure EMARS NetCDFs exist under emars_root/MY/; download if missing."""
    out: dict[str, Path] = {}
    for product in products:
        dest = emars_root / mars_year / emars_filename(product, mars_year, ls_bin)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[emars] skip existing {dest.name}")
            out[product] = dest
            continue
        # also accept legacy locations via _product_path
        try:
            existing = _product_path(emars_root, product, mars_year, ls_bin)
            if existing.exists() and existing.stat().st_size > 0:
                print(f"[emars] skip existing {existing}")
                out[product] = existing
                continue
        except FileNotFoundError:
            pass
        if not download:
            raise FileNotFoundError(f"Missing EMARS {product}: {dest}")
        url = emars_url(product, mars_year, ls_bin)
        print(f"[emars] downloading {url}")
        try:
            nbytes = http_download(url, dest)
        except urllib.error.HTTPError as e:
            raise FileNotFoundError(f"EMARS download failed ({e.code}): {url}") from e
        print(f"[emars] wrote {dest} ({nbytes / 1e9:.2f} GB)")
        out[product] = dest
    return out


def file_covers_earth_date(back_mean: Path, y: int, m: int, d: int) -> bool:
    cache = back_mean.with_suffix(back_mean.suffix + ".days.json")
    key = f"{y:04d}-{m:02d}-{d:02d}"
    if cache.exists():
        try:
            days = set(json.loads(cache.read_text(encoding="utf-8")))
            return key in days
        except (json.JSONDecodeError, OSError):
            pass

    table = load_emars_time_table(back_mean)
    days = sorted(
        {
            f"{int(table['earth_year'][i]):04d}-"
            f"{int(table['earth_month'][i]):02d}-"
            f"{int(table['earth_day'][i]):02d}"
            for i in range(table["n"])
        }
    )
    try:
        cache.write_text(json.dumps(days), encoding="utf-8")
    except OSError:
        pass
    return key in days


def resolve_emars_bin(
    earth_date: str,
    *,
    mars_year: str | None = None,
    ls_bin: str | None = None,
    emars_root: Path = EMARS_ROOT,
) -> tuple[str, str]:
    """Resolve (mars_year, ls_bin) for an Earth date.

    Prefer explicit args; else scan local back_mean files for a matching day.
    """
    y, m, d = [int(x) for x in earth_date.split("-")]
    if mars_year and ls_bin:
        return mars_year, ls_bin

    candidates: list[Path] = []
    if mars_year:
        candidates = sorted((emars_root / mars_year).glob("emars_v1.0_back_mean_*.nc"))
    else:
        candidates = sorted(emars_root.glob("MY*/emars_v1.0_back_mean_*.nc"))

    for path in candidates:
        parts = path.stem.split("_")
        my = parts[-2]
        ls = parts[-1]
        if mars_year and my != mars_year:
            continue
        if ls_bin and ls != ls_bin:
            continue
        try:
            if file_covers_earth_date(path, y, m, d):
                print(f"[emars] resolved {earth_date} → {my} {ls} via {path.name}")
                return my, ls
        except Exception as e:
            print(f"[emars] skip {path.name}: {e}")
            continue

    raise RuntimeError(
        f"Could not resolve EMARS MY/Ls for {earth_date}. "
        "Pass --mars-year and --ls-bin (e.g. MY31 Ls210-240)."
    )


def load_openmars_download():
    """Import repo download_assimilation_obs helpers."""
    if str(REPO_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(REPO_SCRIPTS))
    import download_assimilation_obs as dl  # type: ignore

    return dl
