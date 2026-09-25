#!/usr/bin/env python3
"""Prepare ML-ready EMARS state + MCS limb-T observation NetCDF pairs (one hour each).

Example:
  python prepare_emars_mcs_hourly_pairs.py \\
      --mars-year MY31 --ls-bin Ls210-240 \\
      --hours 0 1 2 3 4 5

If --earth-date is omitted, the first Earth day in the EMARS window that has all
requested hours and matching MCS DDR coverage is selected automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from .config import EMARS_ROOT, OBS_ROOT, OUTPUTS_ROOT, REPO_ROOT

REPO = REPO_ROOT
DEFAULT_EMARS_ROOT = EMARS_ROOT
DEFAULT_MCS_ROOT = OBS_ROOT  # searches MY*/MCS
DEFAULT_OUT = OUTPUTS_ROOT / "samples"

FILL = -9999.0
N_MCS_LEVELS = 105
P0_PA = 610.0
EPS_DEFAULT = 1.0e-6

# MCS QC used in notebooks/coords_MCS.ipynb (stored as mask; profiles kept as-is)
QC_GQUAL_OK = {0}
QC_PQUAL_OK = {0}
QC_TQUAL_OK = {0, 3}

TIME_META_VARS = {
    "time",
    "emars_sol",
    "mars_hour",
    "MY",
    "Ls",
    "earth_year",
    "earth_month",
    "earth_day",
    "earth_hour",
    "earth_minute",
    "earth_second",
    "mars_soy",
    "macda_sol",
}

STATIC_1D = {"ak", "bk", "lat", "lon", "latu", "lonv", "pfull", "phalf"}

# Option A: temperature-only DA core profile (see AR_v2_MARSDA deck)
DA_CORE_T_TIME_META = {"Ls", "MY", "mars_hour", "earth_year", "earth_month", "earth_day", "earth_hour"}
DA_CORE_T_STATIC = {"ak", "bk", "lat", "lon", "pfull", "phalf"}
DA_CORE_T_CONTEXT = {
    "surface_geopotential": ("anal_sprd", "Surface_geopotential"),
    "dod": ("back_mean", "dod"),
}
# native variable names per EMARS product for temperature
DA_CORE_T_STATE = {
    "xb": ("back_mean", "t"),
    "xa": ("anal_mean", "T"),
    "sigma_a": ("anal_sprd", "T"),
    "xb_memb": ("back_memb", "t"),
}

MCS_DA_CORE_T_PROFILE_FIELDS = [
    ("ls", "degree", "areocentric longitude", "ls", "L_s"),
    ("profile_lat", "degrees_north", "profile latitude", "profile_lat", "Profile_lat"),
    ("profile_lon", "degrees_east", "profile longitude", "profile_lon", "Profile_lon"),
    ("ltst", "hour", "local true solar time", "ltst", "LTST"),
    ("gqual", "1", "geometry quality flag", "gqual", "Gqual"),
    ("p_qual", "1", "pressure quality flag", "p_qual", "P_qual"),
    ("t_qual", "1", "temperature quality flag", "t_qual", "T_qual"),
    ("obs_qual", "1", "observation quality / viewing mode", "obs_qual", "Obs_qual"),
]
MCS_DA_CORE_T_LEVEL_FIELDS = [
    ("pressure", "Pa", "retrieved pressure on limb levels", "pressure_pa", "Pressure"),
    ("temperature", "K", "limb temperature retrieval (y)", "temperature_k", "Temperature"),
    ("temperature_err", "K", "observation error σ_o", "temperature_err_k", "Temp_err"),
]


def _product_path(emars_dir: Path, product: str, mars_year: str, ls_bin: str) -> Path:
    name = f"emars_v1.0_{product}_{mars_year}_{ls_bin}.nc"
    direct = emars_dir / mars_year / name
    if direct.exists() or direct.is_symlink():
        return direct
    # allow files living directly under EMARS/ or repo root
    for cand in (emars_dir / name, REPO / name):
        if cand.exists() or cand.is_symlink():
            return cand
    raise FileNotFoundError(f"Missing EMARS product: {direct}")


def _copy_attrs(src_var, dst_var, extra: dict[str, str] | None = None) -> None:
    for a in src_var.ncattrs():
        if a in {"_FillValue"}:
            continue
        try:
            dst_var.setncattr(a, src_var.getncattr(a))
        except (TypeError, ValueError):
            dst_var.setncattr(a, str(src_var.getncattr(a)))
    if extra:
        for k, v in extra.items():
            dst_var.setncattr(k, v)


def _read_slice(var, t_idx: int):
    data = var[t_idx, ...] if var.ndim >= 1 and var.dimensions and var.dimensions[0] == "time" else var[...]
    return np.array(data)


def _canonical(name: str) -> str:
    return name.lower()


def _find_var(ds: Dataset, logical: str) -> str | None:
    """Map logical names across EMARS products (t/T, u/U, ...)."""
    keys = {k.lower(): k for k in ds.variables}
    return keys.get(logical.lower())


def load_emars_time_table(back_mean_path: Path) -> dict[str, Any]:
    ds = Dataset(back_mean_path)
    try:
        n = len(ds.dimensions["time"])
        table = {
            "time": np.array(ds.variables["time"][:]),
            "MY": np.array(ds.variables["MY"][:]),
            "Ls": np.array(ds.variables["Ls"][:]),
            "mars_hour": np.array(ds.variables["mars_hour"][:]),
            "earth_year": np.array(ds.variables["earth_year"][:], dtype=int),
            "earth_month": np.array(ds.variables["earth_month"][:], dtype=int),
            "earth_day": np.array(ds.variables["earth_day"][:], dtype=int),
            "earth_hour": np.array(ds.variables["earth_hour"][:], dtype=int),
            "earth_minute": np.array(ds.variables["earth_minute"][:]),
            "earth_second": np.array(ds.variables["earth_second"][:]),
        }
    finally:
        ds.close()
    table["n"] = n
    return table


def find_time_index(table: dict[str, Any], y: int, m: int, d: int, hour: int) -> int:
    hits = np.where(
        (table["earth_year"] == y)
        & (table["earth_month"] == m)
        & (table["earth_day"] == d)
        & (table["earth_hour"] == hour)
    )[0]
    if len(hits) == 0:
        raise ValueError(f"No EMARS timestep for {y:04d}-{m:02d}-{d:02d} hour={hour:02d}")
    return int(hits[0])


def emars_hours_on_day(table: dict[str, Any], y: int, m: int, d: int) -> list[int]:
    """Earth hours present in the EMARS time table for one calendar day."""
    hits = np.where(
        (table["earth_year"] == y) & (table["earth_month"] == m) & (table["earth_day"] == d)
    )[0]
    return sorted({int(table["earth_hour"][i]) for i in hits})


def list_days_with_hours(table: dict[str, Any], hours: list[int]) -> list[tuple[int, int, int]]:
    want = set(hours)
    by_day: dict[tuple[int, int, int], set[int]] = {}
    for i in range(table["n"]):
        day = (int(table["earth_year"][i]), int(table["earth_month"][i]), int(table["earth_day"][i]))
        by_day.setdefault(day, set()).add(int(table["earth_hour"][i]))
    return [day for day, hrs in sorted(by_day.items()) if want.issubset(hrs)]


def find_mcs_ddr_for_hour(mcs_root: Path, y: int, m: int, d: int, hour: int) -> list[Path]:
    """MCS DDR files are 4-hour chunks named YYYYMMDDHH with HH in {0,4,8,12,16,20}.

    Search only the monthly volume tree (fast); fall back to full MY*/MCS glob.
    """
    stamp = f"{y:04d}{m:02d}{d:02d}"
    chunk = (hour // 4) * 4
    name = f"{stamp}{chunk:02d}_DDR.TAB"
    # Prefer volume-scoped search (avoids walking all Mars years).
    months_since = (y - 2006) * 12 + (m - 9)
    volume = f"MROM_{2001 + months_since}"
    hits: list[Path] = []
    for root in mcs_root.glob(f"MY*/MCS/{volume}"):
        hits.extend(root.rglob(name))
    if hits:
        return sorted(hits)
    pattern = f"**/{name}"
    return sorted(mcs_root.glob(f"MY*/MCS/{pattern}"))


def day_has_mcs_for_hours(mcs_root: Path, day: tuple[int, int, int], hours: list[int]) -> bool:
    y, m, d = day
    for h in hours:
        if not find_mcs_ddr_for_hour(mcs_root, y, m, d, h):
            return False
    return True


def auto_select_earth_date(
    table: dict[str, Any],
    mcs_root: Path,
    hours: list[int],
) -> tuple[int, int, int]:
    for day in list_days_with_hours(table, hours):
        if day_has_mcs_for_hours(mcs_root, day, hours):
            return day
    raise RuntimeError(
        "No Earth day found with all requested EMARS hours and MCS DDR coverage. "
        "Pass --earth-date explicitly or download overlapping MCS/EMARS."
    )


def _f(row: list[str], idx: dict[str, int], name: str) -> float:
    try:
        v = float(row[idx[name]].strip().strip('"'))
        return np.nan if v <= FILL else v
    except (KeyError, ValueError, IndexError):
        return np.nan


def parse_utc_hour(utc: str) -> int | None:
    """Parse hour from MCS UTC strings like 2012-11-19T00:12:34.567."""
    utc = utc.strip().strip('"')
    m = re.search(r"T(\d{2}):", utc)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{2}):\d{2}:\d{2}", utc)
    if m:
        return int(m.group(1))
    return None


def read_mcs_limb_t_for_hour(paths: list[Path], hour: int) -> dict[str, Any]:
    """Read limb temperature profiles for one exact UTC hour; keep flags; no interpolation."""
    profiles: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(errors="replace")
        lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
        rows = list(csv.reader(lines))
        if len(rows) < 3:
            continue
        hdr = [h.strip() for h in rows[0]]
        idx = {h: i for i, h in enumerate(hdr)}
        i = 2
        while i < len(rows):
            row = rows[i]
            if len(row) < 5 or row[0].strip() != "0" or '"' not in row[1]:
                i += 1
                continue
            utc = row[idx["UTC"]].strip().strip('"')
            uh = parse_utc_hour(utc)
            summary = {
                "utc": utc,
                "utc_hour": uh if uh is not None else -1,
                "ls": _f(row, idx, "L_s"),
                "orb_num": _f(row, idx, "Orb_num"),
                "profile_lat": _f(row, idx, "Profile_lat"),
                "profile_lon": _f(row, idx, "Profile_lon"),
                "profile_alt_km": _f(row, idx, "Profile_alt"),
                "ltst": _f(row, idx, "LTST"),
                "gqual": _f(row, idx, "Gqual"),
                "p_qual": _f(row, idx, "P_qual"),
                "t_qual": _f(row, idx, "T_qual"),
                "dust_qual": _f(row, idx, "Dust_qual"),
                "h2oice_qual": _f(row, idx, "H2Oice_qual"),
                "co2ice_qual": _f(row, idx, "CO2ice_qual"),
                "obs_qual": _f(row, idx, "Obs_qual"),
                "source_file": str(path),
            }
            i += 1
            pres = np.full(N_MCS_LEVELS, np.nan, dtype=np.float64)
            temp = np.full(N_MCS_LEVELS, np.nan, dtype=np.float64)
            terr = np.full(N_MCS_LEVELS, np.nan, dtype=np.float64)
            alt = np.full(N_MCS_LEVELS, np.nan, dtype=np.float64)
            lat_lev = np.full(N_MCS_LEVELS, np.nan, dtype=np.float64)
            lon_lev = np.full(N_MCS_LEVELS, np.nan, dtype=np.float64)
            k = 0
            while i < len(rows) and rows[i][0].strip() == "0" and '"' not in rows[i][1]:
                lr = rows[i]
                try:
                    p = float(lr[1].strip())
                    t = float(lr[2].strip())
                    te = float(lr[3].strip()) if len(lr) > 3 else FILL
                    a = float(lr[12].strip()) if len(lr) > 12 else FILL
                    la = float(lr[13].strip()) if len(lr) > 13 else FILL
                    lo = float(lr[14].strip()) if len(lr) > 14 else FILL
                except (ValueError, IndexError):
                    i += 1
                    continue
                if k < N_MCS_LEVELS:
                    if p > 0:
                        pres[k] = p
                    if t > FILL:
                        temp[k] = t
                    if te > FILL:
                        terr[k] = te
                    if a > FILL:
                        alt[k] = a
                    if la > FILL:
                        lat_lev[k] = la
                    if lo > FILL:
                        lon_lev[k] = lo
                    k += 1
                i += 1

            if uh != hour:
                continue

            gq, pq, tq = summary["gqual"], summary["p_qual"], summary["t_qual"]
            qc_pass = (
                (not np.isnan(gq) and int(gq) in QC_GQUAL_OK)
                and (not np.isnan(pq) and int(pq) in QC_PQUAL_OK)
                and (not np.isnan(tq) and int(tq) in QC_TQUAL_OK)
            )
            summary.update(
                {
                    "pressure_pa": pres,
                    "temperature_k": temp,
                    "temperature_err_k": terr,
                    "altitude_km": alt,
                    "level_lat": lat_lev,
                    "level_lon": lon_lev,
                    "qc_pass": qc_pass,
                }
            )
            profiles.append(summary)

    level_index = np.arange(1, N_MCS_LEVELS + 1, dtype=np.int32)
    p_formula = P0_PA * np.exp(-0.125 * (level_index - 10))
    return {
        "profiles": profiles,
        "level_index": level_index,
        "pressure_formula_pa": p_formula,
    }


def write_mcs_nc(
    out_path: Path,
    bundle: dict[str, Any],
    y: int,
    m: int,
    d: int,
    hour: int,
    source_files: list[Path],
    profile: str = "full",
) -> None:
    profiles = bundle["profiles"]
    n_prof = len(profiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("profile", n_prof)
        ds.createDimension("level", N_MCS_LEVELS)

        ds.setncattr("title", "MCS limb temperature soundings for one Earth hour")
        ds.setncattr("Conventions", "CF-1.8")
        ds.setncattr("profile", profile)
        ds.setncattr("instrument", "Mars Climate Sounder (MCS) on MRO")
        ds.setncattr("product", "DDR limb retrievals (temperature only)")
        ds.setncattr(
            "time_selection",
            f"Exact Earth UTC hour match: {y:04d}-{m:02d}-{d:02d}T{hour:02d}:00–{hour:02d}:59",
        )
        ds.setncattr("source_files", ";".join(str(p) for p in source_files))
        ds.setncattr(
            "qc_rule",
            "qc_pass true iff Gqual in {0} and P_qual in {0} and T_qual in {0,3}; "
            "profiles are stored as-is (not filtered)",
        )
        ds.setncattr("n_profiles", n_prof)
        ds.setncattr("n_profiles_qc_pass", int(sum(1 for p in profiles if p["qc_pass"])))
        ds.setncattr("created_utc", datetime.now(timezone.utc).isoformat())
        ds.setncattr(
            "pressure_formula",
            "p(i)=610*exp(-0.125*(i-10)) Pa for level index i=1..105 (MCS DDR reference)",
        )

        lev = ds.createVariable("level", "i4", ("level",))
        lev[:] = bundle["level_index"]
        lev.setncattr("long_name", "MCS limb level index")
        lev.setncattr("units", "1")

        pf = ds.createVariable("pressure_formula_pa", "f8", ("level",))
        pf[:] = bundle["pressure_formula_pa"]
        pf.setncattr("long_name", "MCS reference pressure from closed-form formula")
        pf.setncattr("units", "Pa")
        pf.setncattr("equation", "p(i)=610*exp(-0.125*(i-10))")

        def create_prof(name, dtype, units, long_name, values, **extra):
            var = ds.createVariable(name, dtype, ("profile",), fill_value=np.nan if "f" in dtype else -1)
            var[:] = values
            var.setncattr("long_name", long_name)
            if units:
                var.setncattr("units", units)
            for k, v in extra.items():
                var.setncattr(k, v)
            return var

        if profile == "full":
            create_prof(
                "utc_hour",
                "i4",
                "hour",
                "Earth UTC hour extracted from MCS UTC string",
                np.array([p["utc_hour"] for p in profiles], dtype=np.int32),
                native_field="parsed from UTC",
            )

        # store UTC as fixed-length strings
        utc_var = ds.createVariable("utc", str, ("profile",))
        for i, p in enumerate(profiles):
            utc_var[i] = p["utc"]
        utc_var.setncattr("long_name", "MCS profile UTC timestamp (native)")
        utc_var.setncattr("native_field", "UTC")
        utc_var.setncattr("ml_role", "observation time / mask context")

        if profile == "full":
            src_var = ds.createVariable("source_file", str, ("profile",))
            for i, p in enumerate(profiles):
                src_var[i] = p["source_file"]
            src_var.setncattr("long_name", "Absolute path of source MCS DDR.TAB")

        profile_fields = MCS_DA_CORE_T_PROFILE_FIELDS if profile == "da_core_t" else [
            ("ls", "degree", "areocentric longitude", "ls", "L_s"),
            ("orb_num", "1", "orbit number", "orb_num", "Orb_num"),
            ("profile_lat", "degrees_north", "profile latitude", "profile_lat", "Profile_lat"),
            ("profile_lon", "degrees_east", "profile longitude", "profile_lon", "Profile_lon"),
            ("profile_alt_km", "km", "profile altitude", "profile_alt_km", "Profile_alt"),
            ("ltst", "hour", "local true solar time", "ltst", "LTST"),
            ("gqual", "1", "geometry quality flag", "gqual", "Gqual"),
            ("p_qual", "1", "pressure quality flag", "p_qual", "P_qual"),
            ("t_qual", "1", "temperature quality flag", "t_qual", "T_qual"),
            ("dust_qual", "1", "dust quality flag", "dust_qual", "Dust_qual"),
            ("h2oice_qual", "1", "water-ice quality flag", "h2oice_qual", "H2Oice_qual"),
            ("co2ice_qual", "1", "CO2-ice quality flag", "co2ice_qual", "CO2ice_qual"),
            ("obs_qual", "1", "observation quality / viewing mode", "obs_qual", "Obs_qual"),
        ]
        for name, units, long_name, key, native in profile_fields:
            create_prof(
                name,
                "f8",
                units,
                long_name,
                np.array([p[key] for p in profiles], dtype=np.float64),
                native_field=native,
            )

        qc = ds.createVariable("qc_pass", "i1", ("profile",))
        qc[:] = np.array([1 if p["qc_pass"] else 0 for p in profiles], dtype=np.int8)
        qc.setncattr("long_name", "QC keep mask (1=pass)")
        qc.setncattr(
            "equation",
            "qc_pass = 1 if (Gqual in {0}) and (P_qual in {0}) and (T_qual in {0,3}) else 0",
        )
        qc.setncattr("note", "Data arrays are not filtered by this mask")

        def create_lev(name, units, long_name, key, native):
            var = ds.createVariable(name, "f8", ("profile", "level"), fill_value=np.nan)
            if n_prof:
                var[:, :] = np.stack([p[key] for p in profiles], axis=0)
            var.setncattr("long_name", long_name)
            var.setncattr("units", units)
            var.setncattr("native_field", native)
            var.setncattr("missing_value_source", f"DDR fill <= {FILL} converted to NaN")
            return var

        level_fields = MCS_DA_CORE_T_LEVEL_FIELDS if profile == "da_core_t" else [
            ("pressure", "Pa", "retrieved / reported pressure on limb levels", "pressure_pa", "Pressure"),
            ("temperature", "K", "limb temperature retrieval", "temperature_k", "Temperature"),
            ("temperature_err", "K", "limb temperature retrieval uncertainty", "temperature_err_k", "Temp_err"),
            ("altitude", "km", "limb point altitude", "altitude_km", "Altitude"),
            ("level_lat", "degrees_north", "latitude at limb level", "level_lat", "Lat"),
            ("level_lon", "degrees_east", "longitude at limb level", "level_lon", "Lon"),
        ]
        for name, units, long_name, key, native in level_fields:
            create_lev(name, units, long_name, key, native)
            if profile == "da_core_t" and name == "temperature":
                ds.variables[name].setncattr("ml_role", "observation y")
            if profile == "da_core_t" and name == "temperature_err":
                ds.variables[name].setncattr("ml_role", "observation error σ_o")

        if profile == "da_core_t":
            ds.setncattr(
                "variable_manifest",
                json.dumps(
                    {
                        "observation": ["temperature", "temperature_err", "pressure"],
                        "geometry": ["profile_lat", "profile_lon", "ls", "ltst"],
                        "quality": ["gqual", "p_qual", "t_qual", "obs_qual", "qc_pass"],
                        "coords": ["level", "pressure_formula_pa"],
                    }
                ),
            )


def write_emars_da_core_t_nc(
    out_path: Path,
    paths: dict[str, Path],
    t_idx: int,
    eps: float,
    y: int,
    m: int,
    d: int,
    hour: int,
) -> None:
    """Temperature-only DA core: xb, sigma_b, targets, minimal context. No xb_memb stored."""
    ds_bm = Dataset(paths["back_mean"])
    ds_am = Dataset(paths["anal_mean"])
    ds_as = Dataset(paths["anal_sprd"])
    ds_mb = Dataset(paths["back_memb"])

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(out_path, "w", format="NETCDF4") as out:
            for dim, size in ds_bm.dimensions.items():
                if dim in {"time", "latu", "lonv"}:
                    continue
                out.createDimension(dim, len(size))

            out.setncattr("title", "EMARS DA-core T sample (single Earth hour)")
            out.setncattr("Conventions", "CF-1.8")
            out.setncattr("profile", "da_core_t")
            out.setncattr(
                "earth_time",
                f"{y:04d}-{m:02d}-{d:02d}T{hour:02d}:00 (matched on EMARS earth_year/month/day/hour)",
            )
            out.setncattr("emars_time_index", int(t_idx))
            out.setncattr("eps", float(eps))
            out.setncattr(
                "sigma_b_definition",
                "sigma_b = |back_memb.t - back_mean.t| (single-member prior spread proxy)",
            )
            out.setncattr("created_utc", datetime.now(timezone.utc).isoformat())
            out.setncattr("source_back_mean", str(paths["back_mean"]))
            out.setncattr("source_anal_mean", str(paths["anal_mean"]))
            out.setncattr("source_anal_sprd", str(paths["anal_sprd"]))
            out.setncattr("source_back_memb", str(paths["back_memb"]))
            out.setncattr(
                "dropped_note",
                "Full-state tracers, xb_memb arrays, winds, and extra time metadata omitted; "
                "see samples/ full profile for complete export.",
            )

            for name in sorted(DA_CORE_T_STATIC):
                for label, ds_i, path_i in (
                    ("back_mean", ds_bm, paths["back_mean"]),
                    ("anal_mean", ds_am, paths["anal_mean"]),
                ):
                    if name not in ds_i.variables:
                        continue
                    src = ds_i.variables[name]
                    var = out.createVariable(name, src.dtype, src.dimensions)
                    var[:] = src[:]
                    _copy_attrs(
                        src,
                        var,
                        extra={
                            "source_product": label,
                            "source_filename": path_i.name,
                            "source_variable": name,
                            "ml_role": "grid coordinate",
                        },
                    )
                    break

            for name in sorted(DA_CORE_T_TIME_META):
                src = ds_bm.variables[name]
                val = float(np.array(src[t_idx]).item())
                var = out.createVariable(name, "f8")
                var.assignValue(val)
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": "back_mean",
                        "source_filename": paths["back_mean"].name,
                        "source_variable": name,
                        "ml_role": "Mars/time context",
                    },
                )

            def read_state(label: str, src_name: str) -> tuple[np.ndarray, tuple[str, ...], Any, Path]:
                product, native = DA_CORE_T_STATE[label]
                src_ds = {"back_mean": ds_bm, "anal_mean": ds_am, "anal_sprd": ds_as, "back_memb": ds_mb}[product]
                src_path = paths[{"back_mean": "back_mean", "anal_mean": "anal_mean", "anal_sprd": "anal_sprd", "back_memb": "back_memb"}[product]]
                src = src_ds.variables[native]
                data = _read_slice(src, t_idx)
                dims = tuple(d for d in src.dimensions if d != "time")
                return data, dims, src, src_path

            xb, xb_dims, xb_src, xb_path = read_state("xb", "t")
            xa, xa_dims, xa_src, xa_path = read_state("xa", "T")
            sa, sa_dims, sa_src, sa_path = read_state("sigma_a", "T")
            mb, mb_dims, mb_src, mb_path = read_state("xb_memb", "t")

            for out_name, data, dims, src, src_path, product, role in [
                ("xb_t", xb, xb_dims, xb_src, xb_path, "back_mean", "prior mean x^b"),
                ("xa_t", xa, xa_dims, xa_src, xa_path, "anal_mean", "analysis mean x^a (supervision)"),
                ("sigma_a_t", sa, sa_dims, sa_src, sa_path, "anal_sprd", "posterior spread σ^a (target)"),
            ]:
                var = out.createVariable(out_name, "f4", dims)
                var[:] = data
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": product,
                        "source_filename": src_path.name,
                        "source_variable": src.name,
                        "ml_role": role,
                    },
                )

            inc = xa.astype(np.float64) - xb.astype(np.float64)
            sig_b = np.abs(mb.astype(np.float64) - xb.astype(np.float64))
            inc_norm = inc / (sig_b + eps)
            r = 1.0 - (sa.astype(np.float64) ** 2) / (sig_b.astype(np.float64) ** 2 + eps)

            for out_name, data, equation, role in [
                ("increment_t", inc, "increment = x^a - x^b", "analysis increment Δx (supervision)"),
                ("sigma_b_t", sig_b, "sigma_b = |x^{b,memb} - x^b|", "prior spread σ^b (input)"),
                ("increment_norm_t", inc_norm, "increment_norm = (x^a - x^b) / (sigma_b + eps)", "normalized increment (target)"),
                ("r_t", r, "r = 1 - (sigma_a^2) / (sigma_b^2 + eps)", "variance-reduction fraction (target)"),
            ]:
                var = out.createVariable(out_name, "f4", xb_dims)
                var[:] = data
                var.setncattr("units", "K" if "increment" in out_name and "norm" not in out_name else "1")
                var.setncattr("equation", equation)
                var.setncattr("ml_role", role)
                var.setncattr("eps", float(eps))
                if out_name == "sigma_b_t":
                    var.setncattr("source_xb_filename", xb_path.name)
                    var.setncattr("source_xb_memb_filename", mb_path.name)
                    var.setncattr("source_xb_variable", xb_src.name)
                    var.setncattr("source_xb_memb_variable", mb_src.name)
                elif out_name == "increment_t":
                    var.setncattr("source_xa_filename", xa_path.name)
                    var.setncattr("source_xb_filename", xb_path.name)
                elif out_name in {"increment_norm_t", "r_t"}:
                    var.setncattr("derived_from", "xb_t, xa_t, sigma_b_t, sigma_a_t")

            for out_ctx, (product, native) in DA_CORE_T_CONTEXT.items():
                src_ds = ds_as if product == "anal_sprd" else ds_bm
                src_path = paths["anal_sprd"] if product == "anal_sprd" else paths["back_mean"]
                src = src_ds.variables[native]
                data = _read_slice(src, t_idx)
                dims = tuple(d for d in src.dimensions if d != "time")
                var = out.createVariable(f"context_{out_ctx}", "f4", dims)
                var[:] = data
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": product,
                        "source_filename": src_path.name,
                        "source_variable": native,
                        "ml_role": "Mars context s",
                    },
                )

            out.setncattr(
                "variable_manifest",
                json.dumps(
                    {
                        "inputs": ["xb_t", "sigma_b_t", "context_surface_geopotential", "context_dod"],
                        "targets": ["increment_norm_t", "sigma_a_t", "r_t"],
                        "supervision_qa": ["xa_t", "increment_t"],
                        "context_scalars": sorted(DA_CORE_T_TIME_META),
                        "coords": sorted(DA_CORE_T_STATIC),
                    }
                ),
            )
    finally:
        ds_bm.close()
        ds_am.close()
        ds_as.close()
        ds_mb.close()


def write_emars_ml_nc(
    out_path: Path,
    paths: dict[str, Path],
    t_idx: int,
    eps: float,
    y: int,
    m: int,
    d: int,
    hour: int,
) -> None:
    """Write one-hour EMARS ML sample with xb/xa/sigma and derived targets."""
    ds_bm = Dataset(paths["back_mean"])
    ds_am = Dataset(paths["anal_mean"])
    ds_as = Dataset(paths["anal_sprd"])
    ds_mb = Dataset(paths["back_memb"])

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(out_path, "w", format="NETCDF4") as out:
            # Union of dimensions across all EMARS products (anal has latu/lonv, etc.)
            dim_sizes: dict[str, int] = {}
            for src_ds in (ds_bm, ds_am, ds_as, ds_mb):
                for dim, size in src_ds.dimensions.items():
                    if dim == "time":
                        continue
                    dim_sizes[dim] = len(size)
            for dim, size in dim_sizes.items():
                out.createDimension(dim, size)

            out.setncattr("title", "EMARS ML-ready assimilation sample (single Earth hour)")
            out.setncattr("Conventions", "CF-1.8")
            out.setncattr(
                "earth_time",
                f"{y:04d}-{m:02d}-{d:02d}T{hour:02d}:00 (matched on EMARS earth_year/month/day/hour)",
            )
            out.setncattr("emars_time_index", int(t_idx))
            out.setncattr("eps", float(eps))
            out.setncattr(
                "sigma_b_definition",
                "Single-member prior-spread proxy: sigma_b = |back_memb - back_mean| "
                "(EMARS v1.0 ships one background member, not a full prior ensemble; "
                "see notebooks/eda_emars_reanalysis.ipynb)",
            )
            out.setncattr("created_utc", datetime.now(timezone.utc).isoformat())
            out.setncattr("source_back_mean", str(paths["back_mean"]))
            out.setncattr("source_anal_mean", str(paths["anal_mean"]))
            out.setncattr("source_anal_sprd", str(paths["anal_sprd"]))
            out.setncattr("source_back_memb", str(paths["back_memb"]))

            # scalar time/context metadata
            for name in sorted(TIME_META_VARS):
                if name not in ds_bm.variables:
                    continue
                src = ds_bm.variables[name]
                val = np.array(src[t_idx]).item() if src.ndim == 1 else np.array(src[t_idx])
                var = out.createVariable(name, "f8")
                var.assignValue(float(val) if np.ndim(val) == 0 else float(np.asarray(val).ravel()[0]))
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": "back_mean",
                        "source_filename": paths["back_mean"].name,
                        "source_variable": name,
                    },
                )

            # static coordinate / hybrid coeffs (prefer back_mean; fall back to anal for latu/lonv)
            for name in STATIC_1D:
                src_ds = None
                src_path = None
                src_product = None
                for label, ds_i, path_i in (
                    ("back_mean", ds_bm, paths["back_mean"]),
                    ("anal_mean", ds_am, paths["anal_mean"]),
                    ("anal_sprd", ds_as, paths["anal_sprd"]),
                    ("back_memb", ds_mb, paths["back_memb"]),
                ):
                    if name in ds_i.variables and "time" not in ds_i.variables[name].dimensions:
                        src_ds, src_path, src_product = ds_i, path_i, label
                        break
                if src_ds is None:
                    continue
                src = src_ds.variables[name]
                var = out.createVariable(name, src.dtype, src.dimensions)
                var[:] = src[:]
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": src_product,
                        "source_filename": src_path.name,
                        "source_variable": name,
                    },
                )

            # topography / height context if present at this time
            for topo_name in ("Surface_geopotential", "h"):
                src_ds = ds_as if topo_name in ds_as.variables else ds_bm if topo_name in ds_bm.variables else None
                src_path = paths["anal_sprd"] if src_ds is ds_as else paths["back_mean"]
                if src_ds is None:
                    continue
                src = src_ds.variables[topo_name]
                data = _read_slice(src, t_idx)
                # use surface level of h if 3D on phalf/pfull
                if data.ndim == 3:
                    # h on phalf: take near-surface (last index, positive-down hybrid often)
                    data = data[-1, ...]
                    dims = ("lat", "lon")
                elif data.ndim == 2:
                    dims = tuple(d for d in src.dimensions if d != "time")
                else:
                    continue
                var = out.createVariable(f"context_{_canonical(topo_name)}", "f4", dims)
                var[:] = data
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": "anal_sprd" if src_ds is ds_as else "back_mean",
                        "source_filename": src_path.name,
                        "source_variable": topo_name,
                        "role": "Mars context (topography / height)",
                    },
                )

            # dust context from background mean
            for dust_name in ("dod", "tod", "vod", "opac"):
                if dust_name not in ds_bm.variables:
                    continue
                src = ds_bm.variables[dust_name]
                data = _read_slice(src, t_idx)
                dims = tuple(d for d in src.dimensions if d != "time")
                var = out.createVariable(f"context_{dust_name}", "f4", dims)
                var[:] = data
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": "back_mean",
                        "source_filename": paths["back_mean"].name,
                        "source_variable": dust_name,
                        "role": "Mars context (dust / opacity)",
                    },
                )

            # local time context
            if "mars_hour" in ds_bm.variables:
                out.setncattr(
                    "context_local_time_note",
                    "mars_hour is Martian hour of day from EMARS (global time coordinate, not grid LT)",
                )

            def emit_state(prefix: str, src_ds: Dataset, src_path: Path, product: str) -> dict[str, str]:
                """Write all time-varying state fields; return map canonical->out_name."""
                mapping: dict[str, str] = {}
                for name, src in src_ds.variables.items():
                    if name in TIME_META_VARS or name in STATIC_1D:
                        continue
                    if "time" not in src.dimensions:
                        continue
                    data = _read_slice(src, t_idx)
                    dims = tuple(d for d in src.dimensions if d != "time")
                    out_name = f"{prefix}_{_canonical(name)}"
                    var = out.createVariable(out_name, "f4", dims)
                    var[:] = data
                    _copy_attrs(
                        src,
                        var,
                        extra={
                            "source_product": product,
                            "source_filename": src_path.name,
                            "source_variable": name,
                            "ml_role": {
                                "xb": "prior/background mean x^b",
                                "xa": "analysis/posterior mean x^a",
                                "sigma_a": "analysis ensemble spread sigma^a",
                                "xb_memb": "archived background ensemble member",
                            }.get(prefix, prefix),
                        },
                    )
                    mapping[_canonical(name)] = out_name
                return mapping

            map_xb = emit_state("xb", ds_bm, paths["back_mean"], "back_mean")
            map_xa = emit_state("xa", ds_am, paths["anal_mean"], "anal_mean")
            map_sa = emit_state("sigma_a", ds_as, paths["anal_sprd"], "anal_sprd")
            map_mb = emit_state("xb_memb", ds_mb, paths["back_memb"], "back_memb")

            # derived: sigma_b, increment, increment_norm, r_i for overlapping fields
            common = sorted(set(map_xb) & set(map_xa))
            derived = []
            for cname in common:
                xb = np.array(out.variables[map_xb[cname]][:], dtype=np.float64)
                xa = np.array(out.variables[map_xa[cname]][:], dtype=np.float64)
                dims = out.variables[map_xb[cname]].dimensions

                inc = xa - xb
                v_inc = out.createVariable(f"increment_{cname}", "f4", dims)
                v_inc[:] = inc
                v_inc.setncattr("long_name", f"analysis increment for {cname}")
                v_inc.setncattr("units", out.variables[map_xb[cname]].units if "units" in out.variables[map_xb[cname]].ncattrs() else "")
                v_inc.setncattr("equation", "increment = x^a - x^b")
                v_inc.setncattr("sources", f"{map_xa[cname]} - {map_xb[cname]}")
                v_inc.setncattr("source_xa_filename", paths["anal_mean"].name)
                v_inc.setncattr("source_xb_filename", paths["back_mean"].name)
                v_inc.setncattr("source_xa_variable", out.variables[map_xa[cname]].getncattr("source_variable"))
                v_inc.setncattr("source_xb_variable", out.variables[map_xb[cname]].getncattr("source_variable"))

                if cname in map_mb:
                    mb = np.array(out.variables[map_mb[cname]][:], dtype=np.float64)
                    sig_b = np.abs(mb - xb)
                    v_sb = out.createVariable(f"sigma_b_{cname}", "f4", dims)
                    v_sb[:] = sig_b
                    v_sb.setncattr("long_name", f"prior spread proxy for {cname}")
                    v_sb.setncattr(
                        "units",
                        out.variables[map_xb[cname]].units if "units" in out.variables[map_xb[cname]].ncattrs() else "",
                    )
                    v_sb.setncattr("equation", "sigma_b = |x^{b,memb} - x^b|")
                    v_sb.setncattr(
                        "note",
                        "Proxy from single archived background member; not full ensemble std",
                    )
                    v_sb.setncattr("source_xb_memb_filename", paths["back_memb"].name)
                    v_sb.setncattr("source_xb_filename", paths["back_mean"].name)

                    v_in = out.createVariable(f"increment_norm_{cname}", "f4", dims)
                    v_in[:] = inc / (sig_b + eps)
                    v_in.setncattr("long_name", f"normalized analysis increment for {cname}")
                    v_in.setncattr("units", "1")
                    v_in.setncattr("equation", "increment_norm = (x^a - x^b) / (sigma_b + eps)")
                    v_in.setncattr("eps", float(eps))

                    if cname in map_sa:
                        sig_a = np.array(out.variables[map_sa[cname]][:], dtype=np.float64)
                        # broadcast-safe if shapes match
                        if sig_a.shape == sig_b.shape:
                            r = 1.0 - (sig_a**2) / (sig_b**2 + eps)
                            v_r = out.createVariable(f"r_{cname}", "f4", dims)
                            v_r[:] = r
                            v_r.setncattr("long_name", f"variance reduction fraction for {cname}")
                            v_r.setncattr("units", "1")
                            v_r.setncattr("equation", "r = 1 - (sigma_a^2) / (sigma_b^2 + eps)")
                            v_r.setncattr("eps", float(eps))
                            v_r.setncattr("source_sigma_a_filename", paths["anal_sprd"].name)

                derived.append(cname)

            out.setncattr("derived_variables", ",".join(derived))
            out.setncattr(
                "variable_groups",
                json.dumps(
                    {
                        "xb": sorted(map_xb.values()),
                        "xa": sorted(map_xa.values()),
                        "sigma_a": sorted(map_sa.values()),
                        "xb_memb": sorted(map_mb.values()),
                    }
                ),
            )
    finally:
        ds_bm.close()
        ds_am.close()
        ds_as.close()
        ds_mb.close()


def write_emars_v3_nc(
    out_path: Path,
    paths: dict[str, Path],
    t_idx: int,
    eps: float,
    y: int,
    m: int,
    d: int,
    hour: int,
) -> None:
    """v3: xb_t, xa_t, increment_t, increment_norm_t = (xa-xb)/(xb+eps). No spread/MCS."""
    ds_bm = Dataset(paths["back_mean"])
    ds_am = Dataset(paths["anal_mean"])
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(out_path, "w", format="NETCDF4") as out:
            for dim, size in ds_bm.dimensions.items():
                if dim == "time":
                    continue
                out.createDimension(dim, len(size))

            out.setncattr("title", "EMARS data-prep v3 (T: background, analysis, increment)")
            out.setncattr("Conventions", "CF-1.8")
            out.setncattr("profile", "v3")
            out.setncattr(
                "earth_time",
                f"{y:04d}-{m:02d}-{d:02d}T{hour:02d}:00 (matched on EMARS earth_year/month/day/hour)",
            )
            out.setncattr("emars_time_index", int(t_idx))
            out.setncattr("eps", float(eps))
            out.setncattr(
                "normalization",
                "increment_norm_t = (xa_t - xb_t) / (xb_t + eps)  [scaled by background mean]",
            )
            out.setncattr("created_utc", datetime.now(timezone.utc).isoformat())
            out.setncattr("source_back_mean", str(paths["back_mean"]))
            out.setncattr("source_anal_mean", str(paths["anal_mean"]))

            for name in ("lat", "lon", "pfull", "phalf", "ak", "bk"):
                if name not in ds_bm.variables:
                    continue
                src = ds_bm.variables[name]
                if "time" in src.dimensions:
                    continue
                var = out.createVariable(name, src.dtype, src.dimensions)
                var[:] = src[:]
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": "back_mean",
                        "source_filename": paths["back_mean"].name,
                        "source_variable": name,
                    },
                )

            for name in ("Ls", "MY", "mars_hour", "earth_year", "earth_month", "earth_day", "earth_hour"):
                if name not in ds_bm.variables:
                    continue
                src = ds_bm.variables[name]
                val = float(np.array(src[t_idx]).item())
                var = out.createVariable(name, "f8")
                var.assignValue(val)
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": "back_mean",
                        "source_filename": paths["back_mean"].name,
                        "source_variable": name,
                    },
                )

            t_bm = ds_bm.variables["t"]
            t_am = ds_am.variables["T"]
            xb = np.array(_read_slice(t_bm, t_idx), dtype=np.float64)
            xa = np.array(_read_slice(t_am, t_idx), dtype=np.float64)
            dims = tuple(d for d in t_bm.dimensions if d != "time")

            for out_name, data, src, path, product, role in [
                ("xb_t", xb, t_bm, paths["back_mean"], "back_mean", "background / prior mean T"),
                ("xa_t", xa, t_am, paths["anal_mean"], "anal_mean", "analysis / posterior mean T"),
            ]:
                var = out.createVariable(out_name, "f4", dims)
                var[:] = data
                _copy_attrs(
                    src,
                    var,
                    extra={
                        "source_product": product,
                        "source_filename": path.name,
                        "source_variable": src.name,
                        "ml_role": role,
                    },
                )

            inc = xa - xb
            inc_norm = inc / (xb + eps)

            v_inc = out.createVariable("increment_t", "f4", dims)
            v_inc[:] = inc
            v_inc.setncattr("long_name", "analysis increment for temperature")
            v_inc.setncattr("units", "K")
            v_inc.setncattr("equation", "increment_t = xa_t - xb_t")
            v_inc.setncattr("ml_role", "analysis increment ΔT")

            v_n = out.createVariable("increment_norm_t", "f4", dims)
            v_n[:] = inc_norm
            v_n.setncattr("long_name", "increment normalized by background mean")
            v_n.setncattr("units", "1")
            v_n.setncattr("equation", "increment_norm_t = (xa_t - xb_t) / (xb_t + eps)")
            v_n.setncattr("eps", float(eps))
            v_n.setncattr("ml_role", "normalized increment (scaled by background mean)")

            out.setncattr(
                "variable_manifest",
                json.dumps(
                    {
                        "fields": ["xb_t", "xa_t", "increment_t", "increment_norm_t"],
                        "coords": ["lat", "lon", "pfull", "phalf", "ak", "bk"],
                        "time_meta": [
                            "Ls",
                            "MY",
                            "mars_hour",
                            "earth_year",
                            "earth_month",
                            "earth_day",
                            "earth_hour",
                        ],
                    }
                ),
            )
    finally:
        ds_bm.close()
        ds_am.close()


def prepare_pairs(
    mars_year: str,
    ls_bin: str,
    hours: list[int],
    earth_date: str | None,
    emars_root: Path,
    mcs_root: Path,
    out_dir: Path,
    eps: float,
    profile: str = "full",
) -> list[dict[str, str]]:
    needed = ["back_mean", "anal_mean"]
    if profile != "v3":
        needed += ["anal_sprd", "back_memb"]
    paths: dict[str, Path] = {}
    for k in needed:
        paths[k] = _product_path(emars_root, k, mars_year, ls_bin)
        if not paths[k].exists():
            raise FileNotFoundError(f"Required {k} not found: {paths[k]}")

    table = load_emars_time_table(paths["back_mean"])
    if earth_date:
        y, m, d = [int(x) for x in earth_date.split("-")]
        day = (y, m, d)
    elif profile == "v3":
        days = list_days_with_hours(table, hours)
        if not days:
            raise RuntimeError("No Earth day found with all requested EMARS hours for v3.")
        day = days[0]
        print(f"Auto-selected earth date {day[0]:04d}-{day[1]:02d}-{day[2]:02d}")
    else:
        day = auto_select_earth_date(table, mcs_root, hours)
        print(f"Auto-selected earth date {day[0]:04d}-{day[1]:02d}-{day[2]:02d}")

    y, m, d = day
    results = []
    for hour in hours:
        t_idx = find_time_index(table, y, m, d, hour)
        stamp = f"{y:04d}{m:02d}{d:02d}{hour:02d}"
        if profile == "da_core_t":
            emars_out = out_dir / f"emars_da_{mars_year}_{stamp}.nc"
            mcs_out = out_dir / f"mcs_da_{mars_year}_{stamp}.nc"
        elif profile == "v3":
            emars_out = out_dir / f"emars_v3_{mars_year}_{stamp}.nc"
            mcs_out = None
        else:
            emars_out = out_dir / f"emars_ml_{mars_year}_{stamp}.nc"
            mcs_out = out_dir / f"mcs_T_{mars_year}_{stamp}.nc"

        print(f"[hour {hour:02d}] EMARS time index {t_idx} -> {emars_out.name}")
        if profile == "da_core_t":
            write_emars_da_core_t_nc(emars_out, paths, t_idx, eps, y, m, d, hour)
        elif profile == "v3":
            write_emars_v3_nc(emars_out, paths, t_idx, eps, y, m, d, hour)
            results.append({"hour": f"{hour:02d}", "emars": str(emars_out)})
            continue
        else:
            write_emars_ml_nc(emars_out, paths, t_idx, eps, y, m, d, hour)

        ddr_files = find_mcs_ddr_for_hour(mcs_root, y, m, d, hour)
        if not ddr_files:
            raise FileNotFoundError(f"No MCS DDR for {stamp}")
        print(f"[hour {hour:02d}] MCS from {[p.name for p in ddr_files]} -> {mcs_out.name}")
        bundle = read_mcs_limb_t_for_hour(ddr_files, hour)
        write_mcs_nc(mcs_out, bundle, y, m, d, hour, ddr_files, profile=profile)
        print(
            f"[hour {hour:02d}] MCS profiles={len(bundle['profiles'])} "
            f"qc_pass={sum(1 for p in bundle['profiles'] if p['qc_pass'])}"
        )
        results.append({"hour": f"{hour:02d}", "emars": str(emars_out), "mcs": str(mcs_out)})
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mars-year", default="MY31")
    p.add_argument("--ls-bin", default="Ls210-240", help="EMARS Ls package, e.g. Ls210-240")
    p.add_argument("--hours", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--earth-date", default=None, help="YYYY-MM-DD; auto-selected if omitted")
    p.add_argument("--emars-root", type=Path, default=DEFAULT_EMARS_ROOT)
    p.add_argument("--mcs-root", type=Path, default=DEFAULT_MCS_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--eps", type=float, default=EPS_DEFAULT)
    p.add_argument(
        "--profile",
        choices=("full", "da_core_t", "v3"),
        default="full",
        help="full | da_core_t | v3 (xb, xa, Δx, Δx/xb)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if args.profile == "da_core_t" and args.out_dir == DEFAULT_OUT:
        out_dir = DEFAULT_OUT / "da_core_t"
    elif args.profile == "v3" and args.out_dir == DEFAULT_OUT:
        out_dir = DEFAULT_OUT / "v3"

    results = prepare_pairs(
        mars_year=args.mars_year,
        ls_bin=args.ls_bin,
        hours=args.hours,
        earth_date=args.earth_date,
        emars_root=args.emars_root,
        mcs_root=args.mcs_root,
        out_dir=out_dir,
        eps=args.eps,
        profile=args.profile,
    )
    print(json.dumps({"pairs": results}, indent=2))


if __name__ == "__main__":
    main()
