"""Prepare hourly EMARS–MCS DA NetCDF products (input, output, MCS limbs).

Builds distance-weighted temperature superobservations on the EMARS grid
(vertical match to local hybrid levels from ak, bk, ps) and writes one-hour
triples under ``EMARS_MCS_training_data/EMARS_MCS_DA_YYYYMMDD/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from . import pairs as v1
from .config import EMARS_ROOT, OBS_ROOT, TRAINING_DATA_ROOT, PKG_ROOT

HERE = PKG_ROOT
REPO = PKG_ROOT.parent
DEFAULT_EMARS_ROOT = EMARS_ROOT
DEFAULT_MCS_ROOT = OBS_ROOT
# Default out dir: TRAINING_DATA_ROOT / f"EMARS_MCS_DA_{yyyymmdd}"

EPS_DEFAULT = 1.0e-6
SIGMA_O_FLOOR_K = 3.0
QC_GQUAL_OK = {0}
QC_PQUAL_OK = {0}
QC_TQUAL_OK = {0}  # Good only (Steve / MCS t_qual=0)


def lon360(x):
    return np.mod(np.asarray(x, dtype=float), 360.0)


# Mars mean radius for horizontal distance weights (km).
R_MARS_KM = 3389.5
# Floor so an obs exactly on the grid center does not get infinite weight.
DIST_EPS_KM = 1.0
# Floor on |ln(p_obs) - ln(p_hybrid)| for vertical inverse-distance weight.
DIST_EPS_LNP = 0.05


def horiz_dist_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Equirectangular distance (km) between two lon/lat points on Mars."""
    lat0r = np.deg2rad(lat0)
    lat1r = np.deg2rad(lat1)
    dlat = lat1r - lat0r
    dlon = np.deg2rad(lon360(lon1) - lon360(lon0))
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi
    x = np.cos(0.5 * (lat0r + lat1r)) * dlon
    y = dlat
    return float(R_MARS_KM * np.sqrt(x * x + y * y))


def qc_pass_v2(profile: dict[str, Any]) -> bool:
    gq, pq, tq = profile["gqual"], profile["p_qual"], profile["t_qual"]
    try:
        return (
            (not np.isnan(gq) and int(gq) in QC_GQUAL_OK)
            and (not np.isnan(pq) and int(pq) in QC_PQUAL_OK)
            and (not np.isnan(tq) and int(tq) in QC_TQUAL_OK)
        )
    except (TypeError, ValueError):
        return False


def apply_v2_qc(bundle: dict[str, Any]) -> None:
    for p in bundle["profiles"]:
        p["qc_pass"] = qc_pass_v2(p)


def _copy_attrs(src_var, dst_var, extra: dict[str, str] | None = None) -> None:
    v1._copy_attrs(src_var, dst_var, extra)


GRID_COORDS = ("lat", "lon", "pfull", "phalf", "ak", "bk")
TIME_COORDS = ("Ls", "MY", "mars_hour", "earth_year", "earth_month", "earth_day", "earth_hour")
ALL_COORDS = GRID_COORDS + TIME_COORDS


def _write_emars_coords(out: Dataset, ds_bm: Dataset, paths: dict[str, Path], t_idx: int) -> None:
    """Write EMARS grid + time fields as CF auxiliary / scalar coordinates."""
    for name in GRID_COORDS:
        if name in out.variables:
            continue
        if name in ("lat", "lon", "pfull", "phalf") and name not in out.dimensions:
            out.createDimension(name, len(ds_bm.dimensions[name]))
        src = ds_bm.variables[name]
        var = out.createVariable(name, src.dtype, src.dimensions)
        var[:] = src[:]
        _copy_attrs(
            src,
            var,
            extra={
                "source_product": "back_mean",
                "source_filename": paths["back_mean"].name,
                "ml_role": "grid coordinate",
            },
        )

    for name in TIME_COORDS:
        if name in out.variables:
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
                "ml_role": "Mars / Earth time coordinate",
            },
        )


def _mark_cf_coordinates(out: Dataset, coord_names: tuple[str, ...] = ALL_COORDS) -> None:
    """List auxiliary coords on data vars so xarray opens them under Coordinates."""
    present = [n for n in coord_names if n in out.variables]
    if not present:
        return
    coord_str = " ".join(present)
    coord_set = set(present)
    for name, var in out.variables.items():
        if name in coord_set:
            continue
        if var.dimensions:  # skip pure scalars that are not coords
            var.setncattr("coordinates", coord_str)


def patch_mcs_limb_metadata(
    path: Path,
    n_qc: int,
    paths: dict[str, Path],
    t_idx: int,
) -> None:
    rule = (
        "qc_pass true iff Gqual in {0} and P_qual in {0} and T_qual in {0}; "
        "profiles are stored as-is (not filtered)"
    )
    ds_bm = Dataset(paths["back_mean"])
    try:
        with Dataset(path, "a") as ds:
            ds.setncattr("qc_rule", rule)
            ds.setncattr("n_profiles_qc_pass", int(n_qc))
            ds.setncattr("profile", "v3_limb")
            ds.setncattr("emars_time_index", int(t_idx))
            ds.setncattr("source_back_mean", str(paths["back_mean"]))
            qc = ds.variables["qc_pass"]
            qc.setncattr(
                "equation",
                "qc_pass = 1 if (Gqual in {0}) and (P_qual in {0}) and (T_qual in {0}) else 0",
            )
            qc.setncattr(
                "note",
                "Data arrays are not filtered by this mask. Super-obs use qc_pass=1 only.",
            )
            _write_emars_coords(ds, ds_bm, paths, t_idx)
            _mark_cf_coordinates(ds)
            ds.setncattr(
                "variable_manifest",
                json.dumps({"coords": list(ALL_COORDS), "profile": "v3_limb"}),
            )
    finally:
        ds_bm.close()


def hybrid_pfull_pa(ak_pa, bk, ps_pa):
    """Local full-level pressure (Pa) from half-level ak, bk and surface ps (Pa).

    p_half(k) = ak(k) + bk(k) * ps
    p_full(k) = 0.5 * (p_half(k) + p_half(k+1))
    """
    ak_pa = np.asarray(ak_pa, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    ps_pa = np.asarray(ps_pa, dtype=np.float64)
    if ps_pa.ndim == 0:
        phalf = ak_pa + bk * float(ps_pa)
        return 0.5 * (phalf[:-1] + phalf[1:])
    phalf = ak_pa[:, None, None] + bk[:, None, None] * ps_pa[None, :, :]
    return 0.5 * (phalf[:-1] + phalf[1:])


def superob_on_emars_grid(bundle: dict[str, Any], lat, lon, ak_pa, bk, ps_pa, xb):
    """Distance-weighted super-obs on EMARS hybrid levels (lat, lon, k).

    Each QC-pass MCS level is assigned to the nearest EMARS (lat, lon) cell, then to
    the nearest *local* hybrid full level at that column:
      p_full(k,j,i) = 0.5 * [ak(k)+bk(k)*ps(j,i) + ak(k+1)+bk(k+1)*ps(j,i)]
    Inverse-distance weights:
      w = 1/(r_h + ε_h) * 1/(|ln p_obs - ln p_full(k,j,i)| + ε_v)
    σ_o uses the same weights on Temp_err (not reduced by N), then floor 3 K.

    Also bins MCS T_qual onto the same grid for all profiles (not only QC-pass).
    """
    nlev, nlat, nlon = xb.shape
    # Local hybrid full-level pressure at every column for this hour (Pa).
    p_hyb_pa = hybrid_pfull_pa(ak_pa, bk, ps_pa)
    if p_hyb_pa.shape != (nlev, nlat, nlon):
        raise ValueError(
            f"hybrid pfull shape {p_hyb_pa.shape} != xb shape {(nlev, nlat, nlon)}"
        )
    sum_wt = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    sum_we = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    sum_w = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    count = np.zeros((nlev, nlat, nlon), dtype=np.int32)
    # Temperature QC flag binning (all profiles with finite T_qual)
    n_tq = np.zeros((nlev, nlat, nlon), dtype=np.int32)
    n_tq_good = np.zeros((nlev, nlat, nlon), dtype=np.int32)

    for prof in bundle["profiles"]:
        plat, plon = prof["profile_lat"], prof["profile_lon"]
        if not (np.isfinite(plat) and np.isfinite(plon)):
            continue
        t = np.asarray(prof["temperature_k"], dtype=np.float64)
        e = np.asarray(prof["temperature_err_k"], dtype=np.float64)
        p = np.asarray(prof["pressure_pa"], dtype=np.float64)
        lev_lat = np.asarray(prof.get("level_lat", plat), dtype=np.float64)
        lev_lon = np.asarray(prof.get("level_lon", plon), dtype=np.float64)
        if lev_lat.shape != t.shape:
            lev_lat = np.full(t.shape, plat, dtype=np.float64)
        if lev_lon.shape != t.shape:
            lev_lon = np.full(t.shape, plon, dtype=np.float64)

        tq = prof.get("t_qual", np.nan)
        try:
            tq_ok = (not np.isnan(tq)) and int(tq) in QC_TQUAL_OK
            tq_finite = not np.isnan(tq)
        except (TypeError, ValueError):
            tq_ok = False
            tq_finite = False

        ok_lev = np.isfinite(t) & np.isfinite(e) & (e > 0) & np.isfinite(p) & (p > 0)
        if not np.any(ok_lev):
            continue

        idxs = np.where(ok_lev)[0]
        for idx in idxs:
            olat = lev_lat[idx] if np.isfinite(lev_lat[idx]) else plat
            olon = lev_lon[idx] if np.isfinite(lev_lon[idx]) else plon
            if not (np.isfinite(olat) and np.isfinite(olon)):
                continue
            j = int(np.argmin(np.abs(lat - olat)))
            i = int(np.argmin(np.abs(lon - lon360(olon))))
            # Vertical: nearest local hybrid full level at this column
            p_col = p_hyb_pa[:, j, i]
            k_idx = int(np.argmin(np.abs(p_col - p[idx])))

            # Gridded T_qual from every profile that has a usable T level here
            if tq_finite:
                n_tq[k_idx, j, i] += 1
                if tq_ok:
                    n_tq_good[k_idx, j, i] += 1

            if not prof["qc_pass"]:
                continue

            r_h = horiz_dist_km(float(lat[j]), float(lon[i]), float(olat), float(olon))
            p_lev = float(p_col[k_idx])
            if not (np.isfinite(p_lev) and p_lev > 0):
                continue
            r_v = abs(np.log(p[idx]) - np.log(p_lev))
            w = (1.0 / (r_h + DIST_EPS_KM)) * (1.0 / (r_v + DIST_EPS_LNP))
            sum_wt[k_idx, j, i] += w * t[idx]
            sum_we[k_idx, j, i] += w * e[idx]
            sum_w[k_idx, j, i] += w
            count[k_idx, j, i] += 1

    mask = sum_w > 0
    y = np.full((nlev, nlat, nlon), np.nan, dtype=np.float32)
    sigma = np.full((nlev, nlat, nlon), np.nan, dtype=np.float32)
    y[mask] = (sum_wt[mask] / sum_w[mask]).astype(np.float32)
    sigma[mask] = (sum_we[mask] / sum_w[mask]).astype(np.float32)
    sigma[mask] = np.maximum(sigma[mask], np.float32(SIGMA_O_FLOOR_K))
    d = np.full((nlev, nlat, nlon), np.nan, dtype=np.float32)
    d[mask] = y[mask] - xb[mask]
    m_o = mask.astype(np.int8)

    # t_qual_flag: 1 where every MCS T sample mapped here has T_qual in {0}; else 0; -1 if none
    t_qual_flag = np.full((nlev, nlat, nlon), -1, dtype=np.int8)
    has_tq = n_tq > 0
    t_qual_flag[has_tq] = np.where(n_tq_good[has_tq] == n_tq[has_tq], 1, 0).astype(np.int8)
    return y, sigma, d, m_o, count, t_qual_flag


def write_input_nc(
    out_path: Path,
    paths: dict[str, Path],
    t_idx: int,
    y: int,
    m: int,
    d: int,
    hour: int,
    y_super,
    sigma_o,
    d_innov,
    m_o,
    n_raw,
    t_qual_flag,
) -> None:
    ds_bm = Dataset(paths["back_mean"])
    ds_as = Dataset(paths["anal_sprd"])
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(out_path, "w", format="NETCDF4") as out:
            for dim in ("pfull", "phalf", "lat", "lon"):
                out.createDimension(dim, len(ds_bm.dimensions[dim]))

            out.setncattr("title", "EMARS–MCS DA v3 inputs (one Earth hour, with MCS super-obs)")
            out.setncattr("Conventions", "CF-1.8")
            out.setncattr("profile", "v3_input")
            out.setncattr(
                "earth_time",
                f"{y:04d}-{m:02d}-{d:02d}T{hour:02d}:00 (matched on EMARS earth_year/month/day/hour)",
            )
            out.setncattr("emars_time_index", int(t_idx))
            out.setncattr("created_utc", datetime.now(timezone.utc).isoformat())
            out.setncattr("source_back_mean", str(paths["back_mean"]))
            out.setncattr("source_anal_sprd", str(paths["anal_sprd"]))
            out.setncattr(
                "operator",
                "(increment_norm, ...) = f(x^b, y, d, sigma_o, M_o, s); sigma_b is not included",
            )
            out.setncattr(
                "innovation",
                "d = y - H(x^b) with H = identity on the EMARS grid after superobbing "
                "(nearest cell, nearest local hybrid full level)",
            )
            out.setncattr(
                "superob",
                "y = inverse-distance-weighted mean of QC-pass MCS T assigned to nearest "
                "(lat, lon) and nearest local hybrid full level "
                "p_full(k,j,i)=0.5*[(ak+bk*ps)_k + (ak+bk*ps)_{k+1}]; "
                "w = 1/(r_h_km+1) * 1/(|ln p - ln p_full|+0.05); "
                "sigma_o = same weights on Temp_err (not reduced by N), then max(sigma_o, 3 K)",
            )
            out.setncattr(
                "qc_rule",
                "super-obs use Gqual in {0} and P_qual in {0} and T_qual in {0} (Good only)",
            )
            out.setncattr("sigma_o_floor_K", float(SIGMA_O_FLOOR_K))

            _write_emars_coords(out, ds_bm, paths, t_idx)

            xb_src = ds_bm.variables["t"]
            xb = np.array(v1._read_slice(xb_src, t_idx), dtype=np.float32)
            dims = ("pfull", "lat", "lon")
            v_xb = out.createVariable("xb_t", "f4", dims)
            v_xb[:] = xb
            _copy_attrs(
                xb_src,
                v_xb,
                extra={
                    "source_product": "back_mean",
                    "source_filename": paths["back_mean"].name,
                    "source_variable": "t",
                    "ml_role": "input x^b prior/background mean",
                },
            )

            v_y = out.createVariable("y_super_t", "f4", dims, fill_value=np.nan)
            v_y[:] = y_super
            v_y.setncattr("long_name", "MCS temperature super-observation")
            v_y.setncattr("units", "K")
            v_y.setncattr(
                "equation",
                "y = sum(w*T)/sum(w); w = 1/(r_h+1km) * 1/(|ln p - ln p_hybrid|+0.05); "
                "nearest EMARS (lat, lon) then nearest local hybrid full level "
                "p_hybrid=0.5*((ak+bk*ps)_k+(ak+bk*ps)_{k+1}); "
                "footprint from level_lat/lon when available",
            )
            v_y.setncattr("ml_role", "input y (super-obs to assimilate)")

            v_d = out.createVariable("d_t", "f4", dims, fill_value=np.nan)
            v_d[:] = d_innov
            v_d.setncattr("long_name", "innovation (observation minus background)")
            v_d.setncattr("units", "K")
            v_d.setncattr("equation", "d = y_super_t - H(xb_t)  with H = identity on EMARS grid")
            v_d.setncattr("ml_role", "input d = y - H(x^b)")

            v_s = out.createVariable("sigma_o_t", "f4", dims, fill_value=np.nan)
            v_s[:] = sigma_o
            v_s.setncattr("long_name", "super-ob observation error")
            v_s.setncattr("units", "K")
            v_s.setncattr(
                "equation",
                "sigma_o = max(sum(w*Temp_err)/sum(w), 3 K) with same inverse-distance "
                "weights as y; error is not reduced by N "
                "(representativeness not in retrieval error; Steve Greybush)",
            )
            v_s.setncattr("ml_role", "input σ_o")
            v_s.setncattr("floor_K", float(SIGMA_O_FLOOR_K))

            v_m = out.createVariable("M_o", "i1", dims)
            v_m[:] = m_o
            v_m.setncattr("long_name", "observation mask")
            v_m.setncattr("equation", "M_o = 1 if at least one QC-pass MCS hit in this cell/level else 0")
            v_m.setncattr("ml_role", "input M_o")

            v_tq = out.createVariable("t_qual_flag", "i1", dims, fill_value=np.int8(-1))
            v_tq[:] = t_qual_flag
            v_tq.setncattr("long_name", "gridded MCS temperature quality flag")
            v_tq.setncattr(
                "flag_values",
                "1 = all MCS T samples in this cell have T_qual in {0} (Good); "
                "0 = at least one sample with T_qual not Good; "
                "-1 = no MCS T samples mapped here",
            )
            v_tq.setncattr("source", "MCS DDR T_qual, binned to EMARS (pfull, lat, lon)")
            v_tq.setncattr("ml_role", "input temperature QC flag (gridded)")

            v_n = out.createVariable("n_raw", "i4", dims)
            v_n[:] = n_raw
            v_n.setncattr("long_name", "number of raw MCS T points in the super-ob bin")
            v_n.setncattr("ml_role", "diagnostic (not an operator input)")

            zg_src = ds_as.variables["Surface_geopotential"]
            v_zg = out.createVariable("context_surface_geopotential", "f4", ("lat", "lon"))
            v_zg[:] = np.array(zg_src[:], dtype=np.float32)
            _copy_attrs(
                zg_src,
                v_zg,
                extra={
                    "source_product": "anal_sprd",
                    "source_filename": paths["anal_sprd"].name,
                    "source_variable": "Surface_geopotential",
                    "ml_role": "Mars context s",
                },
            )

            dod_src = ds_bm.variables["dod"]
            dod = np.array(v1._read_slice(dod_src, t_idx), dtype=np.float32)
            v_dod = out.createVariable("context_dod", "f4", ("lat", "lon"))
            v_dod[:] = dod
            _copy_attrs(
                dod_src,
                v_dod,
                extra={
                    "source_product": "back_mean",
                    "source_filename": paths["back_mean"].name,
                    "source_variable": "dod",
                    "ml_role": "Mars context s",
                },
            )

            _mark_cf_coordinates(out)
            out.setncattr(
                "variable_manifest",
                json.dumps(
                    {
                        "inputs": [
                            "xb_t",
                            "y_super_t",
                            "d_t",
                            "sigma_o_t",
                            "M_o",
                            "t_qual_flag",
                            "context_surface_geopotential",
                            "context_dod",
                        ],
                        "coords": list(ALL_COORDS),
                        "diagnostic": ["n_raw"],
                        "not_included": ["sigma_b"],
                    }
                ),
            )
    finally:
        ds_bm.close()
        ds_as.close()


def write_output_nc(
    out_path: Path,
    paths: dict[str, Path],
    t_idx: int,
    eps: float,
    y: int,
    m: int,
    d: int,
    hour: int,
) -> None:
    """v3 outputs: increments + analysis temperature spread from anal_sprd."""
    ds_bm = Dataset(paths["back_mean"])
    ds_am = Dataset(paths["anal_mean"])
    ds_as = Dataset(paths["anal_sprd"])
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(out_path, "w", format="NETCDF4") as out:
            for dim in ("pfull", "phalf", "lat", "lon"):
                out.createDimension(dim, len(ds_bm.dimensions[dim]))

            out.setncattr("title", "EMARS–MCS DA v3 outputs (increments + analysis spread)")
            out.setncattr("Conventions", "CF-1.8")
            out.setncattr("profile", "v3_output")
            out.setncattr(
                "earth_time",
                f"{y:04d}-{m:02d}-{d:02d}T{hour:02d}:00 (matched on EMARS earth_year/month/day/hour)",
            )
            out.setncattr("emars_time_index", int(t_idx))
            out.setncattr("eps", float(eps))
            out.setncattr(
                "normalization",
                "increment_norm_t = (xa_t - xb_t) / (xb_t + eps)  [same as v3; scaled by background mean]",
            )
            out.setncattr("created_utc", datetime.now(timezone.utc).isoformat())
            out.setncattr("source_back_mean", str(paths["back_mean"]))
            out.setncattr("source_anal_mean", str(paths["anal_mean"]))
            out.setncattr("source_anal_sprd", str(paths["anal_sprd"]))

            _write_emars_coords(out, ds_bm, paths, t_idx)

            xb = np.array(v1._read_slice(ds_bm.variables["t"], t_idx), dtype=np.float64)
            xa = np.array(v1._read_slice(ds_am.variables["T"], t_idx), dtype=np.float64)
            sigma_a = np.array(v1._read_slice(ds_as.variables["T"], t_idx), dtype=np.float32)
            inc = xa - xb
            inc_norm = inc / (xb + eps)
            dims = ("pfull", "lat", "lon")

            v_inc = out.createVariable("increment_t", "f4", dims)
            v_inc[:] = inc
            v_inc.setncattr("long_name", "analysis increment for temperature")
            v_inc.setncattr("units", "K")
            v_inc.setncattr("equation", "increment_t = xa_t - xb_t")
            v_inc.setncattr("ml_role", "output ΔT")
            v_inc.setncattr("source_xa_filename", paths["anal_mean"].name)
            v_inc.setncattr("source_xb_filename", paths["back_mean"].name)
            v_inc.setncattr("source_xa_variable", "T")
            v_inc.setncattr("source_xb_variable", "t")

            v_n = out.createVariable("increment_norm_t", "f4", dims)
            v_n[:] = inc_norm
            v_n.setncattr("long_name", "increment normalized by background mean")
            v_n.setncattr("units", "1")
            v_n.setncattr("equation", "increment_norm_t = (xa_t - xb_t) / (xb_t + eps)")
            v_n.setncattr("eps", float(eps))
            v_n.setncattr("ml_role", "output normalized increment (v3 formula)")

            v_sa = out.createVariable("sigma_a_t", "f4", dims)
            v_sa[:] = sigma_a
            _copy_attrs(
                ds_as.variables["T"],
                v_sa,
                extra={
                    "long_name": "analysis ensemble temperature spread",
                    "source_product": "anal_sprd",
                    "source_filename": paths["anal_sprd"].name,
                    "source_variable": "T",
                    "ml_role": "output σ_a (analysis spread)",
                },
            )

            _mark_cf_coordinates(out)
            out.setncattr(
                "variable_manifest",
                json.dumps(
                    {
                        "outputs": ["increment_t", "increment_norm_t", "sigma_a_t"],
                        "coords": list(ALL_COORDS),
                    }
                ),
            )
    finally:
        ds_bm.close()
        ds_am.close()
        ds_as.close()


def prepare_da_hourly(
    mars_year: str,
    ls_bin: str,
    hours: list[int],
    earth_date: str,
    emars_root: Path,
    mcs_root: Path,
    out_dir: Path,
    eps: float,
) -> list[dict[str, Any]]:
    paths = {
        "back_mean": v1._product_path(emars_root, "back_mean", mars_year, ls_bin),
        "anal_mean": v1._product_path(emars_root, "anal_mean", mars_year, ls_bin),
        "anal_sprd": v1._product_path(emars_root, "anal_sprd", mars_year, ls_bin),
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Required {k} not found: {p}")

    table = v1.load_emars_time_table(paths["back_mean"])
    y, m, d = [int(x) for x in earth_date.split("-")]
    ds_bm = Dataset(paths["back_mean"])
    lat = np.array(ds_bm.variables["lat"][:], dtype=np.float64)
    lon = np.array(ds_bm.variables["lon"][:], dtype=np.float64)
    ak_pa = np.array(ds_bm.variables["ak"][:], dtype=np.float64)
    bk = np.array(ds_bm.variables["bk"][:], dtype=np.float64)
    ds_bm.close()

    results = []
    skipped: list[dict[str, str]] = []
    input_dir = out_dir / "input"
    output_dir = out_dir / "output"
    limb_dir = out_dir / "mcs_limbs"
    for dpath in (input_dir, output_dir, limb_dir):
        dpath.mkdir(parents=True, exist_ok=True)

    available = set(v1.emars_hours_on_day(table, y, m, d))
    print(f"[prep] EMARS hours on {earth_date}: {sorted(available)}")
    print("[prep] vertical match: local hybrid p_full(k,j,i) from ak, bk, ps")

    for hour in hours:
        stamp = f"{y:04d}{m:02d}{d:02d}{hour:02d}"
        input_path = input_dir / f"input_{mars_year}_{stamp}.nc"
        output_path = output_dir / f"output_{mars_year}_{stamp}.nc"
        limb_path = limb_dir / f"mcs_limb_{mars_year}_{stamp}.nc"

        if hour not in available:
            print(f"[hour {hour:02d}] skip: no EMARS timestep")
            skipped.append({"hour": f"{hour:02d}", "reason": "no_emars"})
            continue

        if (
            input_path.exists()
            and input_path.stat().st_size > 0
            and output_path.exists()
            and output_path.stat().st_size > 0
            and limb_path.exists()
            and limb_path.stat().st_size > 0
        ):
            print(f"[hour {hour:02d}] skip: outputs exist")
            results.append(
                {
                    "hour": f"{hour:02d}",
                    "input": str(input_path),
                    "output": str(output_path),
                    "mcs_limb": str(limb_path),
                    "status": "skipped_exists",
                }
            )
            continue

        t_idx = v1.find_time_index(table, y, m, d, hour)
        ddr_files = v1.find_mcs_ddr_for_hour(mcs_root, y, m, d, hour)
        if not ddr_files:
            print(f"[hour {hour:02d}] skip: no MCS DDR")
            skipped.append({"hour": f"{hour:02d}", "reason": "no_mcs"})
            continue

        print(f"[hour {hour:02d}] EMARS t_idx={t_idx}  MCS {[p.name for p in ddr_files]}")
        bundle = v1.read_mcs_limb_t_for_hour(ddr_files, hour)
        apply_v2_qc(bundle)
        n_qc = int(sum(1 for p in bundle["profiles"] if p["qc_pass"]))
        print(f"[hour {hour:02d}] MCS profiles={len(bundle['profiles'])}  qc_pass(T_qual=0)={n_qc}")

        ds_bm = Dataset(paths["back_mean"])
        xb = np.array(v1._read_slice(ds_bm.variables["t"], t_idx), dtype=np.float32)
        ps_pa = np.array(v1._read_slice(ds_bm.variables["ps"], t_idx), dtype=np.float64)
        ds_bm.close()
        y_s, sig, d_inn, m_o, n_raw, t_qual_flag = superob_on_emars_grid(
            bundle, lat, lon, ak_pa, bk, ps_pa, xb
        )
        print(
            f"[hour {hour:02d}] super-obs filled cells={int(m_o.sum())}  "
            f"max N/bin={int(n_raw.max())}"
        )

        write_input_nc(
            input_path, paths, t_idx, y, m, d, hour, y_s, sig, d_inn, m_o, n_raw, t_qual_flag
        )
        write_output_nc(output_path, paths, t_idx, eps, y, m, d, hour)
        v1.write_mcs_nc(limb_path, bundle, y, m, d, hour, ddr_files, profile="full")
        patch_mcs_limb_metadata(limb_path, n_qc, paths, t_idx)

        results.append(
            {
                "hour": f"{hour:02d}",
                "input": str(input_path),
                "output": str(output_path),
                "mcs_limb": str(limb_path),
                "n_mcs": len(bundle["profiles"]),
                "n_qc_pass": n_qc,
                "n_superob": int(m_o.sum()),
            }
        )

    if skipped:
        print(f"[prep] skipped {len(skipped)} hour(s): {skipped}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mars-year", default="MY31")
    p.add_argument("--ls-bin", default="Ls210-240")
    p.add_argument("--hours", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--earth-date", default="2012-11-19")
    p.add_argument("--emars-root", type=Path, default=DEFAULT_EMARS_ROOT)
    p.add_argument("--mcs-root", type=Path, default=DEFAULT_MCS_ROOT)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root. Default: EMARS_MCS_training_data/EMARS_MCS_DA_{yyyymmdd}/ under this package.",
    )
    p.add_argument("--eps", type=float, default=EPS_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    y, m, d = [int(x) for x in args.earth_date.split("-")]
    out_dir = args.out_dir or (TRAINING_DATA_ROOT / f"EMARS_MCS_DA_{y:04d}{m:02d}{d:02d}")
    print(f"out_dir: {out_dir}")
    results = prepare_da_hourly(
        mars_year=args.mars_year,
        ls_bin=args.ls_bin,
        hours=args.hours,
        earth_date=args.earth_date,
        emars_root=args.emars_root,
        mcs_root=args.mcs_root,
        out_dir=out_dir,
        eps=args.eps,
    )
    print(json.dumps({"pairs": results}, indent=2))


if __name__ == "__main__":
    main()
