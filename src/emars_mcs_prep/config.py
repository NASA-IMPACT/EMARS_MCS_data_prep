"""Shared paths for the EMARS–MCS DA data-prep workflow."""

from __future__ import annotations

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]  # EMARS_MCS_data_prep/
REPO_ROOT = PKG_ROOT.parent
SRC_ROOT = PKG_ROOT / "src"
SCRIPTS_ROOT = PKG_ROOT / "scripts"
JOBS_ROOT = PKG_ROOT / "jobs"
# Prepared hourly DA NetCDFs (input / output / mcs_limbs).
TRAINING_DATA_ROOT = PKG_ROOT / "EMARS_MCS_training_data"
OUTPUTS_ROOT = TRAINING_DATA_ROOT  # alias for pairs.py DEFAULT_OUT
LOGS_ROOT = PKG_ROOT / "logs"

EMARS_ROOT = REPO_ROOT / "EMARS"
OBS_ROOT = REPO_ROOT / "obs"
REPO_SCRIPTS = REPO_ROOT / "scripts"  # existing OpenMARS download parsers

EMARS_BASE_URL = (
    "https://www.datacommons.psu.edu/download/meteorology/greybush/emars-1p0/data"
)

# Products required for prepare_da_hourly (input / output / limbs).
EMARS_V2_PRODUCTS = ("back_mean", "anal_mean", "anal_sprd")

LS_BINS = tuple(
    f"Ls{a:03d}-{a + 30:03d}" if a < 330 else "Ls330-360" for a in range(0, 360, 30)
)
