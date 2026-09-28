# EMARS–MCS data prep

Prepare hourly **EMARS–MCS** data-assimilation training triples: background / super-obs inputs, analysis increments, and MCS limb NetCDFs.

## What it does

For a given Earth date and hours:

1. Ensures local **EMARS** products (`back_mean`, `anal_mean`, `anal_sprd`) and **MCS DDR** coverage (download optional).
2. Builds **distance-weighted** MCS temperature super-observations on the EMARS grid.
3. Writes one-hour NetCDF triples under `EMARS_MCS_training_data/EMARS_MCS_DA_YYYYMMDD/`.

### Super-ob method

- QC: `Gqual=0`, `P_qual=0`, `T_qual=0`.
- Horizontal: nearest EMARS `(lat, lon)` (uses MCS `level_lat` / `level_lon` when available).
- Vertical: nearest **local hybrid** full level at that column  
  \(p_\mathrm{full}(k,j,i)=\tfrac12\big[(a_k+b_k p_s)+(a_{k+1}+b_{k+1}p_s)\big]\)  
  (not the 1D reference `pfull` alone).
- Average: inverse-distance weights  
  \(w = 1/(r_h+1\,\mathrm{km})\cdot 1/(|\ln p-\ln p_\mathrm{full}|+0.05)\).
- \(\sigma_o\): same weights on `Temp_err`, floored at 3 K (not reduced by \(\sqrt{N}\)).

## Layout

```
EMARS_MCS_data_prep/
  src/emars_mcs_prep/   # package (prepare_da_hourly, workflow, I/O)
  scripts/              # CLI: run_workflow.py, run_month.py
  jobs/run_day.sh       # example Slurm job
  EMARS_MCS_training_data/   # outputs (gitignored)
```

Default data roots (sibling of this package):

| Path | Role |
|------|------|
| `../EMARS/` | EMARS v1.0 NetCDFs by Mars year |
| `../obs/` | MCS DDR volumes (`MY*/MCS/MROM_####/`) |

Override with `--emars-root` / `--mcs-root`.

## Requirements

- Python 3 with `numpy`, `netCDF4` (and `xarray` if you use the notebooks outside this package).
- Set `PYTHONPATH` to include `src/`:

```bash
cd EMARS_MCS_data_prep
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

## Quick start

One Earth day (example: 2012-11-19, MY31, all hours):

```bash
python3 scripts/run_workflow.py \
  --earth-date 2012-11-19 \
  --hours $(seq 0 23) \
  --mars-year MY31 \
  --ls-bin Ls210-240 \
  --emars-root ../EMARS \
  --mcs-root ../obs \
  --no-download-mcs \
  --no-download-emars \
  --force-prep
```

Omit `--no-download-*` to allow fetching missing EMARS/MCS if your download helpers are configured.

Slurm (edit date/paths in the script as needed):

```bash
cd EMARS_MCS_data_prep
sbatch jobs/run_day.sh
```

## Outputs

For each hour `HH` under `EMARS_MCS_training_data/EMARS_MCS_DA_YYYYMMDD/`:

| File | Contents |
|------|----------|
| `input/input_MY##_YYYYMMDDHH.nc` | `xb_t`, `y_super_t`, `d_t`, `sigma_o_t`, `M_o`, `n_raw`, … |
| `output/output_MY##_YYYYMMDDHH.nc` | `increment_t`, `increment_norm_t`, `sigma_a_t` |
| `mcs_limbs/mcs_limb_MY##_YYYYMMDDHH.nc` | Raw MCS limb \(T\), \(\sigma_o\), QC flags |

Custom output root: `--out-dir /path/to/EMARS_MCS_DA_YYYYMMDD`.

## Citation / data

- **EMARS:** Greybush et al., Penn State Data Commons ([EMARS 1.0](https://www.datacommons.psu.edu/download/meteorology/greybush/emars-1p0/data)).
- **MCS:** Mars Climate Sounder DDR (PDS).
