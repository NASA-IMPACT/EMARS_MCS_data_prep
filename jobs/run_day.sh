#!/bin/bash
#SBATCH -J emars_mcs
#SBATCH -p standard
#SBATCH --ntasks=1
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#
# Submit from the package root:
#   cd EMARS_MCS_data_prep
#   sbatch jobs/run_day.sh
#
set -euo pipefail

# Under Slurm, BASH_SOURCE is a spool copy — use the directory you submitted from.
PKG="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PKG}"
mkdir -p logs EMARS_MCS_training_data
export PYTHONPATH="${PKG}/src:${PYTHONPATH:-}"

python3 scripts/run_workflow.py \
  --earth-date 2012-11-19 \
  --hours $(seq 0 23) \
  --mars-year MY31 \
  --ls-bin Ls210-240 \
  --emars-root "${PKG}/../EMARS" \
  --mcs-root "${PKG}/../obs" \
  --mcs-workers 8
