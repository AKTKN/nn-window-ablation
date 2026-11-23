#!/usr/bin/env bash
# 各ジョブがリポジトリのルートに移動してから呼び出す共通処理。
set -euo pipefail
ulimit -s unlimited

if [[ -n "${CONDA_BASE:-}" ]]; then
    nnwa_conda_base="$CONDA_BASE"
elif command -v conda >/dev/null 2>&1; then
    nnwa_conda_base="$(conda info --base)"
else
    echo 'Condaのモジュールをロードするか、qsub -v CONDA_BASE=... でベースディレクトリを指定してください。' >&2
    exit 1
fi
source "$nnwa_conda_base/etc/profile.d/conda.sh"
conda activate "${NNWA_CONDA_ENV:-nn_window_decoding}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
exec python scripts/pbs/run_from_yaml.py "$@"
