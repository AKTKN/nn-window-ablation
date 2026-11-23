#!/usr/bin/env bash
#PBS -N nnwa_sample_te
#PBS -l select=1:ncpus=96:host=cpucluster031
#PBS -o logs/sample_test.o
#PBS -e logs/sample_test.e

set -euo pipefail
cd -- "${PBS_O_WORKDIR:?リポジトリのルートからqsubで投入してください}"
exec bash scripts/pbs/run_stage.sh generate "${CONFIG_FILE:-scripts/pbs/config/sample_test.yaml}"
