#!/usr/bin/env bash
#PBS -N nnwa_evaluate
#PBS -l select=1:ncpus=96:host=cpucluster031
#PBS -o logs/evaluate.o
#PBS -e logs/evaluate.e

set -euo pipefail
cd -- "${PBS_O_WORKDIR:?リポジトリのルートからqsubで投入してください}"
exec bash scripts/pbs/run_stage.sh evaluate "${CONFIG_FILE:-scripts/pbs/config/evaluate.yaml}"
