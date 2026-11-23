#!/usr/bin/env bash
#PBS -N nnwa_train
#PBS -l select=1:ncpus=96:host=cpucluster031
#PBS -o logs/train.o
#PBS -e logs/train.e

set -euo pipefail
cd -- "${PBS_O_WORKDIR:?リポジトリのルートからqsubで投入してください}"
exec bash scripts/pbs/run_stage.sh train "${CONFIG_FILE:-scripts/pbs/config/train.yaml}"
