#!/usr/bin/env bash
#PBS -N nnwa_analyze
#PBS -l select=1:ncpus=1:host=cpucluster031
#PBS -o logs/analyze.o
#PBS -e logs/analyze.e

set -euo pipefail
# ローカル解析時はリポジトリのルートからbashで実行する。
cd -- "${PBS_O_WORKDIR:-$PWD}"
exec bash scripts/pbs/run_stage.sh analyze "${CONFIG_FILE:-scripts/pbs/config/analyze.yaml}"
