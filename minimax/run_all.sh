#!/bin/bash
set -euo pipefail

# Always run from this script's directory
cd "$(dirname "$0")"

# Make conda available and run R from the ltcinf-r prefix env
source /share/amine.mcharrak/miniconda3/etc/profile.d/conda.sh
conda activate /share/amine.mcharrak/conda_envs/ltcinf-r

mkdir -p logs

# Launch 25 chunks (0..24), each chunk generates 8 indices => 200 total.
pids=()
for i in $(seq 0 24); do
  echo "chunk $i"
  nohup R --no-save --args "$i" < surrogate.R > "logs/surrogate_${i}.log" 2>&1 &
  pids+=("$!")
done

# Wait for all chunks to finish
for pid in "${pids[@]}"; do
  wait "$pid"
done

# Sanity check: expect 200 obs + 200 exp files
exp_n=200
obs_cnt=$(ls -1 tmp/obs_*.csv 2>/dev/null | wc -l | tr -d ' ')
exp_cnt=$(ls -1 tmp/exp_*.csv 2>/dev/null | wc -l | tr -d ' ')
if [[ "$obs_cnt" -lt "$exp_n" || "$exp_cnt" -lt "$exp_n" ]]; then
  echo "ERROR: missing synthetic data files: obs=$obs_cnt/$exp_n exp=$exp_cnt/$exp_n" >&2
  exit 1
fi

# Run ridge after data generation is complete
nohup R --no-save < ridge.R > "logs/ridge.log" 2>&1
