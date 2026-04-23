#!/bin/bash
#SBATCH --job-name=TimeFlow-Eval-Extrap
#SBATCH --account=project_2010689
#SBATCH --partition=gpusmall
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
module load pytorch/2.5
export PYTHONUSERBASE=/projappl/project_2010689/packages
export PATH=$PYTHONUSERBASE/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/scratch/project_2010689/alifarki/tmp
export PIP_CACHE_DIR=/scratch/project_2010689/alifarki/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$PYTHONUSERBASE/bin"
python -m pip install --user -U pip setuptools wheel
python -m pip install --user mmengine
python -m pip install --user mmcv-lite

DATA_ROOT="/scratch/project_2010689/datasetfull"
PROJECT_DIR="/scratch/project_2010689/2026/TimeFlow_New2"
VOXEL_SIZE="1mm"
MODEL_DIR="/scratch/project_2010689/2026/TimeFlow_Results/timeflow_${VOXEL_SIZE}_extrap"
MODEL_PATH="$MODEL_DIR/best_model.pth"
SAVE_DIR="$MODEL_DIR/evaluation"
IMAGE_SIZE="128 128 128"
MIN_INTERVAL_MONTHS=12
MIN_GAP_MONTHS=6
MAX_EXTRAP_T="2.5"
T_EMBED_DIM=16
USE_DIFFEOMORPHIC=false

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
mkdir -p "$SAVE_DIR"

CMD=(python -u scripts/evaluate_timeflow_adni.py --data-root "$DATA_ROOT" --splits val test --model-path "$MODEL_PATH" --save-dir "$SAVE_DIR" --voxel-size "$VOXEL_SIZE" --image-size $IMAGE_SIZE --min-interval-months "$MIN_INTERVAL_MONTHS" --min-gap-months "$MIN_GAP_MONTHS" --max-extrap-t "$MAX_EXTRAP_T" --t-embed-dim "$T_EMBED_DIM")
[ "$USE_DIFFEOMORPHIC" = true ] && CMD+=(--use-diffeomorphic)

echo "Command:"
printf ' %q' "${CMD[@]}"
echo
echo
"${CMD[@]}"
EXIT_CODE=$?
echo "════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Evaluation completed successfully!"
  echo "Results saved to: $SAVE_DIR"
else
  echo "✗ Evaluation failed with exit code: $EXIT_CODE"
fi
echo "════════════════════════════════════════════════════════════════"
exit $EXIT_CODE
