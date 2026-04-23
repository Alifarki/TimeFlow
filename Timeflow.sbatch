#!/bin/bash
#SBATCH --job-name=TimeFlow-Extrap
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

python - <<'PY'
import mmengine
import mmcv
from mmcv.cnn import ConvModule
print('mmengine OK:', mmengine.__version__)
print('mmcv OK:', mmcv.__version__)
print('ConvModule import OK')
PY

DATA_ROOT="/scratch/project_2010689/datasetfull"
PROJECT_DIR="/scratch/project_2010689/2026/TimeFlow_New2"
VOXEL_SIZE="1mm"
IMAGE_SIZE="128 128 128"
MIN_INTERVAL_MONTHS=12
MIN_GAP_MONTHS=6
MAX_EXTRAP_T="2.5"
T_EMBED_DIM=16
USE_DIFFEOMORPHIC=false
BATCH_SIZE=1
EPOCHS=100
LEARNING_RATE="1e-4"
LR_DECAY="0.999"
USE_AMP=true
AMP_DTYPE="bf16"
ANCHOR_SIM_WEIGHT="0.5"
FUTURE_SIM_WEIGHT="0.5"
INTERP_SIM_WEIGHT="1.0"
EXTRAP_SIM_WEIGHT="2.0"
INTERP_DELTA_WEIGHT="1.0"
EXTRAP_DELTA_WEIGHT="2.0"
GRADICON_WEIGHT="0.05"
NUM_WORKERS=4
SAVE_INTERVAL=10
SAVE_DIR="/scratch/project_2010689/2026/TimeFlow_Results/timeflow_${VOXEL_SIZE}_extrap"

echo "════════════════════════════════════════════════════════════════"
echo "TIMEFLOW EXTRAPOLATION-FOCUSED TRAINING CONFIGURATION"
echo "════════════════════════════════════════════════════════════════"
echo "Job ID:             $SLURM_JOB_ID"
echo "Start time:         $(date)"
echo "Node:               $SLURM_NODELIST"
echo "Data Root:          $DATA_ROOT"
echo "Project Directory:  $PROJECT_DIR"
echo "Save Directory:     $SAVE_DIR"
echo "Image Size:         $IMAGE_SIZE"
echo "Min Interval:       ${MIN_INTERVAL_MONTHS} months"
echo "Min Gap:            ${MIN_GAP_MONTHS} months"
echo "Max Extrap t:       $MAX_EXTRAP_T"
echo "Batch Size:         $BATCH_SIZE"
echo "Epochs:             $EPOCHS"
echo "Use AMP:            $USE_AMP ($AMP_DTYPE)"
echo "════════════════════════════════════════════════════════════════"

auto_exit() { echo "$1"; exit 1; }
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
mkdir -p "$SAVE_DIR"
for subdir in train val test; do [ -d "$DATA_ROOT/$subdir" ] || auto_exit "Missing $DATA_ROOT/$subdir"; done
[ -f "scripts/train_timeflow_adni.py" ] || auto_exit "Missing scripts/train_timeflow_adni.py"

CMD=(python -u scripts/train_timeflow_adni.py --data-root "$DATA_ROOT" --voxel-size "$VOXEL_SIZE" --image-size $IMAGE_SIZE --min-interval-months "$MIN_INTERVAL_MONTHS" --min-gap-months "$MIN_GAP_MONTHS" --max-extrap-t "$MAX_EXTRAP_T" --t-embed-dim "$T_EMBED_DIM" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --lr "$LEARNING_RATE" --lr-decay "$LR_DECAY" --anchor-sim-weight "$ANCHOR_SIM_WEIGHT" --future-sim-weight "$FUTURE_SIM_WEIGHT" --interp-sim-weight "$INTERP_SIM_WEIGHT" --extrap-sim-weight "$EXTRAP_SIM_WEIGHT" --interp-delta-weight "$INTERP_DELTA_WEIGHT" --extrap-delta-weight "$EXTRAP_DELTA_WEIGHT" --gradicon-weight "$GRADICON_WEIGHT" --save-dir "$SAVE_DIR" --save-interval "$SAVE_INTERVAL" --num-workers "$NUM_WORKERS" --amp-dtype "$AMP_DTYPE")
[ "$USE_AMP" = true ] && CMD+=(--use-amp)
[ "$USE_DIFFEOMORPHIC" = true ] && CMD+=(--use-diffeomorphic)

echo "Command:"
printf ' %q' "${CMD[@]}"
echo
echo
"${CMD[@]}"
EXIT_CODE=$?

echo "════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Training completed successfully!"
  echo "Best model selected by validation extrapolation Delta-Corr"
else
  echo "✗ Training failed with exit code: $EXIT_CODE"
fi
echo "════════════════════════════════════════════════════════════════"
exit $EXIT_CODE
