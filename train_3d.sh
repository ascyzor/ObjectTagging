#!/bin/bash
# Usage: bash train_3d.sh <colmap_base_dir> <scene_name>
#
# <colmap_base_dir>  Path to the dataset folder containing scene subdirectories
#                    (external, read-only — e.g. /data/spotlite).
#                    The dataset name is derived from the basename of this path.
# <scene_name>       Name of the scene subfolder to train (e.g. baesoohyun).
#
# Before running this script, run ply_preprocessing.py to generate the
# labeled point cloud:
#   python ply_preprocessing.py \
#     --colmap_dir <colmap_base_dir>/<scene_name> \
#     --dataset_name <dataset_name>
#
# All outputs are written to outputs/<dataset_name>/<scene_name>/
# After training the script additionally runs:
#   vis_ply.py            → point_cloud_color.ply  (label-coloured raw point cloud)
#   analyze_ply_segments.py → label_distribution.txt, object_centroids.json,
#                             point_cloud_segmented.ply

if [ $# -lt 2 ]; then
  echo "Usage: $0 <colmap_base_dir> <scene_name>"
  exit 1
fi

SCENE_DIR=$1
SCENE_NAME=$2
scene_dataset=$(basename "$SCENE_DIR")

echo "Training scene: $SCENE_NAME"
python train.py --config config/objectgs/3d/$scene_dataset/config.yaml \
                --scene_name $SCENE_NAME

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# Find the model_path written by this training run: the most recently created
# timestamp directory under outputs/<dataset_name>/<scene_name>/.
OUTPUT_BASE="outputs/${scene_dataset}/${SCENE_NAME}"
MODEL_PATH=$(ls -td "${OUTPUT_BASE}"/[0-9]* 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
  echo "WARNING: Could not find output directory under ${OUTPUT_BASE}. Skipping post-processing."
  exit 0
fi

echo ""
echo "========================================================================"
echo "  Post-processing: MODEL_PATH = $MODEL_PATH"
echo "========================================================================"

# Find the highest saved iteration
ITER=$(ls -d "${MODEL_PATH}/point_cloud/iteration_"* 2>/dev/null \
        | sed 's/.*iteration_//' | sort -n | tail -1)

if [ -z "$ITER" ]; then
  echo "WARNING: No iteration_* directories found in ${MODEL_PATH}/point_cloud/. Skipping post-processing."
  exit 0
fi

ITER_DIR="${MODEL_PATH}/point_cloud/iteration_${ITER}"
PLY_PATH="${ITER_DIR}/point_cloud.ply"
EXPLICIT_PLY="${ITER_DIR}/point_cloud_explicit.ply"
FINAL_PLY="${ITER_DIR}/point_cloud_final.ply"

echo "  Latest iteration : ${ITER}"
echo "  Iteration dir    : ${ITER_DIR}"
echo ""

# ── Step 1: vis_ply.py → point_cloud_color.ply ─────────────────────────────
if [ -f "$PLY_PATH" ]; then
  echo "------------------------------------------------------------------------"
  echo "  Running vis_ply.py on point_cloud.ply …"
  echo "------------------------------------------------------------------------"
  python vis_ply.py "$PLY_PATH" \
         --output "${ITER_DIR}/point_cloud_color.ply"
else
  echo "WARNING: ${PLY_PATH} not found — skipping vis_ply.py."
fi

# ── Step 2: analyze_ply_segments.py ────────────────────────────────────────
if [ -f "$FINAL_PLY" ]; then
  echo ""
  echo "------------------------------------------------------------------------"
  echo "  Running analyze_ply_segments.py …"
  echo "------------------------------------------------------------------------"
  python analyze_ply_segments.py --ply "$FINAL_PLY"
else
  echo "WARNING: ${FINAL_PLY} not found — skipping analyze_ply_segments.py."
  echo "         (point_cloud_final.ply is produced from point_cloud_explicit.ply"
  echo "          when color_attr starts with 'SH' and view_dim == 0.)"
fi

echo ""
echo "========================================================================"
echo "  All done.  Outputs in: ${ITER_DIR}"
echo "========================================================================"
