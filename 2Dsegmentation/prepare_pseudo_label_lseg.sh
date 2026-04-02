#!/bin/bash
# DEVA + LangSeg pipeline  (parallel to prepare_pseudo_label_sam.sh).
#
# Usage:
#   bash 2Dsegmentation/prepare_pseudo_label_lseg.sh <input_path> <scale> [max_num_objects]
#
# input_path must follow the format:  ~/data/dataset_name/scene_name/
# Output is written to:               <project_root>/2Dmask/dataset_name/scene_name/
#
#   Gray masks (LangSeg) : 2Dmask/dataset_name/scene_name/annotations/gray_mask_lseg/
#   Labels     (LangSeg) : 2Dmask/dataset_name/scene_name/annotations/id2label_lseg.json
#   DEVA metadata        : 2Dmask/dataset_name/scene_name/pred_lseg.json
#
# Two-step architecture (required by Python version incompatibility):
#
#   Step 1 — LangSeg inference
#     Runs under the feature-3dgs Python 3.10 venv (has pytorch_lightning,
#     encoding, etc.).  Produces per-frame 16-bit instance mask PNGs and
#     id2label_lseg.json in a temporary directory.
#
#   Step 2 — DEVA temporal tracking
#     Runs under the objecttag Python 3.8 env (the current interpreter).
#     Reads the pre-computed masks, feeds them to DEVA frame-by-frame for
#     temporal consistency, and writes the final gray_mask_lseg/ PNGs.
#
# Environment variables you can override:
#   FEATURE3DGS_PYTHON  — path to the Python 3.10 interpreter that has LangSeg
#                         (default: ~/PycharmProjects/feature-3dgs/.venv/bin/python)
#   LSEG_ENCODER_PATH   — path to the lseg_encoder directory
#                         (default: ~/PycharmProjects/feature-3dgs/encoders/lseg_encoder)
#   LSEG_CHECKPOINT     — path to demo_e200.ckpt
#                         (default: ${LSEG_ENCODER_PATH}/demo_e200.ckpt)

set -euo pipefail

# -------------------------------------------------------------------------
# Argument validation
# -------------------------------------------------------------------------
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_path> <scale> [max_num_objects]"
    echo "  input_path:      directory ~/data/dataset_name/scene_name/ containing images/ and sparse/."
    echo "  scale:           downscale factor (use 1 for no downscaling)."
    echo "  max_num_objects: max objects DEVA can track (default: 100)."
    exit 1
fi

# -------------------------------------------------------------------------
# Resolve paths
# -------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

input_path="$(realpath "$1")"
scale="$2"
max_num_objects="${3:-100}"

if [ ! -d "$input_path" ]; then
    echo "Error: input_path '$input_path' does not exist."
    exit 2
fi

scene_name="$(basename "$input_path")"
dataset_name="$(basename "$(dirname "$input_path")")"

output_base="${PROJECT_ROOT}/2Dmask/${dataset_name}/${scene_name}"
annotations_dir="${output_base}/annotations"
mask_dir_lseg="${annotations_dir}/gray_mask_lseg"
deva_lseg_tmp="${output_base}/.deva_lseg_tmp"
deva_work_dir="${output_base}/.deva_lseg_out"

mkdir -p "$output_base" "$deva_lseg_tmp" "$deva_work_dir"

echo "==> Dataset : ${dataset_name}"
echo "==> Scene   : ${scene_name}"
echo "==> Output  : ${output_base}"

# -------------------------------------------------------------------------
# Configurable environment variables
# -------------------------------------------------------------------------
FEATURE3DGS_PYTHON="${FEATURE3DGS_PYTHON:-${HOME}/PycharmProjects/feature-3dgs/.venv/bin/python}"
LSEG_ENCODER_PATH="${LSEG_ENCODER_PATH:-${HOME}/PycharmProjects/feature-3dgs/encoders/lseg_encoder}"
LSEG_CHECKPOINT="${LSEG_CHECKPOINT:-${LSEG_ENCODER_PATH}/demo_e200.ckpt}"

if [ ! -f "$FEATURE3DGS_PYTHON" ]; then
    echo "Error: FEATURE3DGS_PYTHON not found: $FEATURE3DGS_PYTHON"
    echo "Set the FEATURE3DGS_PYTHON environment variable to the correct path."
    exit 3
fi
if [ ! -f "$LSEG_CHECKPOINT" ]; then
    echo "Error: LSEG_CHECKPOINT not found: $LSEG_CHECKPOINT"
    echo "Set the LSEG_CHECKPOINT environment variable to the correct path."
    exit 4
fi

# -------------------------------------------------------------------------
# Step 0 — Normalise + optionally downscale images (same as SAM pipeline).
# -------------------------------------------------------------------------
echo "==> Normalising image resolutions + resizing by factor ${scale}x (if > 1) ..."
python "$SCRIPT_DIR/resize_images.py" \
    --image_dir  "${input_path}/images" \
    --factor     "$scale" \
    --output_dir "${output_base}/images_${scale}"

if [ "$scale" -le 1 ]; then
    deva_img_path="${input_path}/images"
else
    deva_img_path="${output_base}/images_${scale}"
fi

# -------------------------------------------------------------------------
# Step 1 — LangSeg inference (feature-3dgs Python 3.10 venv).
#
# Produces:
#   ${deva_lseg_tmp}/lseg_masks/**/*.png   — 16-bit instance mask PNGs
#   ${deva_lseg_tmp}/id2label_lseg.json    — instance_id → label_name
# -------------------------------------------------------------------------
echo "==> Running LangSeg inference (Python 3.10 venv) ..."
"$FEATURE3DGS_PYTHON" "$SCRIPT_DIR/lseg_inference.py" \
    --img_dir    "${deva_img_path}" \
    --out_dir    "${deva_lseg_tmp}" \
    --lseg_path  "${LSEG_ENCODER_PATH}" \
    --checkpoint "${LSEG_CHECKPOINT}"

echo "==> LangSeg masks written to: ${deva_lseg_tmp}/lseg_masks"

# -------------------------------------------------------------------------
# Step 2 — DEVA temporal tracking (objecttag Python 3.8 env).
#
# Reads the 16-bit instance masks from Step 1 and propagates IDs
# consistently across frames using DEVA's memory-based tracker.
#
# Produces inside deva_work_dir:
#   Annotations/   — gray mask PNGs (DEVA output, short_id 8-bit)
#   pred_lseg.json — DEVA video-level metadata
#   id2label_lseg.json — final label map (from LangSeg, intersected with DEVA IDs)
# -------------------------------------------------------------------------
echo "==> Running DEVA tracking (Python 3.8 env) ..."
cd "$SCRIPT_DIR/Tracking-Anything-with-DEVA/"
export PYTHONPATH="$SCRIPT_DIR/Tracking-Anything-with-DEVA:${PYTHONPATH:-}"

python demo/demo_automatic_lseg.py \
    --img_path        "${deva_img_path}" \
    --lseg_mask_dir   "${deva_lseg_tmp}/lseg_masks" \
    --id2label_lseg   "${deva_lseg_tmp}/id2label_lseg.json" \
    --output          "${deva_work_dir}" \
    --use_short_id \
    --chunk_size      4 \
    --temporal_setting semionline \
    --size            768 \
    --amp \
    --max_num_objects "$max_num_objects"

cd "$SCRIPT_DIR"

# -------------------------------------------------------------------------
# Step 3 — Move outputs to final locations with _lseg suffix.
# -------------------------------------------------------------------------
mkdir -p "$mask_dir_lseg" "$annotations_dir"

mv "${deva_work_dir}/Annotations/"* "${mask_dir_lseg}/"
mv "${deva_work_dir}/pred_lseg.json"      "${output_base}/pred_lseg.json"
mv "${deva_work_dir}/id2label_lseg.json"  "${annotations_dir}/id2label_lseg.json"

# Clean up temporary directories.
# rm -rf "$deva_lseg_tmp" "$deva_work_dir"

echo ""
echo "==> LangSeg gray masks  : ${mask_dir_lseg}"
echo "==> id2label_lseg.json  : ${annotations_dir}/id2label_lseg.json"
echo "==> pred_lseg.json      : ${output_base}/pred_lseg.json"