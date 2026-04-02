#!/bin/bash
# Run this script from anywhere — paths are resolved automatically.
# Usage:
#   bash 2Dsegmentation/prepare_pseudo_label.sh <input_path> <scale> [max_num_objects]
#
# input_path must follow the format:  ~/data/dataset_name/scene_name/
# Output is written automatically to: <project_root>/2Dmask/dataset_name/scene_name/
#   Gray masks : 2Dmask/dataset_name/scene_name/annotations/gray_mask/
#   Labels     : 2Dmask/dataset_name/scene_name/id2label.json

# Check if the user provided an argument
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_path> <scale> [max_num_objects]"
    echo "  input_path:      directory ~/data/dataset_name/scene_name/ containing images/ and sparse/ (COLMAP data)."
    echo "                   This folder is never modified."
    echo "  scale:           downscale factor for DEVA (use 1 for no downscaling)."
    echo "  max_num_objects: maximum number of objects DEVA can track (default: 100)."
    echo "                   Must match num_classes - 1 in your training config JSON."
    exit 1
fi

# Resolve the script's own directory so relative paths work regardless of
# where the caller's working directory is.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve absolute paths so they stay valid after cd into subdirectories.
input_path="$(realpath "$1")"
scale="$2"
max_num_objects="${3:-100}"

if [ ! -d "$input_path" ]; then
    echo "Error: input_path '$input_path' does not exist."
    exit 2
fi

# Derive dataset_name and scene_name from the last two components of input_path.
scene_name="$(basename "$input_path")"
dataset_name="$(basename "$(dirname "$input_path")")"

# Output locations derived from input path structure.
output_base="${PROJECT_ROOT}/2Dmask/${dataset_name}/${scene_name}"
annotations_dir="${output_base}/annotations"
mask_dir="${annotations_dir}/gray_mask"
deva_work_dir="${output_base}/.deva_tmp"   # DEVA writes Annotations/ here; we move it after

mkdir -p "$output_base" "$deva_work_dir"

echo "==> Dataset : ${dataset_name}"
echo "==> Scene   : ${scene_name}"
echo "==> Output  : ${output_base}"


# 0. Normalise image resolutions (always) + optionally downscale by factor.
#    resize_images.py runs a resolution-consistency check as its first step
#    regardless of the scale factor, so it is called unconditionally.
#    Images that already share a single resolution are left untouched.
#    Output is always written to <output_path>/images_<scale>/ so that
#    the input directory is never modified.
echo "==> Normalising image resolutions + resizing by factor ${scale}x (if > 1) ..."
python "$SCRIPT_DIR/resize_images.py" \
    --image_dir  "${input_path}/images" \
    --factor     "$scale" \
    --output_dir "${output_base}/images_${scale}"

# When scale <= 1 no downscaled copy is produced; use the (normalised)
# source images directory directly.  Otherwise use the resized output.
if [ "$scale" -le 1 ]; then
    deva_img_path="${input_path}/images"
else
    deva_img_path="${output_base}/images_${scale}"
fi


# 1. DEVA anything mask
#    DEVA always writes masks to <output>/Annotations/ and metadata to <output>/pred.json.
#    We point it at a temporary working dir, then move Annotations/ to the final location.
cd "$SCRIPT_DIR/Tracking-Anything-with-DEVA/"
export PYTHONPATH="$SCRIPT_DIR/Tracking-Anything-with-DEVA:${PYTHONPATH}"

# colored mask for visualization check (disabled by default)
#python demo/demo_automatic.py \
#  --chunk_size 4 \
#  --img_path "${deva_img_path}" \
#  --amp \
#  --temporal_setting semionline \
#  --size 480 \
#  --output "$deva_work_dir" \
#  --suppress_small_objects  \
#  --SAM_PRED_IOU_THRESHOLD 0.7 \
#  --max_num_objects "$max_num_objects" \

# gray mask for training
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path   "${deva_img_path}" \
  --amp \
  --temporal_setting semionline \
  --size 768 \
  --output     "$deva_work_dir" \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_NUM_POINTS_PER_SIDE 8 \
  --SAM_PRED_IOU_THRESHOLD 0.9 \
  --max_num_objects "$max_num_objects"

cd "$SCRIPT_DIR"

# Move DEVA's Annotations/ output to the final mask location.
mkdir -p "$mask_dir"
mv "${deva_work_dir}/Annotations/"* "$mask_dir/"
# Keep pred.json in output_base for the CLIP step, then clean up the temp dir.
mv "${deva_work_dir}/pred.json" "${output_base}/pred.json"
rm -rf "$deva_work_dir"

echo "==> Gray masks saved to: ${mask_dir}"


# 2. CLIP post-hoc classification: overwrite generic labels with predicted class names.
#    Runs once on all unique objects (not per-frame) — typically adds < 1 s on GPU.
echo "==> Running CLIP post-hoc object classification ..."
python "$SCRIPT_DIR/label_objects_clip.py" \
    --pred_json  "${output_base}/pred.json" \
    --mask_dir   "${mask_dir}" \
    --image_dir  "${deva_img_path}" \
    --output     "${annotations_dir}/id2label.json"

echo "==> id2label.json saved to: ${annotations_dir}/id2label.json"