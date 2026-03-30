#!/usr/bin/env bash
# =============================================================================
# extract_all_objects.sh
#
# Run edit_ply_object.py --action extract --method remove once for every
# object ID listed in label_distribution.txt, producing one isolated PLY
# per object.
#
# Usage
# -----
#   bash extract_all_objects.sh <ply_file> [options]
#
# Positional argument
# -------------------
#   <ply_file>          Path to point_cloud_explicit.ply
#
# Options
# -------
#   --dist FILE         Path to label_distribution.txt
#                       Default: auto-detected as <ply_dir>/label_distribution.txt
#
#   --output-dir DIR    Directory where extracted PLY files are saved
#                       Default: <ply_dir>/extracted_objects/
#
#   --include-background
#                       Also extract object ID 0 (background / unlabelled).
#                       Background is skipped by default.
#
#   --python EXEC       Python executable to use (default: python)
#
#   --dry-run           Print the commands that would be run without executing them.
#
# Examples
# --------
#   # Minimal — auto-detect label_distribution.txt next to the PLY:
#   bash extract_all_objects.sh \
#       outputs/spotlite/baesoohyun/2026-03-23_18:51:59/point_cloud/iteration_30000/point_cloud_explicit.ply
#
#   # Custom distribution file and output directory:
#   bash extract_all_objects.sh \
#       point_cloud_explicit.ply \
#       --dist my_label_distribution.txt \
#       --output-dir extracted_objects/
#
#   # Also extract background (ID 0):
#   bash extract_all_objects.sh point_cloud_explicit.ply --include-background
#
#   # Preview commands without running:
#   bash extract_all_objects.sh point_cloud_explicit.ply --dry-run
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
PLY_FILE=""
DIST_FILE=""
OUTPUT_DIR=""
INCLUDE_BACKGROUND=0
PYTHON="python"
DRY_RUN=0

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: bash extract_all_objects.sh <ply_file> [options]"
    echo "Run with --help for full usage."
    exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    head -n 50 "$0" | grep "^#" | sed 's/^# \{0,1\}//'
    exit 0
fi

PLY_FILE="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dist)
            DIST_FILE="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --include-background)
            INCLUDE_BACKGROUND=1; shift ;;
        --python)
            PYTHON="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# Resolve paths
# -----------------------------------------------------------------------------
PLY_FILE="$(realpath "$PLY_FILE")"
PLY_DIR="$(dirname "$PLY_FILE")"
PLY_STEM="$(basename "${PLY_FILE%.ply}")"

if [[ -z "$DIST_FILE" ]]; then
    DIST_FILE="${PLY_DIR}/label_distribution.txt"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${PLY_DIR}/extracted_objects"
fi

# -----------------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------------
if [[ ! -f "$PLY_FILE" ]]; then
    echo "ERROR: PLY file not found: $PLY_FILE"
    exit 1
fi

if [[ ! -f "$DIST_FILE" ]]; then
    echo "ERROR: label_distribution.txt not found: $DIST_FILE"
    echo "       Run analyze_ply_segments.py first to generate it, or pass --dist <path>."
    exit 1
fi

# -----------------------------------------------------------------------------
# Parse object IDs from label_distribution.txt
#
# Data lines look like:
#   "  3         person                            500,000   40.50%"
# Field 1 (awk $1) is the numeric object ID.
# We select lines where $1 is a pure non-negative integer.
# -----------------------------------------------------------------------------
mapfile -t ALL_IDS < <(
    awk 'NF >= 4 && $1 ~ /^[0-9]+$/ { print $1 }' "$DIST_FILE" | sort -n
)

if [[ ${#ALL_IDS[@]} -eq 0 ]]; then
    echo "ERROR: No object IDs found in: $DIST_FILE"
    exit 1
fi

# Optionally exclude background (ID 0)
IDS=()
for id in "${ALL_IDS[@]}"; do
    if [[ "$id" -eq 0 && "$INCLUDE_BACKGROUND" -eq 0 ]]; then
        continue
    fi
    IDS+=("$id")
done

if [[ ${#IDS[@]} -eq 0 ]]; then
    echo "WARNING: All IDs were filtered out (only ID 0 present and --include-background not set)."
    exit 0
fi

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
SEP="========================================================================"
echo "$SEP"
echo "  extract_all_objects.sh — ObjectGS batch object extraction"
echo "$SEP"
echo "  PLY file   : $PLY_FILE"
echo "  Dist file  : $DIST_FILE"
echo "  Output dir : $OUTPUT_DIR"
echo "  Method     : remove"
echo "  Python     : $PYTHON"
echo "  IDs found  : ${ALL_IDS[*]}"
if [[ "$INCLUDE_BACKGROUND" -eq 0 ]]; then
    echo "  Skipping   : ID 0 (background) — pass --include-background to keep"
fi
echo "  IDs to run : ${IDS[*]}  (${#IDS[@]} total)"
[[ "$DRY_RUN" -eq 1 ]] && echo "  *** DRY RUN — no commands will be executed ***"
echo "$SEP"
echo ""

# -----------------------------------------------------------------------------
# Create output directory
# -----------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# -----------------------------------------------------------------------------
# Loop over IDs
# -----------------------------------------------------------------------------
FAILED=()
SUCCESS=0

for id in "${IDS[@]}"; do
    OUT_PLY="${OUTPUT_DIR}/${PLY_STEM}_extract_obj${id}_remove.ply"

    CMD=(
        "$PYTHON" edit_ply_object.py
        "$PLY_FILE"
        --action extract
        --ids "$id"
        --method remove
        -o "$OUT_PLY"
    )

    echo "────────────────────────────────────────────────────────────────────────"
    echo "  Object ID : $id"
    echo "  Output    : $OUT_PLY"
    echo "  Command   : ${CMD[*]}"
    echo ""

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [DRY RUN — skipped]"
    else
        if "${CMD[@]}"; then
            SUCCESS=$(( SUCCESS + 1 ))
        else
            echo "  WARNING: Command failed for ID $id (exit code $?)" >&2
            FAILED+=("$id")
        fi
    fi

    echo ""
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "$SEP"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  Dry run complete.  ${#IDS[@]} commands shown."
else
    echo "  Extraction complete."
    echo "  Succeeded : $SUCCESS / ${#IDS[@]}"
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "  Failed IDs: ${FAILED[*]}"
    fi
    echo "  Output dir: $OUTPUT_DIR"
fi
echo "$SEP"

