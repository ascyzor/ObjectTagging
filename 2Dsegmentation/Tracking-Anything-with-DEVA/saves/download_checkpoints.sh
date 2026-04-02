#!/bin/bash
# Download model checkpoints required by prepare_pseudo_label.sh.
# Run this script from the saves/ directory:
#   cd 2Dsegmentation/Tracking-Anything-with-DEVA/saves
#   bash download_checkpoints.sh
#
# Alternatively, symlink the checkpoints from an existing download:
#   ln -s ~/PycharmProjects/gaussian-grouping/Tracking-Anything-with-DEVA/saves/* .

SAVES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DEVA propagation model
if [ ! -f "$SAVES_DIR/DEVA-propagation.pth" ]; then
    echo "Downloading DEVA-propagation.pth ..."
    wget -P "$SAVES_DIR" \
        "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth"
fi

# SAM ViT-H checkpoint
if [ ! -f "$SAVES_DIR/sam_vit_h_4b8939.pth" ]; then
    echo "Downloading sam_vit_h_4b8939.pth ..."
    wget -P "$SAVES_DIR" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
fi

echo "Done. Required checkpoints:"
ls -lh "$SAVES_DIR"/*.pth 2>/dev/null