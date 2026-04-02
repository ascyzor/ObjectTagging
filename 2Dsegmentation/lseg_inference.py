#!/usr/bin/env python
"""
LangSeg inference script — runs under the feature-3dgs Python 3.10 venv.

Converts a directory of RGB frames into per-frame 16-bit instance mask PNGs
plus an id2label_lseg.json file.  These outputs are consumed by
demo_automatic_lseg.py (DEVA step, objecttag Python 3.8 env).

Pipeline per frame
------------------
1. Load image → apply module.val_transform → resize to (360, 480) if width > 480.
2. Run LangSeg (LSeg_MultiEvalModule, scales=[0.75,1.0,1.25,1.75], flip=True)
   → per-pixel logits [1, N_labels, H, W].
3. argmax → semantic class mask [H, W].
4. Per class: cv2.connectedComponents → unique instance IDs.
5. Save as 16-bit grayscale PNG  (pixel value = instance ID, 0 = unassigned).

After all frames:
6. Build id2label_lseg.json  { "instance_id": "label_name" } and write to --out_dir.

Usage
-----
    /path/to/feature3dgs/.venv/bin/python lseg_inference.py \\
        --img_dir    /path/to/images_N              \\
        --out_dir    /path/to/.deva_lseg_tmp        \\
        --lseg_path  /path/to/lseg_encoder          \\
        --checkpoint /path/to/demo_e200.ckpt

Requirements (feature-3dgs venv)
---------------------------------
    pytorch_lightning, clip, encoding (PyTorch-Encoding), torchvision, opencv-python
"""

import argparse
import json
import os
import sys
from os import path
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description='LangSeg per-frame inference')
    parser.add_argument('--img_dir',    required=True,
                        help='Directory containing input RGB frames.')
    parser.add_argument('--out_dir',    required=True,
                        help='Output directory for lseg_masks/ and id2label_lseg.json.')
    parser.add_argument('--lseg_path',  required=True,
                        help='Absolute path to the lseg_encoder/ directory '
                             '(e.g. ~/PycharmProjects/feature-3dgs/encoders/lseg_encoder).')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to demo_e200.ckpt (LangSeg weights).')
    parser.add_argument('--backbone',   default='clip_vitl16_384',
                        help='LSeg backbone name (default: clip_vitl16_384).')
    return parser.parse_args()


def _collect_frames(img_dir: str):
    """Return sorted list of image file paths under img_dir (recursive)."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    frames = []
    for root, _, files in os.walk(img_dir):
        for fn in files:
            if Path(fn).suffix.lower() in exts:
                frames.append(path.join(root, fn))
    return sorted(frames)


def _load_model(checkpoint: str, backbone: str, device: torch.device):
    """Load LSegModule and wrap with MultiEvalModule — mirrors encode_images.py exactly.

    Matches encode_images.py:
      - dataset='ignore' skips trainset/valset setup but still sets base_size=520
      - model = module (not module.net) because LSegNet is not a BaseNet
      - model.cpu() first, then MultiEvalModule(...).to(device)
      - scales=[0.75, 1.0, 1.25, 1.75], flip=True  (same as encode_images.py)

    Returns evaluator, labels, val_transform.
    """
    from modules.lseg_module import LSegModule
    from encoding.models.sseg.base import BaseNet
    from additional_utils.encoding_models import MultiEvalModule as LSeg_MultiEvalModule

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint,
        data_path='.',
        dataset='ignore',
        backbone=backbone,
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin='cpu',
        arch_option=0,
        strict=False,
        block_depth=0,
        activation='lrelu',
    )

    labels        = module.get_labels('ade20k')
    num_classes   = len(labels)
    val_transform = module.val_transform

    # Exact same conditional as encode_images.py line 318-321.
    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval().cpu()   # CPU first — MultiEvalModule moves to GPU

    evaluator = LSeg_MultiEvalModule(
        model, num_classes, scales=[0.75, 1.0, 1.25, 1.75], flip=True
    ).to(device)
    evaluator.eval()
    return evaluator, labels, val_transform


def _preprocess_image(image_np: np.ndarray, val_transform, device: torch.device) -> torch.Tensor:
    """
    Apply module.val_transform then resize to (360, 480) if width > 480.
    Mirrors encode_images.py preprocessing exactly.
    Returns a (1, 3, H', W') float32 tensor on *device*.
    """
    pil_img = PILImage.fromarray(image_np)   # H×W×3 uint8 RGB → PIL
    tensor = val_transform(pil_img)          # (3, H, W)

    # encode_images.py: resize to (h=360, w=480) when width > 480.
    if tensor.shape[-1] > 480:
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(360, 480),
            mode='bilinear', align_corners=True,
        )[0]

    return tensor.unsqueeze(0).to(device)    # (1, 3, H', W')


def _semantic_to_instance(semantic_mask: np.ndarray, labels: list):
    """
    Convert a (H, W) semantic class-ID array into a (H, W) instance-ID array.

    Each connected region of any class becomes a unique instance (pixel value
    starts at 1; 0 means no class was assigned to that pixel).

    Returns
    -------
    instance_mask : np.ndarray  dtype=uint16  shape (H, W)
    id_to_label   : dict  { int instance_id -> str label_name }
    """
    h, w = semantic_mask.shape
    instance_mask = np.zeros((h, w), dtype=np.uint16)
    id_to_label   = {}
    next_id       = 1

    for class_idx, label in enumerate(labels):
        clean_label = label.split(';')[0].strip()

        binary = (semantic_mask == class_idx).astype(np.uint8)
        if binary.sum() == 0:
            continue

        num_cc, cc_map = cv2.connectedComponents(binary, connectivity=8)
        for cc_id in range(1, num_cc):
            component = (cc_map == cc_id)
            instance_mask[component] = next_id
            id_to_label[next_id]     = clean_label
            next_id += 1

    return instance_mask, id_to_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ------------------------------------------------------------------
    # Add lseg_encoder to sys.path so its modules are importable.
    # ------------------------------------------------------------------
    lseg_path = path.abspath(path.expanduser(args.lseg_path))
    if lseg_path not in sys.path:
        sys.path.insert(0, lseg_path)

    # Also add PyTorch-Encoding so `encoding` is importable.
    pytorch_encoding_path = path.join(
        path.dirname(path.dirname(path.dirname(lseg_path))),  # feature-3dgs root
        'PyTorch-Encoding',
    )
    if path.isdir(pytorch_encoding_path) and pytorch_encoding_path not in sys.path:
        sys.path.insert(0, pytorch_encoding_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LSegModule.get_labels() opens 'label_files/ade20k_objectInfo150.txt' as a
    # relative path, so the working directory must be lseg_encoder/ when the
    # model is instantiated.  We restore the original cwd immediately after.
    _orig_cwd = os.getcwd()
    os.chdir(lseg_path)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print('Loading LangSeg model …')
    evaluator, labels, val_transform = _load_model(args.checkpoint, args.backbone, device)
    os.chdir(_orig_cwd)
    print(f'  {len(labels)} labels loaded.')

    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------
    mask_out_dir = path.join(args.out_dir, 'lseg_masks')
    os.makedirs(mask_out_dir, exist_ok=True)

    frames = _collect_frames(args.img_dir)
    if not frames:
        raise RuntimeError(f'No image files found in: {args.img_dir}')
    print(f'Found {len(frames)} frames in {args.img_dir}')

    # Accumulate the union of all instance→label mappings across frames.
    # (An instance ID is local to a frame, not globally unique across frames —
    #  DEVA handles cross-frame ID unification in the tracking step.)
    all_id_to_label: dict = {}

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------
    for frame_path in tqdm(frames, desc='LangSeg inference'):
        # Load image
        bgr = cv2.imread(frame_path)
        if bgr is None:
            print(f'  Warning: could not read {frame_path}, skipping.')
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        # Preprocess — mirrors encode_images.py
        tensor = _preprocess_image(rgb, val_transform, device)  # (1, 3, H', W')

        # LangSeg forward pass — mirrors encode_images.py exactly.
        # parallel_forward expects a list of (3, H', W') tensors and returns
        # a list of (1, N_classes, H', W') logit tensors.
        with torch.no_grad():
            outputs = evaluator.parallel_forward([tensor[0]])
        # outputs[0]: (1, N_classes, H', W') — take argmax over class dim (dim=1)
        semantic_mask = torch.max(outputs[0], 1)[1][0].cpu().numpy().astype(np.int32)
        fh, fw = semantic_mask.shape

        # Resize semantic mask back to original image dimensions.
        if (fh, fw) != (orig_h, orig_w):
            semantic_mask = cv2.resize(
                semantic_mask, (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert semantic → instance via connected components
        instance_mask, frame_id2label = _semantic_to_instance(semantic_mask, labels)
        all_id_to_label.update(frame_id2label)

        # Mirror the input directory structure in the output mask directory.
        rel       = path.relpath(frame_path, args.img_dir)
        stem      = path.splitext(rel)[0]
        mask_path = path.join(mask_out_dir, stem + '.png')
        os.makedirs(path.dirname(mask_path), exist_ok=True)

        # Save as 16-bit PNG (supports up to 65535 instances per frame).
        cv2.imwrite(mask_path, instance_mask)

    # ------------------------------------------------------------------
    # Write id2label_lseg.json
    # ------------------------------------------------------------------
    id2label_path = path.join(args.out_dir, 'id2label_lseg.json')
    # JSON keys must be strings.
    id2label_str = {str(k): v for k, v in sorted(all_id_to_label.items())}
    with open(id2label_path, 'w') as f:
        json.dump(id2label_str, f, indent=4)

    print(f'\nInstance masks → {mask_out_dir}')
    print(f'id2label_lseg  → {id2label_path}')
    print(f'Unique instances across all frames: {len(id2label_str)}')


if __name__ == '__main__':
    main()