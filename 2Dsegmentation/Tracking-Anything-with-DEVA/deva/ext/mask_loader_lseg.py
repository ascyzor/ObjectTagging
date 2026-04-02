# DEVA adapter for the LangSeg pipeline.
#
# Replaces automatic_sam.py + automatic_processor.py for the LangSeg path.
# No LangSeg / encoding / pytorch_lightning imports — runs entirely in the
# objecttag Python 3.8 environment.
#
# Contract with DEVA (same as automatic_processor.py):
#   mask          : torch.Tensor  shape (H, W)  dtype int64
#                   0 = background, 1..N = unique instance IDs
#   segments_info : List[ObjectInfo]
#                   one entry per non-zero instance in mask
#                   .id    matches the pixel value
#                   .score  set to 1.0 (LangSeg gives no per-instance confidence)

import json
from os import path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from deva.inference.object_info import ObjectInfo
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.utils.tensor_utils import pad_divide_by, unpad


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_mask_for_frame(frame_path: str, lseg_mask_dir: str,
                        img_path_root: str) -> Tuple[torch.Tensor, List[ObjectInfo]]:
    """
    Load the pre-computed LangSeg instance mask that corresponds to *frame_path*.

    The mask PNG was saved as a 16-bit grayscale image by lseg_inference.py:
        pixel value 0          = background
        pixel value 1 .. 65535 = instance ID

    Returns
    -------
    mask         : torch.int64 tensor  (H, W)
    segments_info: list of ObjectInfo — one per unique non-zero instance in mask
    """
    # Derive the relative path from the image root so we can mirror the layout
    # (e.g. pano_camera0/frame001.jpg → lseg_masks/pano_camera0/frame001.png).
    rel = path.relpath(frame_path, img_path_root)
    stem = path.splitext(rel)[0]
    mask_file = path.join(lseg_mask_dir, stem + '.png')

    if not path.exists(mask_file):
        raise FileNotFoundError(
            f'LangSeg mask not found: {mask_file}\n'
            f'Expected to find a 16-bit PNG mirroring frame: {frame_path}'
        )

    # Read as 16-bit (IMREAD_UNCHANGED preserves the bit depth).
    raw = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f'cv2.imread failed for: {mask_file}')

    mask = torch.from_numpy(raw.astype(np.int64))  # (H, W)

    unique_ids = mask.unique()
    segments_info: List[ObjectInfo] = []
    for uid in unique_ids:
        uid_int = int(uid.item())
        if uid_int == 0:
            continue  # background
        segments_info.append(ObjectInfo(id=uid_int, score=1.0))

    return mask, segments_info


# ---------------------------------------------------------------------------
# DEVA forward-mask estimation (same helper as in automatic_processor.py)
# ---------------------------------------------------------------------------

def _estimate_forward_mask(deva: DEVAInferenceCore, image: torch.Tensor) -> torch.Tensor:
    image, pad = pad_divide_by(image, 16)
    image = image.unsqueeze(0)
    ms_features = deva.image_feature_store.get_ms_features(deva.curr_ti + 1, image)
    key, _, selection = deva.image_feature_store.get_key(deva.curr_ti + 1, image)
    prob = deva._segment(key, selection, ms_features)
    forward_mask = torch.argmax(prob, dim=0)
    return unpad(forward_mask, pad)


# ---------------------------------------------------------------------------
# Per-frame processing — mirrors process_frame_automatic in automatic_processor.py
# ---------------------------------------------------------------------------

@torch.inference_mode()
def process_frame_lseg(deva: DEVAInferenceCore,
                       lseg_mask_dir: str,
                       frame_path: str,
                       result_saver: ResultSaver,
                       ti: int,
                       image_np: np.ndarray = None) -> None:
    """
    Process one frame in the DEVA + LangSeg pipeline.

    Parameters
    ----------
    deva          : DEVA inference core (already initialised).
    lseg_mask_dir : Directory that contains the pre-computed 16-bit PNG masks
                    produced by lseg_inference.py.
    frame_path    : Absolute path to the current RGB frame on disk.
    result_saver  : DEVA result saver.
    ti            : Frame index (0-based).
    image_np      : RGB image array (H×W×3, uint8).  Loaded from disk when None.
    """
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    cfg = deva.config
    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    # Preserve subfolder structure (mirrors automatic_processor.py behaviour).
    frame_name = path.relpath(frame_path, cfg['img_path'])
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    # Load the LangSeg mask for this frame.
    # The PNG was saved at the original image resolution; resize it to DEVA's
    # processing resolution (image tensor: 3 × deva_h × deva_w) using
    # nearest-neighbour so integer instance IDs are preserved.
    mask, segments_info = load_mask_for_frame(frame_path, lseg_mask_dir,
                                              cfg['img_path'])
    deva_h, deva_w = image.shape[1], image.shape[2]
    if mask.shape[0] != deva_h or mask.shape[1] != deva_w:
        mask_np = mask.numpy().astype(np.uint16)
        mask_np = cv2.resize(mask_np, (deva_w, deva_h), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask_np.astype(np.int64))
    mask = mask.to(image.device)

    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np

            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                this_image    = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                _, voted_mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image, voted_mask, new_segments_info)
                deva.next_voting_frame += cfg['detection_every']

                result_saver.save_mask(prob, this_frame_name,
                                       need_resize=need_resize, shape=(h, w),
                                       image_np=this_image_np)

                for buffered in deva.frame_buffer[1:]:
                    prob = deva.step(buffered.image, None, None)
                    result_saver.save_mask(prob, buffered.name,
                                           need_resize, shape=(h, w),
                                           image_np=buffered.image_np)

                deva.clear_buffer()
        else:
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob, frame_name,
                                   need_resize=need_resize, shape=(h, w),
                                   image_np=image_np)

    elif cfg['temporal_setting'] == 'online':
        # detection_every=1 (default) means every frame gets a fresh mask.
        if ti % cfg['detection_every'] == 0:
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info)
        else:
            prob = deva.step(image, None, None)
        result_saver.save_mask(prob, frame_name,
                               need_resize=need_resize, shape=(h, w),
                               image_np=image_np)