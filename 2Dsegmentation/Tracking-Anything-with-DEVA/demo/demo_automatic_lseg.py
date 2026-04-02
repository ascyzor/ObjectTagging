"""
DEVA entry point for the LangSeg pipeline.

Mirrors demo_automatic.py but loads pre-computed LangSeg instance masks
(produced by lseg_inference.py running under the feature-3dgs Python 3.10 venv)
instead of calling SAM at runtime.

No LangSeg / encoding / pytorch_lightning imports — safe to run in the
objecttag Python 3.8 environment.

Usage (called by prepare_pseudo_label_lseg.sh):
    python demo/demo_automatic_lseg.py \\
        --img_path        <images_dir>          \\
        --lseg_mask_dir   <lseg_masks_dir>      \\
        --id2label_lseg   <id2label_lseg.json>  \\
        --output          <output_dir>          \\
        --use_short_id                          \\
        --chunk_size 4                          \\
        --temporal_setting semionline           \\
        --size 768                              \\
        --max_num_objects 100
"""

import json
import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args_lseg import add_ext_eval_args_lseg, add_lseg_default_args
from deva.ext.mask_loader_lseg import process_frame_lseg

from tqdm import tqdm


if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)
    np.random.seed(42)

    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args_lseg(parser)
    add_lseg_default_args(parser)

    deva_model, cfg, args = get_model_and_config(parser)

    # Resolve paths from parsed args into cfg so process_frame_lseg can read them.
    lseg_mask_dir = args.lseg_mask_dir
    id2label_path = args.id2label_lseg

    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']

    # -----------------------------------------------------------------------
    # Data loader
    # -----------------------------------------------------------------------
    video_reader = SimpleVideoReader(cfg['img_path'])
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate,
                        num_workers=8)
    out_path = cfg['output']

    # -----------------------------------------------------------------------
    # Long-term memory enable/disable heuristic (same as demo_automatic.py)
    # -----------------------------------------------------------------------
    vid_length = len(loader)
    cfg['enable_long_term_count_usage'] = (
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
             cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    print('Configuration:', cfg)

    # -----------------------------------------------------------------------
    # DEVA initialisation
    # -----------------------------------------------------------------------
    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = args.num_voting_frames - 1
    if not args.use_short_id:
        deva.enabled_long_id()

    result_saver = ResultSaver(out_path, None, dataset='demo',
                               object_manager=deva.object_manager)

    # -----------------------------------------------------------------------
    # Main inference loop
    # -----------------------------------------------------------------------
    with torch.cuda.amp.autocast(enabled=args.amp):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            process_frame_lseg(deva, lseg_mask_dir, im_path,
                                result_saver, ti, image_np=frame)
        flush_buffer(deva, result_saver)

    result_saver.end()

    # -----------------------------------------------------------------------
    # Save video-level metadata
    # -----------------------------------------------------------------------
    with open(path.join(out_path, 'pred_lseg.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)

    # -----------------------------------------------------------------------
    # Build id2label_lseg.json from the LangSeg labels produced upstream.
    #
    # lseg_inference.py already wrote a comprehensive label map covering every
    # instance ID it detected.  Here we take the intersection with the IDs that
    # DEVA actually kept after temporal tracking, and fall back to the LangSeg
    # label for any ID still present.
    # -----------------------------------------------------------------------
    # Load the upstream label map (written by lseg_inference.py).
    with open(id2label_path, 'r') as f:
        upstream_id2label = json.load(f)  # {str(id): label_name}

    # Collect every object ID that appears in at least one DEVA output frame.
    tracked_ids = set()
    for ann in result_saver.video_json['annotations']:
        for seg in ann['segments_info']:
            tracked_ids.add(seg['id'])

    # Build the final map: prefer upstream LangSeg labels, fall back to generic.
    id2label_final = {}
    for obj_id in sorted(tracked_ids):
        label = upstream_id2label.get(str(obj_id), f'object_{obj_id}')
        id2label_final[str(obj_id)] = label

    os.makedirs(out_path, exist_ok=True)
    with open(path.join(out_path, 'id2label_lseg.json'), 'w') as f:
        json.dump(id2label_final, f, indent=4)

    print(f'pred_lseg.json     → {path.join(out_path, "pred_lseg.json")}')
    print(f'id2label_lseg.json → {path.join(out_path, "id2label_lseg.json")}')