"""
analyze_ply_segments.py  —  ObjectGS point_cloud_explicit.ply analysis & segmentation

This script takes a point_cloud_explicit.ply (from ObjectGS training output) and
performs three tasks:

  1.  Distribution report
        Read the 'label' field, count Gaussian points per object ID, look up
        human-readable names from id2label.json, and write label_distribution.txt.

  2.  Object centroids
        Compute the mean position (centroid) and bounding box of Gaussians for
        each object ID, and save to object_centroids.json.

  3.  Segmented PLY
        Replace each Gaussian's SH colour with a fixed representative colour
        per object ID (using the same deterministic ID→RGB mapping as
        vis_ply.py / train.py, seed=42) and write point_cloud_segmented.ply
        in the same directory as the input PLY.  SH rest-band coefficients
        are zeroed so the colour appears uniform.

Usage
-----
    # Auto-find latest iteration from model output dir:
    python analyze_ply_segments.py --model_path outputs/spotlite/baesoohyun/2026-03-23_18:51:59

    # Specify a direct PLY path:
    python analyze_ply_segments.py --ply outputs/spotlite/baesoohyun/2026-03-23_18:51:59/point_cloud/iteration_30000/point_cloud_explicit.ply

    # Override iteration:
    python analyze_ply_segments.py --model_path outputs/spotlite/baesoohyun/2026-03-23_18:51:59 --iteration 10000

    # Override id2label:
    python analyze_ply_segments.py --model_path ... --id2label 2Dmask/spotlite/baesoohyun/annotations/id2label.json

Expected layout (auto-detected when --model_path is given)
----------------------------------------------------------
    <model_path>/                          ← timestamp dir, e.g. 2026-03-23_18:51:59
        config.yaml
        point_cloud/
            iteration_N/
                point_cloud_explicit.ply   ← input (auto-found at highest N)
                label_distribution.txt     ← output: distribution report
                object_centroids.json      ← output: per-object centroids
                point_cloud_segmented.ply  ← output: colour-coded PLY

id2label.json is auto-detected from:
    2Dmask/<dataset_name>/<scene_name>/annotations/id2label.json
where dataset_name comes from config.yaml and scene_name is the parent of
the timestamp directory.
"""

import os
import sys
import json
import argparse
from collections import Counter
from typing import Optional

import numpy as np
import yaml
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------------------------
# SH degree-0 normalisation constant (consistent with 3DGS / gsplat / SuperSplat)
# ---------------------------------------------------------------------------
SH_C0 = 0.28209479177387814   # = 1 / (2 * sqrt(pi))


# ===========================================================================
# 1.  Colour mapping  (consistent with train.py and vis_ply.py)
# ===========================================================================

class ID2RGBConverter:
    """Deterministic int → uint8 RGB mapping (seed=42, same as rest of codebase)."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self._map = np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)

    def convert(self, obj_id: int) -> np.ndarray:
        """Return uint8 RGB (3,) for *obj_id*.  ID 0 → black."""
        if obj_id == 0:
            return np.array([0, 0, 0], dtype=np.uint8)
        if 0 < obj_id <= 255:
            return self._map[obj_id]
        raise ValueError(f"Object ID {obj_id} is outside [0, 255].")


def rgb_uint8_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """
    Convert uint8 RGB [0, 255] → float32 SH degree-0 (DC) coefficients.

    Standard 3DGS/SuperSplat convention:
        f_dc = (rgb_float − 0.5) / SH_C0
    """
    return (rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0


# ===========================================================================
# 2.  Path helpers
# ===========================================================================

def find_latest_iteration(model_path: str) -> int:
    """Return the highest iteration number under <model_path>/point_cloud/."""
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        raise FileNotFoundError(
            f"point_cloud/ directory not found inside model_path:\n  {model_path}"
        )
    iters = [
        int(name.split("_")[-1])
        for name in os.listdir(pc_dir)
        if name.startswith("iteration_") and name.split("_")[-1].isdigit()
    ]
    if not iters:
        raise FileNotFoundError(
            f"No iteration_* subdirectories found in:\n  {pc_dir}"
        )
    return max(iters)


def resolve_ply_path(args) -> str:
    """
    Resolve the absolute path to point_cloud_explicit.ply.

    Priority:
        1. --ply   (explicit path; used as-is)
        2. --model_path + --iteration  (locate specific iteration)
        3. --model_path   (auto-detect latest iteration)
    """
    if args.ply:
        path = os.path.abspath(args.ply)
        if not os.path.isfile(path):
            sys.exit(f"\nERROR: PLY file not found:\n  {path}\n")
        return path

    if not args.model_path:
        sys.exit(
            "\nERROR: Provide either --ply <path> or --model_path <dir>.\n"
            "Run with -h for usage.\n"
        )

    model_path = os.path.abspath(args.model_path)
    if args.iteration > 0:
        iteration = args.iteration
    else:
        try:
            iteration = find_latest_iteration(model_path)
        except FileNotFoundError as exc:
            sys.exit(f"\nERROR: {exc}\n")

    path = os.path.join(
        model_path, "point_cloud",
        f"iteration_{iteration}", "point_cloud_final.ply"
    )
    if not os.path.isfile(path):
        sys.exit(
            f"\nERROR: point_cloud_final.ply not found at:\n  {path}\n"
            "Ensure the model was trained with color_attr starting with 'SH' "
            "and view_dim == 0 (so save_explicit() and _save_final_ply() are "
            "called during training).\n"
        )
    return path


def model_path_from_ply(ply_path: str) -> str:
    """
    Infer model root (timestamp directory) from PLY path.
    Expected tree:  <model_path>/point_cloud/iteration_N/point_cloud_explicit.ply
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(ply_path)))


def load_config(model_path: str) -> dict:
    """Load config.yaml from the model root directory; return {} if absent."""
    cfg_path = os.path.join(model_path, "config.yaml")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader) or {}


def auto_locate_id2label(model_path: str, cfg: dict) -> Optional[str]:
    """
    Locate id2label.json from the standard ObjectGS 2Dmask annotation tree:
        2Dmask/<dataset_name>/<scene_name>/annotations/id2label.json

    dataset_name  ← config.yaml → model_params → dataset_name
    scene_name    ← parent directory of the timestamp folder (model_path)
    """
    dataset_name = cfg.get("model_params", {}).get("dataset_name", "")
    scene_name   = os.path.basename(os.path.dirname(os.path.abspath(model_path)))

    if not dataset_name:
        return None

    candidate = os.path.join(
        "2Dmask", dataset_name, scene_name, "annotations", "id2label.json"
    )
    return candidate if os.path.isfile(candidate) else None


def load_id2label(id2label_arg: Optional[str], model_path: str, cfg: dict) -> dict:
    """Load and return id2label dict; return {} if not available."""
    path = id2label_arg or auto_locate_id2label(model_path, cfg)
    if path and os.path.isfile(path):
        with open(path) as fh:
            data = json.load(fh)
        print(f"  id2label : {path}  ({len(data)} entries)")
        return data
    msg = "  id2label : not found — label names shown as 'unknown'."
    if path:
        msg += f"\n             (Tried: {path})"
    print(msg)
    return {}


# ===========================================================================
# 3.  PLY I/O
# ===========================================================================

def read_explicit_ply(ply_path: str) -> dict:
    """
    Read point_cloud_explicit.ply and return arrays:
        xyz        (N, 3)  float32   Gaussian centres
        f_dc       (N, 3)  float32   SH DC coefficients [f_dc_0, f_dc_1, f_dc_2]
        f_rest     (N, K)  float32   SH rest coefficients (K may be 0)
        opacity    (N,)    float32   pre-sigmoid logit
        scale      (N, 3)  float32   log scale
        rot        (N, 4)  float32   unit quaternion [rot_0..3]
        label      (N,)    uint8     object ID  (zeros if field absent)
        rest_names list[str]         names of the f_rest_* properties
        has_label  bool              whether the PLY contained a 'label' field
    """
    plydata    = PlyData.read(ply_path)
    vertex     = plydata.elements[0]
    prop_names = {p.name for p in vertex.properties}

    xyz = np.stack([
        np.asarray(vertex["x"]),
        np.asarray(vertex["y"]),
        np.asarray(vertex["z"]),
    ], axis=1).astype(np.float32)

    f_dc = np.stack([
        np.asarray(vertex["f_dc_0"]),
        np.asarray(vertex["f_dc_1"]),
        np.asarray(vertex["f_dc_2"]),
    ], axis=1).astype(np.float32)

    rest_names = sorted(
        [n for n in prop_names if n.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    f_rest = (
        np.stack([np.asarray(vertex[n]) for n in rest_names], axis=1).astype(np.float32)
        if rest_names
        else np.zeros((xyz.shape[0], 0), dtype=np.float32)
    )

    opacity = np.asarray(vertex["opacity"]).astype(np.float32)
    scale   = np.stack([
        np.asarray(vertex["scale_0"]),
        np.asarray(vertex["scale_1"]),
        np.asarray(vertex["scale_2"]),
    ], axis=1).astype(np.float32)
    rot = np.stack([
        np.asarray(vertex["rot_0"]),
        np.asarray(vertex["rot_1"]),
        np.asarray(vertex["rot_2"]),
        np.asarray(vertex["rot_3"]),
    ], axis=1).astype(np.float32)

    has_label = "label" in prop_names
    if has_label:
        label = np.asarray(vertex["label"]).astype(np.uint8)
    else:
        print("  WARNING: 'label' field absent — assigning all points to ID 0.")
        label = np.zeros(xyz.shape[0], dtype=np.uint8)

    return {
        "xyz":        xyz,
        "f_dc":       f_dc,
        "f_rest":     f_rest,
        "opacity":    opacity,
        "scale":      scale,
        "rot":        rot,
        "label":      label,
        "rest_names": rest_names,
        "has_label":  has_label,
    }


def write_segmented_ply(out_path: str, data: dict, seg_f_dc: np.ndarray) -> None:
    """
    Write point_cloud_segmented.ply — identical schema to the input explicit PLY
    but with per-object-colour SH DC coefficients and zeroed rest SH bands.

    Parameters
    ----------
    out_path  : output file path
    data      : dict returned by read_explicit_ply()
    seg_f_dc  : (N, 3) float32 — replacement SH DC colours
    """
    xyz     = data["xyz"]
    f_rest  = data["f_rest"]
    opacity = data["opacity"]
    scale   = data["scale"]
    rot     = data["rot"]
    label   = data["label"]
    N, K    = xyz.shape[0], f_rest.shape[1]

    dtype = (
          [("x", "f4"), ("y", "f4"), ("z", "f4")]
        + [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
        + [(f"f_rest_{i}", "f4") for i in range(K)]
        + [("opacity", "f4")]
        + [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
        + [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
        # label omitted — web format for Supersplat
    )

    el = np.empty(N, dtype=dtype)
    el["x"],     el["y"],     el["z"]     = xyz[:, 0],     xyz[:, 1],     xyz[:, 2]
    el["f_dc_0"], el["f_dc_1"], el["f_dc_2"] = (
        seg_f_dc[:, 0], seg_f_dc[:, 1], seg_f_dc[:, 2]
    )
    for i in range(K):
        el[f"f_rest_{i}"] = 0.0           # zero out rest SH bands → solid colour
    el["opacity"]                          = opacity
    el["scale_0"], el["scale_1"], el["scale_2"] = (
        scale[:, 0], scale[:, 1], scale[:, 2]
    )
    el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"] = (
        rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    )

    PlyData([PlyElement.describe(el, "vertex")]).write(out_path)
    print(f"  Saved → {out_path}  ({N:,} points)")


# ===========================================================================
# 4.  Analysis tasks
# ===========================================================================

def task_distribution(out_path: str, label: np.ndarray,
                      id2label: dict, ply_basename: str) -> None:
    """
    Count Gaussian points per object ID, write label_distribution.txt.
    """
    total   = len(label)
    counter = Counter(label.tolist())

    # Sort: foreground objects by descending count; background (ID 0) at bottom
    rows = sorted(
        counter.items(),
        key=lambda kv: (kv[0] == 0, -kv[1], kv[0]),
    )

    W   = 72
    sep = "=" * W

    lines = [
        sep,
        "  Gaussian Point Distribution by Object ID",
        f"  Source          : {ply_basename}",
        f"  Total Gaussians : {total:,}",
        f"  Distinct IDs    : {len(counter)}",
        sep,
        f"  {'ID':<8}  {'Label':<30}  {'Gaussians':>12}  {'%':>8}",
        "  " + "-" * (W - 2),
    ]

    for obj_id, count in rows:
        name = (
            "background / unlabelled"
            if obj_id == 0
            else id2label.get(str(obj_id), "unknown")
        )
        pct = 100.0 * count / total
        lines.append(
            f"  {obj_id:<8}  {name:<30}  {count:>12,}  {pct:>7.2f}%"
        )

    lines += ["  " + "-" * (W - 2), sep]
    report = "\n".join(lines)

    with open(out_path, "w") as fh:
        fh.write(report + "\n")

    print(f"  Saved → {out_path}")
    print()
    print(report)


def task_centroids(out_path: str, xyz: np.ndarray,
                   label: np.ndarray, id2label: dict) -> None:
    """
    Compute per-object centroid (mean xyz) and axis-aligned bounding box,
    then write object_centroids.json.

    JSON schema per object:
    {
        "<id>": {
            "label":       "<name>",
            "centroid":    [cx, cy, cz],     // mean position
            "bbox_min":    [xmin, ymin, zmin],
            "bbox_max":    [xmax, ymax, zmax],
            "point_count": <int>
        },
        ...
    }
    """
    result = {}

    for obj_id in sorted(np.unique(label).tolist()):
        mask = label == obj_id
        pts  = xyz[mask]

        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        cz = float(pts[:, 2].mean())

        name = (
            "background / unlabelled"
            if obj_id == 0
            else id2label.get(str(obj_id), "unknown")
        )

        result[str(obj_id)] = {
            "label":       name,
            "centroid":    [round(cx, 6), round(cy, 6), round(cz, 6)],
            "bbox_min":    [
                round(float(pts[:, 0].min()), 6),
                round(float(pts[:, 1].min()), 6),
                round(float(pts[:, 2].min()), 6),
            ],
            "bbox_max":    [
                round(float(pts[:, 0].max()), 6),
                round(float(pts[:, 1].max()), 6),
                round(float(pts[:, 2].max()), 6),
            ],
            "point_count": int(mask.sum()),
        }

    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"  Saved → {out_path}  ({len(result)} objects)")

    # Pretty-print summary table
    W = 72
    print()
    print("  " + "-" * (W - 2))
    print(f"  {'ID':<6}  {'Label':<28}  {'Centroid (x, y, z)':^40}  {'Pts':>10}")
    print("  " + "-" * (W - 2))
    for obj_id_str, info in result.items():
        cx, cy, cz = info["centroid"]
        print(
            f"  {obj_id_str:<6}  {info['label']:<28}  "
            f"({cx:>10.4f}, {cy:>10.4f}, {cz:>10.4f})  "
            f"{info['point_count']:>10,}"
        )
    print("  " + "-" * (W - 2))


def task_segmented_ply(out_path: str, data: dict) -> None:
    """
    Replace each Gaussian's SH colour with a per-object representative colour
    (using ID2RGBConverter, seed=42) and write point_cloud_segmented.ply.
    """
    label     = data["label"]
    converter = ID2RGBConverter(seed=42)
    unique_ids = np.unique(label)

    # Build per-point SH-DC colour array
    seg_f_dc = np.zeros((len(label), 3), dtype=np.float32)
    for obj_id in unique_ids:
        rgb_u8    = converter.convert(int(obj_id))          # uint8 [3]
        sh_dc     = rgb_uint8_to_sh_dc(rgb_u8)              # float32 [3]
        seg_f_dc[label == obj_id] = sh_dc

    # Print colour map for reference
    print()
    print(f"  {'ID':<6}  {'Representative RGB (uint8)':^30}")
    print("  " + "-" * 40)
    for obj_id in unique_ids:
        rgb = converter.convert(int(obj_id))
        print(f"  {int(obj_id):<6}  ({rgb[0]:>3}, {rgb[1]:>3}, {rgb[2]:>3})")
    print()

    write_segmented_ply(out_path, data, seg_f_dc)


# ===========================================================================
# 5.  CLI entry point
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_grp = ap.add_mutually_exclusive_group()
    input_grp.add_argument(
        "--model_path", default=None, metavar="DIR",
        help=(
            "Path to a training-output timestamp directory, e.g. "
            "outputs/spotlite/baesoohyun/2026-03-23_18:51:59.  "
            "The script auto-locates the latest iteration's "
            "point_cloud_explicit.ply inside it."
        ),
    )
    input_grp.add_argument(
        "--ply", default=None, metavar="FILE",
        help=(
            "Direct path to point_cloud_explicit.ply.  "
            "Mutually exclusive with --model_path."
        ),
    )
    ap.add_argument(
        "--iteration", type=int, default=-1, metavar="N",
        help=(
            "Specific iteration to load (used with --model_path).  "
            "Default: -1 → auto-select the latest saved iteration."
        ),
    )
    ap.add_argument(
        "--id2label", default=None, metavar="FILE",
        help=(
            "Path to id2label.json.  "
            "Auto-detected from the 2Dmask/ tree when omitted."
        ),
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Resolve PLY path
    # ------------------------------------------------------------------
    ply_path = resolve_ply_path(args)
    iter_dir = os.path.dirname(ply_path)           # outputs go here too

    model_path = (
        os.path.abspath(args.model_path)
        if args.model_path
        else model_path_from_ply(ply_path)
    )

    W = 72
    print("=" * W)
    print("  ObjectGS — PLY Segment Analysis")
    print("=" * W)
    print(f"  Input PLY  : {ply_path}")
    print(f"  Model root : {model_path}")
    print(f"  Output dir : {iter_dir}")
    print("=" * W)

    # ------------------------------------------------------------------
    # Load config.yaml + id2label.json
    # ------------------------------------------------------------------
    cfg      = load_config(model_path)
    id2label = load_id2label(args.id2label, model_path, cfg)

    # ------------------------------------------------------------------
    # Read the PLY
    # ------------------------------------------------------------------
    print(f"\n  Reading PLY …")
    data = read_explicit_ply(ply_path)
    N, K = data["xyz"].shape[0], data["f_rest"].shape[1]

    print(f"  Points loaded  : {N:,}")
    print(f"  SH rest bands  : {K}")
    print(f"  Label field    : {'present' if data['has_label'] else 'ABSENT (all zeros)'}")
    unique_ids = np.unique(data["label"]).tolist()
    print(f"  Unique IDs     : {len(unique_ids)}  →  {unique_ids}")

    ply_basename = os.path.basename(ply_path)

    # ------------------------------------------------------------------
    # Task 1 — Distribution report
    # ------------------------------------------------------------------
    print(f"\n{'─' * W}")
    print("  TASK 1 : Label distribution  →  label_distribution.txt")
    print(f"{'─' * W}")
    dist_path = os.path.join(iter_dir, "label_distribution.txt")
    task_distribution(dist_path, data["label"], id2label, ply_basename)

    # ------------------------------------------------------------------
    # Task 2 — Object centroids
    # ------------------------------------------------------------------
    print(f"\n{'─' * W}")
    print("  TASK 2 : Object centroids  →  object_centroids.json")
    print(f"{'─' * W}")
    centroids_path = os.path.join(iter_dir, "object_centroids.json")
    task_centroids(centroids_path, data["xyz"], data["label"], id2label)

    # ------------------------------------------------------------------
    # Task 3 — Segmented PLY
    # ------------------------------------------------------------------
    print(f"\n{'─' * W}")
    print("  TASK 3 : Per-object colour replacement  →  point_cloud_segmented.ply")
    print(f"{'─' * W}")
    seg_path = os.path.join(iter_dir, "point_cloud_segmented.ply")
    task_segmented_ply(seg_path, data)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("  All tasks complete.")
    print(f"{'─' * W}")
    print(f"  label_distribution.txt     →  {dist_path}")
    print(f"  object_centroids.json      →  {centroids_path}")
    print(f"  point_cloud_segmented.ply  →  {seg_path}")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()

