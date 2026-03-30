"""
edit_ply_object.py  —  ObjectGS  Extract / Delete objects in point_cloud_explicit.ply

Given a point_cloud_explicit.ply produced by ObjectGS training this script
edits the Gaussian point cloud based on object IDs stored in the `label` field.

Actions
-------
  extract   Keep only the Gaussians belonging to the specified object ID(s).
            All other Gaussians are either removed (method=remove) or made
            effectively invisible (method=zeroop).

  delete    Remove the Gaussians belonging to the specified object ID(s).
            Matching Gaussians are either removed (method=remove) or made
            effectively invisible (method=zeroop).

Methods
-------
  remove    Physically delete the target Gaussians from the output PLY.
            The output file has fewer rows than the input.

  zeroop    Keep every Gaussian but set the pre-sigmoid opacity logit of the
            target rows to ZERO_OPACITY_LOGIT (default −15.0), which maps to
            sigmoid(−15) ≈ 3e-7 — effectively invisible in any renderer.
            The output file has the same number of rows as the input.

Usage examples
--------------
    # Extract object 3; physically remove non-matching Gaussians:
    python edit_ply_object.py input.ply --action extract --ids 3 --method remove

    # Extract objects 1 and 2; keep all Gaussians but zero out the rest:
    python edit_ply_object.py input.ply --action extract --ids 1 2 --method zeroop

    # Delete object 5; physically remove it:
    python edit_ply_object.py input.ply --action delete --ids 5 --method remove

    # Delete object 5; make it invisible but keep the PLY shape intact:
    python edit_ply_object.py input.ply --action delete --ids 5 --method zeroop

    # Custom output path:
    python edit_ply_object.py input.ply --action extract --ids 3 --method remove -o object_3.ply

    # Custom near-zero logit threshold (default is -15.0):
    python edit_ply_object.py input.ply --action delete --ids 2 --method zeroop --zero_logit -20.0

Notes
-----
  • The `opacity` attribute in point_cloud_explicit.ply is stored as a
    pre-sigmoid logit value.  The renderer applies sigmoid() at runtime, so
    a very negative logit (−15) produces opacity ≈ 3e-7 — invisible.
  • The `label` field is a uint8, so valid object IDs are 0–255.
  • ID 0 is conventionally the background / unlabelled class.
"""

import os
import sys
import argparse
from typing import List

import numpy as np
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pre-sigmoid logit that renders as near-zero opacity:  sigmoid(-15) ≈ 3.06e-7
DEFAULT_ZERO_OPACITY_LOGIT: float = -15.0


# ===========================================================================
# I/O helpers
# ===========================================================================

def read_explicit_ply(ply_path: str) -> dict:
    """
    Read point_cloud_explicit.ply and return a dict of numpy arrays.

    Returns
    -------
    {
        "xyz"        : (N, 3)  float32  — Gaussian centres
        "f_dc"       : (N, 3)  float32  — SH degree-0 (DC) coefficients
        "f_rest"     : (N, K)  float32  — higher SH-band coefficients (K ≥ 0)
        "opacity"    : (N,)    float32  — pre-sigmoid logit
        "scale"      : (N, 3)  float32  — log scale
        "rot"        : (N, 4)  float32  — unit quaternion [rot_0..rot_3]
        "label"      : (N,)    uint8    — object ID
        "rest_names" : list[str]        — ordered f_rest_* property names
        "has_label"  : bool             — whether 'label' was present in PLY
    }
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
        print("  WARNING: 'label' field absent in input PLY — all points assigned ID 0.")
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


def write_explicit_ply(out_path: str, data: dict) -> None:
    """
    Write an explicit PLY using the same schema as the input file.

    All f_rest_* bands and the label field are preserved; the number of
    columns is inferred from data["f_rest"].
    """
    xyz     = data["xyz"]
    f_dc    = data["f_dc"]
    f_rest  = data["f_rest"]
    opacity = data["opacity"]
    scale   = data["scale"]
    rot     = data["rot"]
    label   = data["label"]
    N, K    = xyz.shape[0], f_rest.shape[1]

    dtype = (
          [("x", "f4"), ("y", "f4"), ("z", "f4")]
        + [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]   # required by Supersplat/standard 3DGS viewers
        + [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
        + [(f"f_rest_{i}", "f4") for i in range(K)]
        + [("opacity", "f4")]
        + [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
        + [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
        # label omitted — web format for Supersplat
    )

    el = np.empty(N, dtype=dtype)
    el["x"],  el["y"],  el["z"]  = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el["nx"], el["ny"], el["nz"] = 0.0, 0.0, 0.0          # zeroed normals (standard 3DGS)
    el["f_dc_0"], el["f_dc_1"], el["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(K):
        el[f"f_rest_{i}"] = f_rest[:, i]
    el["opacity"]                              = opacity
    el["scale_0"], el["scale_1"], el["scale_2"] = (
        scale[:, 0], scale[:, 1], scale[:, 2]
    )
    el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"] = (
        rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    )

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    PlyData([PlyElement.describe(el, "vertex")]).write(out_path)
    print(f"  Saved → {out_path}  ({N:,} Gaussians)")


# ===========================================================================
# Core editing logic
# ===========================================================================

def build_id_mask(label: np.ndarray, ids: List[int]) -> np.ndarray:
    """Return bool mask that is True for every point whose label is in *ids*."""
    mask = np.zeros(len(label), dtype=bool)
    for obj_id in ids:
        mask |= (label == obj_id)
    return mask


def edit_point_cloud(
    data: dict,
    action: str,
    ids: List[int],
    method: str,
    zero_logit: float = DEFAULT_ZERO_OPACITY_LOGIT,
) -> dict:
    """
    Apply the requested action + method combination to the Gaussian data.

    Parameters
    ----------
    data       : dict from read_explicit_ply()
    action     : 'extract' | 'delete'
    ids        : object IDs to operate on
    method     : 'remove' | 'zeroop'
    zero_logit : pre-sigmoid logit used when method == 'zeroop'

    Returns
    -------
    Edited copy of *data* (original dict is never mutated).
    """
    label       = data["label"]
    target_mask = build_id_mask(label, ids)   # True  = matches requested IDs

    n_total   = len(label)
    n_matched = int(target_mask.sum())

    print(f"  Total Gaussians in input  : {n_total:,}")
    print(f"  Gaussians matching IDs {ids}: {n_matched:,}  "
          f"({100.0 * n_matched / max(n_total, 1):.2f} %)")

    if n_matched == 0:
        print("  WARNING: No Gaussians matched the specified IDs — "
              "output will be identical to the input.")

    # ------------------------------------------------------------------
    # Determine which Gaussians to KEEP (untouched) vs HIDE/REMOVE
    # ------------------------------------------------------------------
    #   extract → keep the target set,   hide/remove everything else
    #   delete  → keep everything else,  hide/remove the target set
    if action == "extract":
        keep_mask = target_mask
        hide_mask = ~target_mask
    else:  # delete
        keep_mask = ~target_mask
        hide_mask = target_mask

    # Deep-copy arrays (avoid mutating the caller's data)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in data.items()}

    if method == "remove":
        # ── physically filter rows ──────────────────────────────────────
        for key in ("xyz", "f_dc", "f_rest", "opacity", "scale", "rot", "label"):
            out[key] = data[key][keep_mask]

        n_removed = int(hide_mask.sum())
        n_kept    = int(keep_mask.sum())
        print(f"  Method: remove  — {n_removed:,} Gaussians removed, "
              f"{n_kept:,} retained.")

    else:  # zeroop
        # ── clamp opacity logit of hidden Gaussians ─────────────────────
        out["opacity"][hide_mask] = zero_logit

        n_zeroed = int(hide_mask.sum())
        n_kept   = int(keep_mask.sum())
        print(f"  Method: zeroop  — {n_zeroed:,} Gaussians set to opacity logit "
              f"{zero_logit:.1f} (sigmoid ≈ {1.0/(1.0+np.exp(-zero_logit)):.1e}), "
              f"{n_kept:,} unchanged.")

    return out


# ===========================================================================
# Auto-naming for the output file
# ===========================================================================

def default_output_path(
    input_path: str, action: str, ids: List[int], method: str
) -> str:
    """
    Build a descriptive output filename next to the input file.

    Pattern:  <stem>_<action>_obj<id1>[-<id2>…]_<method>.ply
    Example:  point_cloud_explicit_extract_obj3_remove.ply
    """
    stem     = os.path.splitext(os.path.basename(input_path))[0]
    ids_str  = "-".join(str(i) for i in sorted(ids))
    out_name = f"{stem}_{action}_obj{ids_str}_{method}.ply"
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), out_name)


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "input",
        help="Path to point_cloud_explicit.ply (ObjectGS training output).",
    )
    ap.add_argument(
        "--action", required=True, choices=["extract", "delete"],
        help=(
            "'extract': keep only Gaussians with the specified label IDs; "
            "'delete': remove Gaussians with the specified label IDs."
        ),
    )
    ap.add_argument(
        "--ids", required=True, type=int, nargs="+", metavar="ID",
        help="One or more integer object IDs (0–255) to extract or delete.",
    )
    ap.add_argument(
        "--method", required=True, choices=["remove", "zeroop"],
        help=(
            "'remove': physically delete matching Gaussians from the PLY "
            "(output has fewer rows). "
            "'zeroop': keep all Gaussians but set opacity logit of matching "
            "rows to --zero_logit (default −15) so they vanish after sigmoid."
        ),
    )
    ap.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help=(
            "Output PLY path.  "
            "Default: auto-generated next to the input file, e.g. "
            "point_cloud_explicit_extract_obj3_remove.ply."
        ),
    )
    ap.add_argument(
        "--zero_logit", type=float, default=DEFAULT_ZERO_OPACITY_LOGIT,
        metavar="LOGIT",
        help=(
            f"Pre-sigmoid opacity logit assigned to hidden Gaussians when "
            f"method=zeroop (default: {DEFAULT_ZERO_OPACITY_LOGIT}).  "
            "sigmoid(−15) ≈ 3e-7."
        ),
    )

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Validate input path
    # ------------------------------------------------------------------
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        sys.exit(f"\nERROR: Input PLY not found:\n  {input_path}\n")

    ids = sorted(set(args.ids))

    # ------------------------------------------------------------------
    # Resolve output path
    # ------------------------------------------------------------------
    output_path = (
        os.path.abspath(args.output)
        if args.output
        else default_output_path(input_path, args.action, ids, args.method)
    )

    # ------------------------------------------------------------------
    # Print banner
    # ------------------------------------------------------------------
    W = 72
    print("=" * W)
    print("  ObjectGS — Edit PLY Object")
    print("=" * W)
    print(f"  Input PLY : {input_path}")
    print(f"  Action    : {args.action}")
    print(f"  Object IDs: {ids}")
    print(f"  Method    : {args.method}", end="")
    if args.method == "zeroop":
        sigmoid_val = 1.0 / (1.0 + np.exp(-args.zero_logit))
        print(f"  (logit={args.zero_logit:.1f}, "
              f"sigmoid≈{sigmoid_val:.1e})", end="")
    print()
    print(f"  Output    : {output_path}")
    print("=" * W)
    print()

    # ------------------------------------------------------------------
    # Load PLY
    # ------------------------------------------------------------------
    print("  Reading PLY …")
    data = read_explicit_ply(input_path)
    unique_ids = np.unique(data["label"]).tolist()
    K = data["f_rest"].shape[1]
    print(f"  Gaussians loaded  : {len(data['label']):,}")
    print(f"  SH rest bands     : {K}")
    print(f"  Label field       : {'present' if data['has_label'] else 'ABSENT (all zeros)'}")
    print(f"  Unique object IDs : {unique_ids}")
    print()

    # Warn about any requested IDs that are absent
    missing = [i for i in ids if i not in unique_ids]
    if missing:
        print(f"  WARNING: Requested ID(s) {missing} not found in the PLY.\n")

    # ------------------------------------------------------------------
    # Apply edit
    # ------------------------------------------------------------------
    result = edit_point_cloud(
        data, args.action, ids, args.method, zero_logit=args.zero_logit
    )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    print()
    write_explicit_ply(output_path, result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_in  = len(data["label"])
    n_out = len(result["label"])
    print()
    print("=" * W)
    print("  Done.")
    print(f"  Gaussians in  : {n_in:,}")
    print(f"  Gaussians out : {n_out:,}")
    if args.method == "remove":
        print(f"  Δ             : {n_in - n_out:,} removed")
    else:
        n_hidden = int((result["opacity"] == args.zero_logit).sum())
        print(f"  Zeroed        : {n_hidden:,}  (opacity logit = {args.zero_logit:.1f})")
    print("=" * W)


if __name__ == "__main__":
    main()

