#!/usr/bin/env python
"""Crop Matterport3D house meshes into the per-region meshes Act3D-data refers to.

Matterport3D ships whole-house meshes. Each Act3D MP3D question is about one region (room), and
its mesh_path is mp3d/regions/<house>_region<N>.ply: the house mesh
(house_segmentations/<house>.ply) cropped to that region's axis-aligned box from <house>.house,
expanded by 0.15 m so the walls are kept.

    python act3d/scripts/prep_mp3d_regions.py --scans /path/to/matterport/v1/scans --root $ACT3D_ROOT

--scans is the official download, containing <house>/house_segmentations.zip for every house.
Only the regions the Act3D-data annotations use are written (check them against
annotations/meshes.sha256). For the evaluation pools, add --pools $ACT3D_ROOT/eval (after act3d/eval/prepare_eval.py).
"""
import argparse
import os
import shutil
import tempfile
import zipfile

import numpy as np
import open3d as o3d
import pyarrow.parquet as pq

MARGIN = 0.15


def _add(regs, mesh_path):
    if mesh_path and mesh_path.startswith("mp3d/regions/"):
        house, reg = os.path.basename(mesh_path)[: -len(".ply")].rsplit("_region", 1)
        regs.setdefault(house, set()).add(int(reg))


def wanted(root, pools=None):
    """Regions used by the Act3D-data annotations under root, or by the eval pools (*.jsonl, mesh_path)."""
    regs = {}
    if pools:
        import glob
        import json
        for f in glob.glob(os.path.join(pools, "*.jsonl")):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    _add(regs, r.get("mesh_path"))
        return regs
    for split in ("sft_train", "sft_val", "rl_train"):
        p = f"{root}/annotations/{split}.parquet"
        for mp in set(pq.read_table(p, columns=["mesh_path"]).column(0).to_pylist()):
            _add(regs, mp)
    return regs


def house_regions(house_file):
    """<house>.house -> {region index: (lo, hi)} from its 'R' lines."""
    out = {}
    for line in open(house_file):
        if line.startswith("R "):
            f = line.split()
            out[int(f[1])] = (np.array(f[9:12], float), np.array(f[12:15], float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", required=True, help="Matterport3D v1/scans directory")
    ap.add_argument("--root", default=os.environ.get("ACT3D_ROOT"), help="$ACT3D_ROOT (meshes go to <root>/mp3d/regions)")
    ap.add_argument("--pools", default=None, help="eval pools directory, e.g. $ACT3D_ROOT/eval: crop the regions they use instead")
    a = ap.parse_args()
    out_dir = f"{a.root}/mp3d/regions"
    os.makedirs(out_dir, exist_ok=True)
    regs = wanted(a.root, a.pools)
    n = 0
    for house, idx in sorted(regs.items()):
        zp = f"{a.scans}/{house}/house_segmentations.zip"
        if not os.path.isfile(zp):
            print(f"missing {zp}")
            continue
        tmp = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zp) as z:
                ply = next(m for m in z.namelist() if m.endswith(f"{house}.ply"))
                hfile = next(m for m in z.namelist() if m.endswith(f"{house}.house"))
                z.extract(ply, tmp)
                z.extract(hfile, tmp)
            mesh = o3d.io.read_triangle_mesh(os.path.join(tmp, ply))
            boxes = house_regions(os.path.join(tmp, hfile))
            for r in sorted(idx):
                lo, hi = boxes[r]
                sub = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo - MARGIN, hi + MARGIN))
                o3d.io.write_triangle_mesh(f"{out_dir}/{house}_region{r}.ply", sub, write_vertex_colors=True)
                n += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"{house}: {len(idx)} regions")
    print(f"wrote {n} region meshes to {out_dir}")


if __name__ == "__main__":
    main()
