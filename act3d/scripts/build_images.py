#!/usr/bin/env python
"""Render the images for the Act3D-data annotations from your own copies of the scene meshes.

Act3D-data ships text only (questions, answers, trajectories, camera poses). The images are
renders of ScanNet / ARKitScenes / Matterport3D meshes, which their licenses do not let us
redistribute, so this script re-creates them locally with the Act3D renderer (act3d/act3d/env.py).

    export ACT3D_ROOT=/path/to/act3d_data   # Act3D-data download + the mesh folders (dataset README)
    PYTHONPATH=act3d python act3d/scripts/build_images.py --mmscan /path/to/LLaVA-3D_MMScan_QA.json --workers 32

Reads   $ACT3D_ROOT/annotations/{sft_train,sft_val,rl_train}.parquet
        $ACT3D_ROOT/benchmarks/scanqa/ScanQA_v1.0_val.json   (official ScanQA val, for RL validation)
Writes  $ACT3D_ROOT/sft/train.parquet, sft/val.parquet   (messages + images: verl multi-turn SFT format)
        $ACT3D_ROOT/rl/train.parquet,  rl/val.parquet    (prompt + start view + extra_info + reward_model)

SFT images: image 0 is the start view; image k is the view after the k-th tool call, obtained by
replaying the recorded actions from the start pose (env.apply_move_action). With the meshes listed in
the README this reproduces the training images pixel-exactly.
"""
import argparse
import io
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as _mp
_SPAWN = _mp.get_context("spawn")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from act3d import env, prompts
except ImportError:
    sys.exit("cannot import act3d: put the code repo's act3d/ directory on PYTHONPATH (it contains act3d/env.py)")


MMSCAN_PLACEHOLDER = "{MMSCAN_QUESTION}"
MMSCAN_ANSWER = "{MMSCAN_ANSWER}"


def load_mmscan(path):
    """{MMScan id: (question, answer)} for the Matterport3D entries of LLaVA-3D_MMScan_QA.json."""
    import json
    out = {}
    for r in json.load(open(path)):
        if r["video"].startswith("matterport3d/"):
            c = r["conversations"]
            out[r["id"]] = (next(t["value"] for t in c if t["from"] == "human").strip(),
                            next(t["value"] for t in c if t["from"] == "gpt").strip())
    return out


def fill_mmscan_rl(r, mm):
    """MMScan rows ship without question/answer text; restore it from the user's MMScan copy."""
    if r.get("mmscan_id"):
        import json
        q, a = mm[r["mmscan_id"]]
        r["question"], r["answers"], r["ground_truth"] = q, json.dumps([a]), a
    return r


def fill_mmscan_sft(r, mm):
    """{MMSCAN_QUESTION} -> the MMScan question; {MMSCAN_ANSWER} -> the MMScan answer without its final period."""
    if r.get("mmscan_id"):
        q, a = mm[r["mmscan_id"]]
        r["messages"] = [dict(m, content=m["content"].replace(MMSCAN_PLACEHOLDER, q).replace(MMSCAN_ANSWER, a.rstrip(".")))
                         for m in r["messages"]]
    return r


def _png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def sft_poses(row):
    """Start pose followed by the pose after each recorded action."""
    eye, at = np.array(row["start_eye"], float), np.array(row["start_at"], float)
    poses = [(eye, at)]
    for act in row["actions"]:
        nxt = env.apply_move_action(poses[-1][0], poses[-1][1], act, row["bounds_min"], row["bounds_max"])
        poses.append(nxt if nxt is not None else poses[-1])
    return poses


def _render_scene(job):
    mesh_path, items = job
    scene = env.Scene(env.resolve_mesh(mesh_path))
    out = {}
    for key, poses in items:
        out[key] = [_png(scene.render(e, a)) for e, a in poses]
    return out


def render_all(jobs, workers):
    """jobs: {mesh_path: [(key, [(eye, at), ...]), ...]} -> {key: [png bytes, ...]}"""
    res = {}
    with ProcessPoolExecutor(workers, mp_context=_SPAWN) as ex:
        for i, out in enumerate(ex.map(_render_scene, jobs.items(), chunksize=1)):
            res.update(out)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} scenes", flush=True)
    return res


def build_sft(src, dst, workers, mm):
    rows = [fill_mmscan_sft(r, mm) for r in pq.read_table(src).to_pylist()]
    jobs = defaultdict(list)
    for i, r in enumerate(rows):
        jobs[r["mesh_path"]].append((i, sft_poses(r)))
    imgs = render_all(jobs, workers)
    out = [{"messages": r["messages"], "images": [{"bytes": b, "path": None} for b in imgs[i]],
            "qid": r["qid"], "scene_id": r["scene_id"], "src": r["src"]} for i, r in enumerate(rows)]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), dst, row_group_size=64)
    print(f"wrote {dst}: {len(out)} trajectories")


def scanqa_val_rows(path):
    """RL validation = the full official ScanQA v1.0 val split (4,675 questions), each starting from its
    scene's start pose in act3d/act3d/data/start_poses.json."""
    import json
    out = []
    for i, r in enumerate(json.load(open(path))):
        sid = r["scene_id"]
        eye, at = env.start_pose(sid)
        out.append({"index": str(i), "data_source": "scanqa", "scene_id": sid,
                    "mesh_path": f"scannet/scans/{sid}/{sid}_vh_clean_2.ply", "question": r["question"],
                    "answers": json.dumps(list(r["answers"])), "answer_boxes": "[]", "qtype": "",
                    "ground_truth": r["answers"][0], "start_eye": eye.tolist(), "start_at": at.tolist(),
                    "mmscan_id": ""})
    return out


def build_rl(src, dst, workers, mm):
    rows = src if isinstance(src, list) else pq.read_table(src).to_pylist()
    rows = [fill_mmscan_rl(r, mm) for r in rows]
    jobs = defaultdict(list)
    for r in rows:  # one start view per scene, shared by all its questions
        key = (r["mesh_path"], tuple(r["start_eye"]), tuple(r["start_at"]))
        if key not in {k for k, _ in jobs[r["mesh_path"]]}:
            jobs[r["mesh_path"]].append((key, [(np.array(r["start_eye"], float), np.array(r["start_at"], float))]))
    imgs = render_all(jobs, workers)
    out = []
    for r in rows:
        key = (r["mesh_path"], tuple(r["start_eye"]), tuple(r["start_at"]))
        out.append({
            "data_source": r["data_source"],
            "prompt": [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                       {"role": "user", "content": "<image>\n" + prompts.first_user_text(r["scene_id"], r["question"])}],
            "images": [{"bytes": imgs[key][0], "path": None}],
            "ability": "3d_spatial_reasoning",
            "reward_model": {"ground_truth": r["ground_truth"], "style": "rule"},
            "extra_info": {"index": str(r["index"]), "scene_id": r["scene_id"], "mesh_path": r["mesh_path"],
                           "question": r["question"], "answers": r["answers"], "answer_boxes": r["answer_boxes"],
                           "qtype": r["qtype"]},
        })
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), dst, row_group_size=256)
    print(f"wrote {dst}: {len(out)} questions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ACT3D_ROOT"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", choices=["sft", "rl"], default=None)
    ap.add_argument("--mmscan", required=True,
                    help="LLaVA-3D_MMScan_QA.json from the MMScan release (fills in the MP3D questions)")
    a = ap.parse_args()
    if not a.root:
        sys.exit("set ACT3D_ROOT or pass --root")
    mm = load_mmscan(a.mmscan)
    os.environ["ACT3D_ROOT"] = a.root
    ann = os.path.join(a.root, "annotations")
    missing = []
    for split in ("sft_train", "sft_val", "rl_train"):
        for mp in set(pq.read_table(f"{ann}/{split}.parquet", columns=["mesh_path"]).column(0).to_pylist()):
            if not os.path.isfile(env.resolve_mesh(mp)):
                missing.append(mp)
    if missing:
        sys.exit(f"{len(set(missing))} meshes missing under $ACT3D_ROOT, e.g. {sorted(set(missing))[:3]} (see README)")
    if a.only in (None, "sft"):
        build_sft(f"{ann}/sft_train.parquet", f"{a.root}/sft/train.parquet", a.workers, mm)
        build_sft(f"{ann}/sft_val.parquet", f"{a.root}/sft/val.parquet", a.workers, mm)
    if a.only in (None, "rl"):
        build_rl(f"{ann}/rl_train.parquet", f"{a.root}/rl/train.parquet", a.workers, mm)
        val = f"{a.root}/benchmarks/scanqa/ScanQA_v1.0_val.json"
        if not os.path.isfile(val):
            sys.exit(f"RL validation uses the official ScanQA val split: put ScanQA_v1.0_val.json at {val} (README, Evaluate)")
        build_rl(scanqa_val_rows(val), f"{a.root}/rl/val.parquet", a.workers, mm)


if __name__ == "__main__":
    main()
