#!/usr/bin/env python3
"""Build the six evaluation pools from the official benchmark releases.

We do not redistribute any benchmark. Download each one from its official source into
$ACT3D_ROOT/benchmarks/ (see the README), then run

    python act3d/eval/prepare_eval.py            # writes $ACT3D_ROOT/eval/{scanqa,sqa3d,spbench_mv,spbench_si,mp3d,arkit}.jsonl

Expected inputs:
    $ACT3D_ROOT/benchmarks/scanqa/ScanQA_v1.0_val.json                      ScanQA v1.0, val split (all 4,675)
    $ACT3D_ROOT/benchmarks/sqa3d/v1_balanced_questions_test_scannetv2.json  SQA3D, test split (all 3,519; the situation
                                                                            phrasing per question is data/sqa3d_situation_index.json)
    $ACT3D_ROOT/benchmarks/sqa3d/v1_balanced_sqa_annotations_test_scannetv2.json
    $ACT3D_ROOT/benchmarks/spbench/SPBench-MV.parquet, SPBench-SI.parquet   SPBench, test split (all 319 / 1,009)
    $ACT3D_ROOT/benchmarks/mmscan/LLaVA-3D_MMScan_QA.json                   MMScan; our 1,000 MP3D questions are
                                                                            listed by MMScan id in data/mp3d_mmscan_ids.json
The ARKit pool is our own question set, published in Act3D-data as annotations/arkit_eval.jsonl
(downloaded to $ACT3D_ROOT/annotations/); only its scenes come from ARKitScenes.

Every pool row: qid, official_qid, scene_id, question, answers (list), mesh_path (relative to
$ACT3D_ROOT, resolved by act3d.env.resolve_mesh), plus question_type/answer_format for SPBench.
"""
import argparse
import json
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SPBENCH_NUMERIC = {"object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"}


def scannet_mesh(scene):
    return f"scannet/scans/{scene}/{scene}_vh_clean_2.ply"


def scanqa(b):
    rows = json.load(open(f"{b}/scanqa/ScanQA_v1.0_val.json"))
    return [{"qid": f"val{i}", "official_qid": r["question_id"], "scene_id": r["scene_id"],
             "question": r["question"], "answers": list(r["answers"]), "mesh_path": scannet_mesh(r["scene_id"]),
             "source": "scannet", "domain": "ScanNet"} for i, r in enumerate(rows)]


def sqa3d(b):
    qs = json.load(open(f"{b}/sqa3d/v1_balanced_questions_test_scannetv2.json"))["questions"]
    ann = {a["question_id"]: a for a in json.load(open(f"{b}/sqa3d/v1_balanced_sqa_annotations_test_scannetv2.json"))["annotations"]}
    pick = json.load(open(f"{DATA}/sqa3d_situation_index.json"))["index"]  # the phrasing our results used
    assert len(pick) == len(qs), "unexpected SQA3D test file"
    out = []
    for i, q in enumerate(qs):
        a = ann[q["question_id"]]
        situation = ([q["situation"]] + list(q.get("alternative_situation") or []))[pick[i]]
        out.append({"qid": f"sqa{i}", "official_qid": str(q["question_id"]), "scene_id": q["scene_id"],
                    "question": situation + " " + q["question"], "answers": [a["answers"][0]["answer"]],
                    "mesh_path": scannet_mesh(q["scene_id"]), "source": "scannet", "domain": "SQA3D"})
    return out


def spbench(b, cfg, tag):
    import pyarrow.parquet as pq
    out = []
    for i, r in enumerate(pq.read_table(f"{b}/spbench/{cfg}.parquet").to_pylist()):
        q = r["question"].strip()
        opts = r.get("options")
        if opts is not None and len(opts):
            q += "\n" + "\n".join(opts) + " Answer with the option's letter from the given choices directly."
        else:
            q += " Answer the question using a single number."
        scene = str(r["scene_name"])
        out.append({"qid": f"{tag}_{i}", "official_qid": str(r.get("id", i)), "scene_id": scene, "question": q,
                    "answers": [str(r["ground_truth"])], "mesh_path": scannet_mesh(scene), "source": "scannet",
                    "domain": "SPBench", "question_type": r["question_type"],
                    "answer_format": "numeric" if r["question_type"] in SPBENCH_NUMERIC else "multiple"})
    return out


def mp3d(b):
    mm = {}
    for r in json.load(open(f"{b}/mmscan/LLaVA-3D_MMScan_QA.json")):
        if r["video"].startswith("matterport3d/"):
            c = r["conversations"]
            mm[r["id"]] = (next(t["value"] for t in c if t["from"] == "human").strip(),
                           next(t["value"] for t in c if t["from"] == "gpt").strip())
    out = []
    for r in json.load(open(f"{DATA}/mp3d_mmscan_ids.json")):
        q, a = mm[r["mmscan_id"]]
        out.append({"qid": r["qid"], "official_qid": r["mmscan_id"], "scene_id": r["scene_id"], "question": q,
                    "answers": [a], "mesh_path": f"mp3d/regions/{r['scene_id']}.ply", "source": "mp3d", "domain": "MP3D"})
    return out


def arkit(b):
    # our own questions, shipped with the training annotations in Act3D-data
    return [dict(json.loads(l), official_qid="") for l in open(f"{b}/../annotations/arkit_eval.jsonl") if l.strip()]


BUILDERS = {"scanqa": scanqa, "sqa3d": sqa3d,
            "spbench_mv": lambda b: spbench(b, "SPBench-MV", "spbench_mv"),
            "spbench_si": lambda b: spbench(b, "SPBench-SI", "spbench_si"),
            "mp3d": mp3d, "arkit": arkit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ACT3D_ROOT"))
    ap.add_argument("--only", nargs="*", choices=list(BUILDERS), default=list(BUILDERS))
    a = ap.parse_args()
    if not a.root:
        sys.exit("set ACT3D_ROOT or pass --root")
    os.makedirs(f"{a.root}/eval", exist_ok=True)
    for name in a.only:
        try:
            rows = BUILDERS[name](f"{a.root}/benchmarks")
        except FileNotFoundError as e:
            sys.exit(f"{name}: {e.filename} not found; see the README, Evaluate (official benchmarks; ARKit comes with Act3D-data)")
        def present(mp):  # the renderer also accepts the dense ScanNet mesh alone
            p = os.path.join(a.root, mp)
            return os.path.exists(p) or (mp.endswith("_vh_clean_2.ply") and os.path.exists(p[: -len("_2.ply")] + ".ply"))
        missing = sorted({r["mesh_path"] for r in rows if not present(r["mesh_path"])})
        with open(f"{a.root}/eval/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:11s} {len(rows):5d} questions, {len({r['scene_id'] for r in rows}):3d} scenes"
              + (f"  ({len(missing)} meshes not yet under $ACT3D_ROOT, e.g. {missing[0]})" if missing else ""))


if __name__ == "__main__":
    main()
