#!/usr/bin/env python3
"""Interactive evaluation: drive a model served behind an OpenAI-compatible endpoint (vLLM)
through the 3D environment, one question per episode (paper Sec. 4.1: one 640x480 start view,
at most twelve interactions).

  python eval/run_eval.py --pool $ACT3D_ROOT/eval/scanqa.jsonl --out out/scanqa.jsonl \
      --base http://127.0.0.1:31000/v1 --model act3d

Pool rows: {"qid", "scene_id", "question", "answers": [...], "mesh_path"} (+ "question_type",
"answer_format" for SPBench). Output rows carry "final_answer" and the full turn log; score them
with eval/score.py. The script appends and skips questions already in --out, so an interrupted run
can simply be restarted.
"""
import argparse
import base64
import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from act3d import env, prompts

ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
MOVE_RE = re.compile(r'"action"\s*:\s*"([a-zA-Z_+0-9\- ]+)"')
DIST_RE = re.compile(r'"distance"\s*:\s*([0-9.]+)')
FORCE_TAIL = "Keep the text inside the <answer> tag to a SHORT PHRASE of 1-5 words - no sentence, no explanation."
INVALID_LIMIT = 2  # consecutive outputs that are neither a tool call nor an answer before forcing one


def b64png(img) -> str:
    bio = io.BytesIO()
    img.save(bio, "PNG")
    return base64.b64encode(bio.getvalue()).decode()


class Client:
    def __init__(self, base, model, temperature, max_tokens):
        self.base, self.model, self.temperature, self.max_tokens = base.rstrip("/"), model, temperature, max_tokens

    def __call__(self, contents) -> str:
        """contents: [{"role": "user"|"assistant", "parts": [...], "obs": bool}]. Observations are
        sent as user turns wrapped in <tool_response>, which the Qwen3.5 template renders like the
        tool turns of the training data (earlier <think> blocks stay in the history)."""
        msgs = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
        for c in contents:
            parts = [{"type": "text", "text": p["text"]} if "text" in p else
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64," + p["png"]}} for p in c["parts"]]
            if c["role"] == "assistant":
                msgs.append({"role": "assistant", "content": " ".join(p["text"] for p in parts if "text" in p)})
                continue
            if c.get("obs"):
                parts = [{"type": "text", "text": "<tool_response>\n"}] + parts
                parts[-1] = {"type": "text", "text": parts[-1]["text"] + "\n</tool_response>"}
            msgs.append({"role": "user", "content": parts})
        r = requests.post(f"{self.base}/chat/completions", timeout=300, json={
            "model": self.model, "messages": msgs, "max_tokens": self.max_tokens, "temperature": self.temperature})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


def clean_forced_answer(txt: str) -> str:
    """The answer in a forced continuation, or "" when the model kept reasoning / calling tools."""
    t = (txt or "").strip()
    if "</think>" in t.lower():
        t = re.split(r"</\s*think\s*>", t, flags=re.I)[-1].strip()
    m = re.search(r"<\s*(?:think|/think|tool_call|answer|/answer)\b", t, re.I)
    if m:
        t = t[: m.start()]
    t = t.split("\n")[0].strip().strip('"').strip()
    return "" if not t or len(t.split()) > 12 else t


def episode(scene, rec, client, max_turns):
    sid = rec["scene_id"]
    b = env.scene_bounds(sid)
    eye, at = env.start_pose(sid)
    contents = [{"role": "user", "parts": [{"png": b64png(scene.render(eye, at))},
                                          {"text": prompts.first_user_text(sid, rec["question"])}]}]
    turns, invalid, n_invalid, final = [], [], 0, None

    def force_final(out, nudge):
        contents.append({"role": "assistant", "parts": [{"text": out}]})
        contents.append({"role": "user", "parts": [{"text": nudge}]})
        contents.append({"role": "assistant", "parts": [{"text": "<answer>"}]})
        completion = client(contents)
        out2 = "<answer>" + completion
        turns.append({"eye": list(eye), "at": list(at), "action": "<answer>", "model_text": out2, "forced": True})
        tagged = ANS_RE.findall(completion)  # the model may still write a complete <answer>...</answer>
        if tagged:
            return tagged[-1].strip() or None
        a2 = ANS_RE.findall(out2 + "</answer>") or ANS_RE.findall(out2)
        return (clean_forced_answer(a2[-1]) or None) if a2 else None

    for _ in range(max_turns + 2):
        out = client(contents)
        if not out:
            invalid.append({"turn": len(turns), "why": "empty"})
            break
        answers, move = ANS_RE.findall(out), MOVE_RE.search(out)
        if answers:
            final = answers[-1].strip()
            turns.append({"eye": list(eye), "at": list(at), "action": "<answer>", "model_text": out})
            break
        if len(turns) >= max_turns:
            final = force_final(out, "You have used all render calls. Do NOT call the tool. Give your final answer "
                                     "as <answer>...</answer>. " + FORCE_TAIL)
            break
        if not move:
            invalid.append({"turn": len(turns), "text": out[-600:], "why": "no_call_no_answer"})
            n_invalid += 1
            if n_invalid >= INVALID_LIMIT:
                final = force_final(out, "Respond ONLY as a <tool_call> or a final <answer>. Give your final answer "
                                         "now as <answer>...</answer>. " + FORCE_TAIL)
                break
            contents.append({"role": "assistant", "parts": [{"text": out}]})
            contents.append({"role": "user", "parts": [{"text": "Respond ONLY as a <tool_call> or a final <answer>."}]})
            continue
        n_invalid = 0
        dm = DIST_RE.search(out)
        try:
            dist = float(dm.group(1)) if dm else None
        except ValueError:  # e.g. "1.5." -> default step
            dist = None
        act = env.action_with_distance(move.group(1).strip().lower(), dist)
        r = env.apply_move_action(eye, at, act, b["min"], b["max"])
        if r is None:
            contents.append({"role": "assistant", "parts": [{"text": out}]})
            contents.append({"role": "user", "parts": [{"text": f"Unknown action '{act}'. Use one of: {', '.join(env.ACTIONS)}."}]})
            continue
        turns.append({"eye": list(eye), "at": list(at), "action": act, "model_text": out})
        eye, at = [float(v) for v in r[0]], [float(v) for v in r[1]]
        contents.append({"role": "assistant", "parts": [{"text": out}]})
        contents.append({"role": "user", "obs": True, "parts": [{"png": b64png(scene.render(eye, at))},
                                                               {"text": prompts.observation(act, eye, at)}]})
    return {"qid": rec["qid"], "scene_id": sid, "question": rec["question"], "gt_answers": rec["answers"],
            "final_answer": final, "turns": turns, "invalid": invalid,
            "n_tool_calls": sum(t["action"] != "<answer>" for t in turns)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True, help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:31000/v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--scenes-parallel", type=int, default=10)
    ap.add_argument("--per-scene", type=int, default=3)
    args = ap.parse_args()

    client = Client(args.base, args.model, args.temperature, args.max_tokens)
    recs = [json.loads(line) for line in open(args.pool) if line.strip()]
    try:  # resume: keep finished episodes; rerun the rest, including failed ones (the scorer keeps the last row)
        done = {r["qid"] for r in map(json.loads, filter(str.strip, open(args.out)))
                if not (r.get("episode_error") or r.get("failed"))}
    except FileNotFoundError:
        done = set()
    recs = [r for r in recs if r["qid"] not in done]
    by_scene: dict = {}
    for r in recs:
        by_scene.setdefault((r["scene_id"], r["mesh_path"]), []).append(r)
    # render the dense mesh, as the RL tool does; a ScanNet scene with only the decimated copy is an error
    no_dense = sorted({m for _, m in by_scene if env.resolve_mesh(m).endswith("_vh_clean_2.ply")})
    if no_dense:
        raise SystemExit(f"{len(no_dense)} ScanNet scenes have no dense _vh_clean.ply next to the pool's mesh, "
                         f"e.g. {no_dense[:3]}")
    lock = threading.Lock()
    out_f = open(args.out, "a")

    def write(row):
        with lock:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

    def run_scene(item):
        (sid, mesh), rows = item
        try:
            scene = env.Scene(env.resolve_mesh(mesh))
        except Exception as exc:
            for r in rows:
                write({"qid": r["qid"], "final_answer": "", "episode_error": f"scene load: {exc}"})
            return

        def one(rec):
            try:
                write(episode(scene, rec, client, args.max_turns))
            except Exception as exc:  # a failed episode is scored as a wrong answer, never dropped
                write({"qid": rec["qid"], "final_answer": "", "turns": [], "n_tool_calls": 0,
                       "failed": f"{type(exc).__name__}: {str(exc)[:120]}"})

        with ThreadPoolExecutor(args.per_scene) as ex:
            list(ex.map(one, rows))

    with ThreadPoolExecutor(args.scenes_parallel) as ex:
        list(ex.map(run_scene, by_scene.items()))
    print(f"done: {len(recs)} questions ({len(done)} already in {args.out})")


if __name__ == "__main__":
    main()
