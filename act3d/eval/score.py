#!/usr/bin/env python3
"""Score run_eval.py outputs on the six benchmarks.

  python eval/score.py --pools $ACT3D_ROOT/eval --run out/act3d

expects <pools>/<bench>.jsonl and <run>/<bench>.jsonl for bench in BENCHES. Rows with an
`episode_error` (the episode never ran, e.g. a missing mesh) are left out of the denominator.

Free-form answers (ScanQA, SQA3D, MP3D, ARKit) are correct when, after lower-casing, replacing
punctuation by spaces and folding number words to digits, either every content word of an
annotated answer appears in the prediction, one is a whole-word (or >= 3-character word-prefix)
substring of the other, or their token F1 is >= 0.67. SPBench is scored as VSI-Bench: mean
relative accuracy on numeric questions, option accuracy on multiple-choice ones, averaged.
"""
import argparse
import json
import os
import re

BENCHES = [("ScanQA", "scanqa"), ("SQA3D", "sqa3d"), ("SPBench-MV", "spbench_mv"),
           ("SPBench-SI", "spbench_si"), ("MP3D", "mp3d"), ("ARKit", "arkit")]

_STOPW = set("the a an is are was were of in on at to and or it its this that there be been with as for from".split())
_NUMS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen twenty".split()
_FOLD = {w: str(i) for i, w in enumerate(_NUMS)} | {"none": "0", "no one": "0"}
_WORDNUM = {w: i for i, w in enumerate(_NUMS)} | {"none": 0, "no": 0, "a": 1, "an": 1, "single": 1,
                                                    "twenty-one": 21, "twenty-two": 22, "twenty-three": 23}


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())


def fold(s):
    return " ".join(_FOLD.get(t, t) for t in s.split())


def correct(pred, gts) -> bool:
    p = fold(norm(pred))
    if not p:
        return False
    ptok = set(p.split())
    for g in gts:
        content = [t for t in fold(norm(g)).split() if t not in _STOPW]
        if content and all(t in ptok for t in content):
            return True
    for g in gts:
        g = fold(norm(g))
        if not g:
            continue
        if g == p:
            return True
        whole = lambda a, b: re.search(r"(?:^|\s)" + re.escape(a) + r"(?:\s|$)", b)
        if whole(p, g) or whole(g, p):
            return True
        if min(len(p), len(g)) >= 3 and (re.search(r"(?:^|\s)" + re.escape(p), g) or re.search(r"(?:^|\s)" + re.escape(g), p)):
            return True
        pt, gt = set(p.split()), set(g.split())
        if pt and gt and 2 * len(pt & gt) / (len(pt) + len(gt)) >= 0.67:
            return True
    return False


def to_float(s):
    t = str(s).strip().lower().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if m:
        return float(m.group())
    for tok in re.split(r"[^a-z-]+", t):
        if tok in _WORDNUM:
            return float(_WORDNUM[tok])
    return None


def mra(pred, gt) -> float:
    p, g = to_float(pred), to_float(gt)
    if p is None or g is None or g == 0:
        return 0.0
    rel = abs(p - g) / abs(g)
    return sum(rel < 1 - (0.5 + 0.05 * i) for i in range(10)) / 10


def choice_hit(pred, gt, question) -> int:
    """Option letter, or the option's full text."""
    p = re.sub(r"[^a-z0-9 ]", " ", str(pred).lower()).strip()
    g = str(gt).strip().lower()
    if not p:
        return 0
    if p.split()[0] == g:
        return 1
    for line in question.split("\n"):
        m = re.match(r"\s*([A-Za-z])[.)]\s*(.+?)\s*$", line)
        if not m or m.group(1).lower() != g:
            continue
        opt = re.sub(r"\s*Answer (the question|with the option).*$", "", m.group(2), flags=re.I)
        if " ".join(p.split()) == " ".join(re.sub(r"[^a-z0-9 ]", " ", opt.lower()).split()):
            return 1
    return 0


def answers(rec):
    a = rec.get("answers")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            a = [a]
    return [a] if isinstance(a, str) else list(a or [])


def score(pool: dict, preds: dict):
    if not preds:
        return None
    if any("answer_format" in pool[q] for q in preds):  # SPBench
        num, cho = [], []
        for q, p in preds.items():
            r = pool[q]
            if r["answer_format"] == "numeric":
                num.append(mra(p, r["answers"][0]))
            else:
                cho.append(choice_hit(p, r["answers"][0], r["question"]))
        halves = [100 * sum(x) / len(x) for x in (num, cho) if x]
        return sum(halves) / len(halves)
    return 100 * sum(correct(p, answers(pool[q])) for q, p in preds.items()) / len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    cells = []
    for name, tag in BENCHES:
        pool_f, run_f = os.path.join(a.pools, f"{tag}.jsonl"), os.path.join(a.run, f"{tag}.jsonl")
        if not os.path.exists(run_f):
            cells.append((name, None, 0))
            continue
        pool = {r["qid"]: r for r in map(json.loads, filter(str.strip, open(pool_f)))}
        preds = {}
        for r in map(json.loads, filter(str.strip, open(run_f))):
            if r.get("qid") in pool and not (r.get("episode_error") or r.get("error")):
                preds[r["qid"]] = r.get("final_answer") or ""
        cells.append((name, score(pool, preds), len(preds)))
    for name, s, n in cells:
        print(f"{name:<12} {('%.1f' % s) if s is not None else '-':>6}   n={n}")
    if all(s is not None for _, s, _ in cells):
        print(f"{'avg6':<12} {sum(s for _, s, _ in cells) / 6:6.1f}")


if __name__ == "__main__":
    main()
