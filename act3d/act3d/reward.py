"""Verifiable reward (paper Eq. 8-9):

    R = l_judge * R_judge + l_EM * R_EM + l_int * R_judge * c(tau) - l_fmt * R_fmt
    c(tau) = N_max                    if the asked-about object was seen (R_gnd = 1) and n_int >= 1
           = min(n_int, N_max)        otherwise

R_judge: LLM judge (OpenAI-compatible endpoint, Qwen3-30B-A3B-Instruct in the paper).
R_EM:    normalised exact match against any annotated answer.
R_gnd:   an annotated answer box projects, unoccluded, into the start view or any later view.
R_fmt:   malformed output (missing/trivial <think>, or no <answer>).
"""

import json
import logging
import os
import random
import re
import time

import requests

from act3d import env

logger = logging.getLogger(__name__)

W_JUDGE, W_EM, W_FMT, W_INT, N_MAX = 0.3, 0.2, 0.2, 0.05, 12

JUDGE_BASE = os.environ.get("JUDGE_BASE", "http://127.0.0.1:30000/v1").rstrip("/")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "judge")
JUDGE_KEY = os.environ.get("JUDGE_API_KEY", "")

_SCENES = env.SceneCache(max_size=int(os.environ.get("ACT3D_SCENE_CACHE", "8")))
_NONANSWERS = frozenset({"null", "none", "na", "nan", "nil", "nadata", "unknown", "tbd"})
_ACTION_WORDS = set(env.ACTIONS)

_JUDGE_SYSTEM = (
    "You are an expert evaluator for 3D spatial reasoning questions. "
    "Determine whether the model's answer is semantically equivalent to the "
    "ground truth, given the question. "
    "COHERENCE REQUIREMENT: first judge whether the model's answer is coherent, "
    "fluent natural language that actually answers the question. If the answer is "
    "gibberish, word-salad, self-contradictory, contains broken/garbled or non-"
    "English junk tokens, or is a rambling string of unrelated phrases — respond "
    "INCORRECT even if it happens to contain some keywords from the ground truth. "
    "Only a coherent answer that is also semantically equivalent to the ground "
    "truth is CORRECT. "
    "Respond with exactly one word: CORRECT or INCORRECT."
)
_COHERENCE_SYSTEM = (
    "Judge ONLY whether the text is coherent language that reads as a real answer to a "
    "question (IGNORE whether it is correct). Terse fragments and noun phrases are COHERENT "
    "(e.g. 'left of the door', 'light brown', '3 chairs', 'right of tall cabinet') — dataset "
    "answers are often not full sentences. Mark GIBBERISH only for garbled/broken tokens, "
    "repeated-word salad, or strings of unrelated phrases. Reply with exactly one word: "
    "COHERENT or GIBBERISH."
)


class JudgeUnavailable(RuntimeError):
    """Raised after every retry failed. `fatal` makes the reward manager re-raise it, so the step
    fails (resume from the last checkpoint) instead of training on rewards scored 0 by an outage."""

    fatal = True


def _judge(system: str, user: str, temperature: float) -> str:
    headers = {"Content-Type": "application/json"}
    if JUDGE_KEY:
        headers["Authorization"] = f"Bearer {JUDGE_KEY}"
    body = {"model": JUDGE_MODEL, "temperature": temperature, "max_tokens": 64,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(int(os.environ.get("JUDGE_RETRIES", "5"))):
        body["seed"] = random.randint(0, 1_000_000)
        try:
            # bounded so both judge calls of a rollout fit well inside the reward manager timeout
            r = requests.post(f"{JUDGE_BASE}/chat/completions", headers=headers, json=body, timeout=20)
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:
            logger.warning("judge request failed (%s), retrying", exc)
            time.sleep(min(2**attempt, 8))
    raise JudgeUnavailable(f"judge at {JUDGE_BASE} unreachable")


def judge_correct(question: str, ground_truth: str, answer: str) -> float:
    resp = _judge(_JUDGE_SYSTEM,
                  "Evaluate the model answer against the ground truth.\n\n"
                  f"[Question]: {question}\n[Ground Truth]: {ground_truth}\n[Model Answer]: {answer}\n[Judgement]:",
                  temperature=0.1)
    if re.search(r"\bCORRECT\b", resp, re.I):
        return 1.0
    return 0.0


def _coherent(answer: str) -> bool:
    """Gate for grounding credit: garbled answers cannot collect it by navigating to the object."""
    t = answer.strip()
    if len(t) <= 24 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .,'/-]*", t):
        return True
    if len(t) < 2 or len(re.findall(r"[^\x00-\x7F]", t)) / len(t) > 0.10:
        return False
    resp = _judge(_COHERENCE_SYSTEM, f"[Text]: {answer}\n[Judgement]:", temperature=0.0)
    return not re.search(r"\bGIBBERISH\b", resp, re.I)


def _em_norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", s)).strip()


def exact_match(answer: str, ground_truth: str, extra_info: dict) -> float:
    gts = [ground_truth]
    try:
        raw = extra_info.get("answers")
        if raw:
            gts = [str(g) for g in (json.loads(raw) if isinstance(raw, str) else raw)] or gts
    except Exception:
        pass
    a = _em_norm(answer)
    return 1.0 if a and any(a == _em_norm(g) for g in gts) else 0.0


def grounded(solution_str: str, extra_info: dict) -> bool:
    """R_gnd: an answer box is visible in the start view or any view the rollout rendered."""
    boxes = extra_info.get("answer_boxes") or []
    if isinstance(boxes, str):
        try:
            boxes = json.loads(boxes) or []
        except Exception:
            boxes = []
    scene_id = extra_info.get("scene_id")
    if not boxes or not scene_id:
        return False
    try:
        # the row's own mesh (decimated for ScanNet): occlusion tests only, cheaper than the dense one
        scene = _SCENES.get(env.resolve_mesh(extra_info["mesh_path"], dense=False))
    except Exception as exc:
        logger.warning("grounding: scene load failed for %s: %s", scene_id, exc)
        return False
    eye, at = env.start_pose(scene_id, scene.min, scene.max)
    return any(env.box_in_view(e, a, boxes, scene) for e, a in env.trajectory(solution_str, eye, at, scene.min, scene.max))


def _parse(solution_str: str):
    """-> (answer_text, has_answer, format_error)"""
    # Qwen3.5 opens the first assistant turn with "<think>\n" inside the prompt
    if "</think>" in solution_str:
        first_open, first_close = solution_str.find("<think>"), solution_str.find("</think>")
        if first_open == -1 or first_open > first_close:
            solution_str = "<think>" + solution_str
    thinks = re.findall(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    last_think = thinks[-1].strip() if thinks else ""
    has_think = len(last_think) >= 3 and last_think.lower() not in _ACTION_WORDS
    fmt_err = solution_str.count("<think>") != solution_str.count("</think>") or not has_think

    tail = solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
    fmt_err |= tail.count("<answer>") != tail.count("</answer>")
    m = re.search(r"<answer>(.*?)</answer>", tail, re.DOTALL)
    answer = m.group(1).strip() if m else ""
    if len(answer) >= 1000:  # overlong "answers" are a judge-hacking surface
        fmt_err, answer = True, answer[:1000]
    norm = re.sub(r"[^a-z0-9]", "", answer.lower())
    has_answer = m is not None and bool(norm) and norm not in _NONANSWERS
    if not has_answer:
        fmt_err = True
        # a bare final phrase without tags is still judged (format penalty kept)
        bare = _bare_answer(solution_str)
        bn = re.sub(r"[^a-z0-9]", "", bare.lower())
        if bn and bn not in _NONANSWERS and len(bare) <= 120:
            answer, has_answer = bare, True
    return answer, has_answer, fmt_err


def _bare_answer(solution_str: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if m:
        return m.group(1).strip()
    tail = solution_str
    ends = list(re.finditer(r"</tool_response>", tail))
    if ends:
        tail = tail[ends[-1].end():]
    tail = re.sub(r"<tool_call>.*?</tool_call>", "", tail, flags=re.DOTALL)
    tail = re.sub(r"<tool_response>.*?</tool_response>", "", tail, flags=re.DOTALL)
    tail = re.sub(r"<\|im_end\|>|<\|im_start\|>", " ", tail)
    return re.sub(r"\b(user|assistant|system)\b", " ", tail).strip()


_KEYS = ("score", "acc", "em", "fmt_err", "grounded", "tool_bonus", "n_tool_calls")


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    """Always returns the same keys (batches index every sample by one key set); a judge outage
    raises, any other failure scores 0 and is logged."""
    try:
        return _score(solution_str, ground_truth, extra_info or {})
    except JudgeUnavailable:
        raise
    except Exception:
        logger.exception("reward failed; scoring 0")
        return dict.fromkeys(_KEYS, 0.0)


def _score(solution_str: str, ground_truth: str, extra_info: dict) -> dict:
    n_calls = len(env.executed_actions(solution_str))  # calls the environment answered
    answer, has_answer, fmt_err = _parse(solution_str)

    acc = em = 0.0
    gnd = False
    if has_answer:
        gnd = _coherent(answer) and grounded(solution_str, extra_info)
        acc = judge_correct(extra_info.get("question", ""), ground_truth, answer)
        em = exact_match(answer, ground_truth, extra_info)
    c = N_MAX if (gnd and n_calls > 0) else min(n_calls, N_MAX)
    interaction = W_INT * c if acc > 0.5 else 0.0
    score = W_JUDGE * acc + W_EM * em + interaction - W_FMT * float(fmt_err)
    return {"score": score, "acc": acc, "em": em, "fmt_err": float(fmt_err), "grounded": float(gnd),
            "tool_bonus": interaction, "n_tool_calls": float(n_calls)}
