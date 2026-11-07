"""Prompt and observation text. SFT data, RL rollouts and evaluation must show the model exactly
these strings."""

from act3d import env

SYSTEM_PROMPT = (
    "You are a 3D spatial reasoning assistant exploring an indoor scene. You are given ONE image: "
    "your current camera view — this is your starting position. To see more of the scene, call "
    "render_view_tool with a single movement action, and optionally a distance. Actions: forward, "
    "backward, left, right, up, down, turn_left, turn_right, turn_up, turn_down. If you do not give "
    "a distance, the camera takes its default step — about 1 metre forward or backward, about 0.6 "
    "metres for the other moves, and about 30 degrees for turns. Give a distance in metres (or "
    "degrees for turns) when a larger or smaller step is more efficient — one 4 m move is better "
    "than four small ones. You will see the new view after each action and can then decide to "
    "move again or to answer. Only answer when confident; if a view is empty or low-coverage, move "
    "to a better viewpoint first. Respond in exactly one of these two formats:\n"
    "<think>your reasoning here</think>\n"
    "<tool_call>\n"
    '{"name": "render_view_tool", "arguments": {"action": "<action>", "distance": <optional number>}}\n'
    "</tool_call>\n"
    "or\n"
    "<think>your reasoning here</think>\n"
    "<answer>your final answer</answer>"
)

SHORT_ANSWER = (
    "Keep the text inside the <answer> tag to a SHORT PHRASE of 1-5 words — "
    "no full sentence, no explanation. Still reason inside <think> first."
)


def first_user_text(scene_id: str, question: str) -> str:
    """Turn-0 text (follows the start image): room extent, question, answer-length limit."""
    b = env.scene_bounds(scene_id)
    dims = ""
    if b:
        mn = [round(v, 2) for v in b["min"]]
        mx = [round(v, 2) for v in b["max"]]
        dims = (f" The room measures roughly {round(mx[0] - mn[0], 1)} m x {round(mx[1] - mn[1], 1)} m, "
                f"with about {round(mx[2] - mn[2], 1)} m of height.")
    return (f"This image is your current camera view — your starting position.{dims} Move through it "
            f"with render_view_tool to gather evidence, then answer.\n\nQuestion: {question}\n\n" + SHORT_ANSWER)


def observation(action: str, eye, at) -> str:
    """Text that follows each rendered view in a tool turn."""
    ep = [round(float(x), 2) for x in eye]
    ap = [round(float(x), 2) for x in at]
    return f"Action '{action}'. Camera now at {ep} looking toward {ap}."
