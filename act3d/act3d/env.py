"""The 3D environment shared by RL rollouts, the reward and evaluation.

A scene is a triangle mesh (ScanNet / ARKitScenes / Matterport3D). The camera is a pose
(eye, at) with the up vector held at +Z; observations are 640x480 ray-cast renders of the mesh's
vertex colours. Actions are relative moves/turns with an optional magnitude (paper Sec. 3.1):

    forward/backward 1 m, left/right/up/down 0.6 m, turn_{left,right,up,down} 30 deg,
    times an integer multiplier n = clip(round(distance / unit), 1, 8),

and the camera is clamped to the scene's bounding box enlarged by 1 m (no collision checking).
"""

import json
import math
import os
import re
import threading

import numpy as np
import open3d as o3d
import open3d.core as o3c
from PIL import Image

RENDER_W, RENDER_H = 640, 480
FOCAL = 500.0 * RENDER_W / 640
K = np.array([[FOCAL, 0, RENDER_W / 2], [0, FOCAL, RENDER_H / 2], [0, 0, 1]], dtype=np.float32)
UP = np.array([0.0, 0.0, 1.0])

MOVE_STEP = 0.6  # metres: left/right/up/down
MOVE_STEP_FB = 1.0  # metres: forward/backward
TURN_RAD = 0.5236  # 30 deg (the value every run used)
TURN_DEG = 30.0  # unit for folding a requested angle into a multiplier
MAX_MULT = 8

ACTIONS = ("forward", "backward", "left", "right", "up", "down", "turn_left", "turn_right", "turn_up", "turn_down")

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_META: dict = {}


def _meta(name: str) -> dict:
    if name not in _META:
        with open(os.path.join(_DATA_DIR, name)) as f:
            _META[name] = json.load(f)
    return _META[name]


def scene_bounds(scene_id: str):
    """{'min': [x,y,z], 'max': [x,y,z]} of the scene (metres), or None."""
    return _meta("scene_bounds.json").get(scene_id)


def start_pose(scene_id: str, mn=None, mx=None):
    """(eye, at) of the scene's initial view c0. Scenes without a recorded start view fall back to
    a camera near the -y wall looking at the centre of the given bounds (None if not given)."""
    poses = _meta("start_poses.json").get(scene_id) or []
    if poses:
        return np.array(poses[0]["camera_position"], float), np.array(poses[0]["look_at"], float)
    if mn is None:
        return None
    mn, mx = np.asarray(mn, float), np.asarray(mx, float)
    c = (mn + mx) / 2.0
    return np.array([c[0], mn[1] + 0.2 * (mx[1] - mn[1]), c[2]]), c


def resolve_mesh(mesh_path: str, dense: bool = True) -> str:
    """Absolute mesh path: relative paths resolve against $ACT3D_ROOT. With `dense`, a ScanNet
    decimated mesh (_vh_clean_2.ply) is upgraded to the dense _vh_clean.ply when present (vertex
    density is the texture resolution of a vertex-coloured render)."""
    if mesh_path and not os.path.isabs(mesh_path):
        mesh_path = os.path.join(os.environ.get("ACT3D_ROOT", ""), mesh_path)
    if dense and mesh_path.endswith("_vh_clean_2.ply"):
        dense = mesh_path[: -len("_2.ply")] + ".ply"
        if os.path.isfile(dense):
            return dense
    return mesh_path


# ---------------------------------------------------------------------------------------------
# Scenes and rendering
# ---------------------------------------------------------------------------------------------
class Scene:
    def __init__(self, mesh_path: str):
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"mesh not found: {mesh_path!r} (set ACT3D_ROOT)")
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not self.mesh.has_vertex_colors():
            self.mesh.paint_uniform_color([0.7, 0.7, 0.7])
        self.raycast = o3d.t.geometry.RaycastingScene()
        self.raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        verts = np.asarray(self.mesh.vertices)
        self.min, self.max = verts.min(0), verts.max(0)
        self._colors = np.asarray(self.mesh.vertex_colors)
        self._tris = np.asarray(self.mesh.triangles)

    def render(self, eye, at) -> Image.Image:
        rays = self.raycast.create_rays_pinhole(
            intrinsic_matrix=o3c.Tensor(K, dtype=o3c.float32),
            extrinsic_matrix=o3c.Tensor(look_at_extrinsic(eye, at), dtype=o3c.float32),
            width_px=RENDER_W,
            height_px=RENDER_H,
        )
        hit = self.raycast.cast_rays(rays)
        prim = hit["primitive_ids"].numpy().reshape(RENDER_H, RENDER_W)
        ys, xs = np.where(np.isfinite(hit["t_hit"].numpy()).reshape(RENDER_H, RENDER_W))
        img = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
        tri = prim[ys, xs]
        ok = tri < len(self._tris)
        ys, xs, tri = ys[ok], xs[ok], tri[ok]
        if tri.size:
            img[ys, xs] = (self._colors[self._tris[tri]].mean(axis=1) * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def visible(self, eye, point, slack: float = 0.30) -> bool:
        """No surface between the camera and `point` (a surface at/behind the point is fine)."""
        d = np.asarray(point, float) - np.asarray(eye, float)
        dd = float(np.linalg.norm(d))
        if dd < 1e-6:
            return True
        ray = o3c.Tensor([[*np.asarray(eye, float).tolist(), *(d / dd).tolist()]], dtype=o3c.float32)
        t_hit = float(self.raycast.cast_rays(ray)["t_hit"].numpy()[0])
        return (not np.isfinite(t_hit)) or t_hit >= dd - slack


class SceneCache:
    """Thread-safe LRU of loaded scenes (each holds a mesh + BVH, 100-300 MB)."""

    def __init__(self, max_size: int = 8):
        self.max_size, self._d, self._lock = max_size, {}, threading.Lock()

    def get(self, mesh_path: str) -> Scene:
        with self._lock:
            if mesh_path in self._d:
                self._d[mesh_path] = self._d.pop(mesh_path)
            else:
                self._d[mesh_path] = Scene(mesh_path)
                while len(self._d) > self.max_size:
                    self._d.pop(next(iter(self._d)))
            return self._d[mesh_path]


def look_at_extrinsic(eye, at, up=UP) -> np.ndarray:
    """4x4 world->camera matrix (OpenCV: x right, y down, z forward), horizon kept level."""
    eye, at, up = (np.asarray(v, dtype=np.float64) for v in (eye, at, up))
    z = at - eye
    zn = np.linalg.norm(z)
    z = z / zn if zn > 1e-8 else np.array([0.0, 0.0, 1.0])
    up_proj = up - np.dot(up, z) * z
    if np.linalg.norm(up_proj) < 1e-8:  # looking straight up/down
        for fb in ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]):
            fb = np.array(fb)
            up_proj = fb - np.dot(fb, z) * z
            if np.linalg.norm(up_proj) > 1e-8:
                break
    y = -up_proj / np.linalg.norm(up_proj)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return np.linalg.inv(c2w).astype(np.float32)


def coverage_pct(img: Image.Image) -> int:
    """Percent of pixels that hit scene geometry."""
    return int(100.0 * (np.asarray(img).sum(axis=2) > 0).sum() / (RENDER_W * RENDER_H))


# ---------------------------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------------------------
def action_with_distance(action: str, distance=None) -> str:
    """Fold an optional distance (metres, degrees for turns) into the `+N` multiplier label the
    observation reports, e.g. ("turn_left", 90) -> "turn_left+3"."""
    a = str(action).lower().strip()
    if distance is None or "+" in a:
        return a
    try:
        dv = float(distance)
    except (TypeError, ValueError):
        return a
    unit = TURN_DEG if a.startswith("turn") else (MOVE_STEP_FB if a in ("forward", "backward") else MOVE_STEP)
    n = max(1, min(MAX_MULT, int(round(dv / unit))))
    return f"{a}+{n}" if n > 1 else a


_ALIASES = {
    "move_forward": "forward", "forward_movement": "forward",
    "back": "backward", "move_backward": "backward", "backward_movement": "backward",
    "left_movement": "left", "right_movement": "right", "upward": "up", "downward": "down",
    "left_rotation": "turn_left", "rotate_left": "turn_left",
    "right_rotation": "turn_right", "rotate_right": "turn_right",
    "look_up": "turn_up", "up_rotation": "turn_up", "look_down": "turn_down", "down_rotation": "turn_down",
}

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _parse_call(raw: str):
    """One <tool_call> body as the rollout's Hermes parser reads it (first JSON object), or None."""
    s = raw.strip()
    try:
        call = json.loads(s)
    except json.JSONDecodeError:
        try:
            call, _ = json.JSONDecoder().raw_decode(s[s.index("{"):])
        except ValueError:
            return None
    return call if isinstance(call, dict) and "name" in call and "arguments" in call else None


def executed_actions(text: str) -> list[str]:
    """The actions the tool actually executed in a decoded rollout, labelled as it applied them.

    A turn's call ran only if a <tool_response> follows it, and the agent loop runs only the
    first parsed call of a turn, so JSON inside reasoning or extra calls are never replayed."""
    actions = []
    for piece in text.split("<tool_response>")[:-1]:
        turn = piece.rsplit("</tool_response>", 1)[-1]
        for raw in _TOOL_CALL_RE.findall(turn):
            call = _parse_call(raw)
            if call is None:
                continue
            args = call["arguments"]
            if call["name"] == "render_view_tool" and isinstance(args, dict) and args.get("action") is not None:
                actions.append(action_with_distance(args["action"], args.get("distance")))
            break
    return actions


def apply_move_action(eye, at, action: str, mn, mx):
    """One relative action -> new (eye, at), or None for an unknown action (paper Eq. 1)."""
    eye = np.array(eye, dtype=float)
    at = np.array(at, dtype=float)
    fwd = at - eye
    dist = float(np.linalg.norm(fwd)) or 1.0
    fwd = fwd / dist
    right = np.cross(fwd, UP)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])

    def rot(v, axis, ang):
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * float(np.dot(axis, v)) * (1 - math.cos(ang))

    a = action.replace("-", "_").replace(" ", "_").lower()
    m = re.search(r"\+?(\d+)\s*$", a)
    n = max(1, min(MAX_MULT, int(m.group(1)))) if m else 1
    a = re.sub(r"\+?\d+\s*$", "", a).strip("_")
    a = _ALIASES.get(a, a)
    if a == "forward":
        eye += fwd * MOVE_STEP_FB * n; at += fwd * MOVE_STEP_FB * n
    elif a == "backward":
        eye -= fwd * MOVE_STEP_FB * n; at -= fwd * MOVE_STEP_FB * n
    elif a == "left":
        eye -= right * MOVE_STEP * n; at -= right * MOVE_STEP * n
    elif a == "right":
        eye += right * MOVE_STEP * n; at += right * MOVE_STEP * n
    elif a == "up":
        eye[2] += MOVE_STEP * n; at[2] += MOVE_STEP * n
    elif a == "down":
        eye[2] -= MOVE_STEP * n; at[2] -= MOVE_STEP * n
    elif a == "turn_left":
        at = eye + rot(fwd, UP, TURN_RAD * n) * dist
    elif a == "turn_right":
        at = eye + rot(fwd, UP, -TURN_RAD * n) * dist
    elif a == "turn_up":
        at = eye + rot(fwd, right, TURN_RAD * n) * dist
    elif a == "turn_down":
        at = eye + rot(fwd, right, -TURN_RAD * n) * dist
    else:
        return None
    # clamp to the 1 m-enlarged bounding box, moving `at` with `eye` so the heading is unchanged
    clipped = np.clip(eye, np.asarray(mn, float) - 1.0, np.asarray(mx, float) + 1.0)
    return clipped, at + (clipped - eye)


def trajectory(text: str, eye, at, mn, mx) -> list[tuple]:
    """Start pose followed by the pose after each executed, valid action in `text`."""
    poses = [(np.asarray(eye, float), np.asarray(at, float))]
    for act in executed_actions(text):
        r = apply_move_action(poses[-1][0], poses[-1][1], act, mn, mx)
        if r is not None:
            poses.append(r)
    return poses


def box_in_view(eye, at, boxes, scene: "Scene | None" = None) -> bool:
    """Is any annotated box (cx, cy, cz, sx, sy, sz) visible from (eye, at)? A box counts when its
    centre or a corner projects inside the image and is not occluded by the mesh."""
    E = np.asarray(look_at_extrinsic(eye, at), float)
    for b in boxes or []:
        if len(b) < 6:
            continue
        ctr, half = np.array(b[:3], float), np.abs(np.array(b[3:6], float)) / 2.0
        pts = [ctr] + [ctr + np.array([sx, sy, sz]) * half for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        for p in pts:
            pc = E @ np.array([*p, 1.0])
            if pc[2] <= 0.05:
                continue
            u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
            if not (0 <= u < RENDER_W and 0 <= v < RENDER_H):
                continue
            if scene is None or scene.visible(eye, p):
                return True
    return False
