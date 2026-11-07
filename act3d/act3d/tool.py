"""render_view_tool: one relative camera action per call, answered with the rendered view.

One instance lives for a whole rollout (the agent loop reuses it across turns), so the camera pose
carries over from call to call. The instance starts at the pose of the image shown in the prompt.
"""

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from act3d import env, prompts

logger = logging.getLogger(__name__)

_SCENES = env.SceneCache(max_size=int(os.environ.get("ACT3D_SCENE_CACHE", "8")))


class RenderViewTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        kw = {**kwargs, **kwargs.get("create_kwargs", {})}
        mesh_path = env.resolve_mesh(kw["mesh_path"])
        # mesh load + BVH build takes seconds: keep it off the agent loop's event loop
        scene = await asyncio.to_thread(_SCENES.get, mesh_path)
        eye, at = env.start_pose(kw["scene_id"], scene.min, scene.max)
        self._instances[instance_id] = {"scene": scene, "eye": eye, "at": at}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        d = self._instances.get(instance_id)
        if d is None:
            return ToolResponse(text="Error: tool instance not found."), -0.05, {"success": False}
        if parameters.get("action") is None:
            return ToolResponse(text="Error: missing 'action'."), -0.05, {"success": False}
        action = env.action_with_distance(parameters["action"], parameters.get("distance"))
        return await asyncio.to_thread(self._move, d, action)

    def _move(self, d: dict, action: str):
        scene = d["scene"]
        res = env.apply_move_action(d["eye"], d["at"], action, scene.min, scene.max)
        if res is None:
            return (ToolResponse(text=f"Unknown action '{action}'. Use one of: {', '.join(env.ACTIONS)}."),
                    -0.05, {"success": False})
        d["eye"], d["at"] = res
        try:
            img = scene.render(d["eye"], d["at"])
        except Exception as exc:
            return ToolResponse(text=f"Render failed: {exc}"), -0.05, {"success": False}
        # leading "\n": the SFT tool turns read "<image>\n" + observation
        text = "\n" + prompts.observation(action, d["eye"], d["at"])
        return ToolResponse(image=[img], text=text), 0.0, {"success": True, "coverage_pct": env.coverage_pct(img)}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
