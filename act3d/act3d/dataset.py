"""RL dataset: one question, one start view, and the scene handle for render_view_tool.

Parquet rows (verl RL format):
  images:       start view(s) as [{"bytes": PNG}]; with several, the second is the start view c0
  extra_info:   scene_id, mesh_path (absolute or relative to $ACT3D_ROOT), question,
                answers (JSON list), answer_boxes (JSON list of [cx, cy, cz, sx, sy, sz]), index
  reward_model: {"ground_truth": str}
The prompt column is ignored: system and user turns are rebuilt from act3d.prompts.
"""

import torch

from verl.utils.dataset.rl_dataset import RLHFDataset

from act3d import prompts


class Act3DDataset(RLHFDataset):
    def __getitem__(self, item):
        row = self.dataframe[item]
        extra = row.get("extra_info") or {}
        images = row.get(self.image_key) or []
        if len(images) >= 2:  # legacy rows carry [bird's-eye view, c0, ...]
            row[self.image_key] = images[1:2]
        row[self.prompt_key] = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": "<image>\n" + prompts.first_user_text(extra["scene_id"], extra["question"])},
        ]
        row["raw_prompt"] = self._build_messages(row)
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        try:
            row["index"] = int(extra.get("index", item))
        except (TypeError, ValueError):
            row["index"] = item
        row["tools_kwargs"] = {
            "render_view_tool": {"create_kwargs": {"scene_id": extra["scene_id"], "mesh_path": extra["mesh_path"]}}
        }
        row["interaction_kwargs"] = {}
        row["agent_name"] = "tool_agent"
        return row
