"""Autonomy stack for the mumt project.

Built incrementally over the M-Agent.* milestones. Modules slot in as they
land:

- ``coverage``       -- top-down occupancy + per-Spot last-seen-time map
                        (M-Agent.2, slice B)
- ``perception``     -- HTTP client for the Jetson YOLO server, plus depth
                        back-projection (M-Agent.2, slice D)
- ``bg_perception``  -- per-Spot background perception worker thread
                        (M-Agent.2, slice D)
- ``memory``         -- episodic stream + JSONL persistence
                        (M-Agent.2, slice E)
- ``loop``           -- ReAct LLM agent loop (M-Agent.3+)
- ``tools``          -- LLM-callable tool registry (M-Agent.3+)
"""
