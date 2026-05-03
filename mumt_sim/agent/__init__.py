"""Autonomy stack for the mumt project.

Built incrementally over the M-Agent.* milestones. Modules slot in as they
land:

- ``coverage``    -- top-down coverage map + 5 m chess-grid sector
                     vocabulary (M-Agent.2, slices B + C)
- ``memory``      -- thread-safe append-only perception-memory table +
                     JSONL persistence (M-Agent.2, slice E)
- ``perception``  -- per-Spot Gemma 4 caption worker that posts rows
                     into ``memory`` (M-Agent.2, slice E)
- ``loop``        -- ReAct LLM agent loop (M-Agent.3+)
- ``tools``       -- LLM-callable tool registry (M-Agent.3+)
"""

from .memory import MemoryRow, MemoryTable, default_jsonl_path
from .perception import CaptionWorker, GemmaClient

__all__ = [
    "MemoryRow",
    "MemoryTable",
    "default_jsonl_path",
    "GemmaClient",
    "CaptionWorker",
]
