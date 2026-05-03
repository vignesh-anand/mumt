"""Autonomy stack for the mumt project.

Built incrementally over the M-Agent.* milestones. Modules slot in as they
land:

- ``coverage``    -- top-down coverage map + 5 m chess-grid sector
                     vocabulary (M-Agent.2, slices B + C)
- ``memory``      -- thread-safe append-only perception-memory table +
                     JSONL persistence (M-Agent.2, slice E)
- ``perception``  -- per-Spot Gemma 4 caption worker that posts rows
                     into ``memory`` (M-Agent.2, slice E)
- ``detection``   -- HTTP client + thread-pool wrapper for the Jetson
                     YOLOE server, used by the find-label primitive
                     (M-Agent.2, slice G)
- ``tools``       -- atomic action primitives (`goto`, `move`,
                     `search`, `find`) + `Controller` base for
                     step-able autonomy (M-Agent.2, slice F-G)
- ``loop``        -- ReAct LLM agent loop (M-Agent.3+)
"""

from .detection import (
    Detection,
    DetectionResponse,
    OnDemandDetector,
    YoloeClient,
)
from .memory import MemoryRow, MemoryTable, default_jsonl_path
from .perception import (
    AMBIENT_CAPTION_PROMPT,
    SEARCH_VIEWPOINT_PROMPT,
    CaptionWorker,
    GemmaClient,
    OnDemandCaptioner,
    SearchViewpointCaption,
    parse_ambient_caption,
    parse_search_caption,
)
from .tools import (
    Controller,
    ControllerCtx,
    FindLabelConfig,
    FindLabelController,
    FindObservation,
    FindResult,
    GotoConfig,
    GotoController,
    MoveConfig,
    MoveController,
    PrimitiveResult,
    SearchObservation,
    SearchResult,
    SearchSectorConfig,
    SearchSectorController,
    resolve_goto_target,
)
from .visibility import Viewpoint, plan_search_tour

__all__ = [
    "MemoryRow",
    "MemoryTable",
    "default_jsonl_path",
    "AMBIENT_CAPTION_PROMPT",
    "SEARCH_VIEWPOINT_PROMPT",
    "Detection",
    "DetectionResponse",
    "OnDemandDetector",
    "YoloeClient",
    "GemmaClient",
    "OnDemandCaptioner",
    "SearchViewpointCaption",
    "parse_ambient_caption",
    "parse_search_caption",
    "CaptionWorker",
    "Controller",
    "ControllerCtx",
    "FindLabelConfig",
    "FindLabelController",
    "FindObservation",
    "FindResult",
    "GotoConfig",
    "GotoController",
    "MoveConfig",
    "MoveController",
    "PrimitiveResult",
    "SearchObservation",
    "SearchResult",
    "SearchSectorConfig",
    "SearchSectorController",
    "Viewpoint",
    "plan_search_tour",
    "resolve_goto_target",
]
