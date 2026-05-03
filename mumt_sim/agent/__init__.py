"""Autonomy stack for the mumt project.

Built incrementally over the M-Agent.* milestones. Modules slot in as they
land:

- ``coverage``    -- top-down coverage map + 5 m chess-grid sector
                     vocabulary (M-Agent.2, slices B + C)
- ``memory``      -- thread-safe append-only perception-memory table +
                     JSONL persistence (M-Agent.2, slice E)
- ``perception``  -- per-Spot Gemini Flash Lite caption worker that
                     posts rows into ``memory`` (M-Agent.2, slice E)
- ``detection``   -- HTTP client + thread-pool wrapper for the Jetson
                     YOLOE server, used by the find-label primitive
                     (M-Agent.2, slice G)
- ``recall``      -- LLM-backed recall over ``memory`` rows via
                     Gemini Flash Lite, used by the recall primitive
                     (M-Agent.2, slice H)
- ``tools``       -- atomic action primitives (`goto`, `move`,
                     `search`, `find`, `recall`) + `Controller` base
                     for step-able autonomy (M-Agent.2, slice F-H)
- ``loop``        -- per-Spot event-driven ReAct agent on top of the
                     above tools, using Gemini native function calling
                     (M-Agent.3, slice A)
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
    GeminiClient,
    OnDemandCaptioner,
    SearchViewpointCaption,
    parse_ambient_caption,
    parse_search_caption,
)
from .recall import (
    RECALL_SYSTEM_PROMPT,
    OnDemandRecaller,
    RecallClient,
    build_recall_user_prompt,
    format_memory_dump,
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
    RecallConfig,
    RecallController,
    RecallResult,
    SearchObservation,
    SearchResult,
    SearchSectorConfig,
    SearchSectorController,
    resolve_goto_target,
)
from .loop import (
    AGENT_SYSTEM_PROMPT,
    AgentClient,
    AgentEvent,
    AgentLoop,
    AgentTraceEntry,
    EventBus,
    StdinChatReader,
    TOOL_NAMES,
    ToolDispatcher,
    ToolFailed,
    ToolProgress,
    ToolResult,
    ToolStarted,
    ToolStopped,
    UserMessage,
    format_event_for_llm,
    format_result_for_llm,
    format_state_block,
    parse_thinking_speak,
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
    "GeminiClient",
    "OnDemandCaptioner",
    "SearchViewpointCaption",
    "parse_ambient_caption",
    "parse_search_caption",
    "CaptionWorker",
    "RECALL_SYSTEM_PROMPT",
    "OnDemandRecaller",
    "RecallClient",
    "build_recall_user_prompt",
    "format_memory_dump",
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
    "RecallConfig",
    "RecallController",
    "RecallResult",
    "SearchObservation",
    "SearchResult",
    "SearchSectorConfig",
    "SearchSectorController",
    "Viewpoint",
    "plan_search_tour",
    "resolve_goto_target",
    # loop
    "AGENT_SYSTEM_PROMPT",
    "AgentClient",
    "AgentEvent",
    "AgentLoop",
    "AgentTraceEntry",
    "EventBus",
    "StdinChatReader",
    "TOOL_NAMES",
    "ToolDispatcher",
    "ToolFailed",
    "ToolProgress",
    "ToolResult",
    "ToolStarted",
    "ToolStopped",
    "UserMessage",
    "format_event_for_llm",
    "format_result_for_llm",
    "format_state_block",
    "parse_thinking_speak",
]
