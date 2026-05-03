"""Per-Spot ReAct agent loop driven by Gemini native function-calling.

Architecture in one paragraph
-----------------------------
The :class:`AgentLoop` is a per-Spot worker thread that wakes up on
events from a per-Spot :class:`EventBus` (user typed a message, a tool
emitted progress, a tool finished, a tool failed, etc.), formats those
events plus a fresh ``<state>`` snapshot into a single user-side message
to a :class:`google.genai.chats.Chat`, parses the model's response into
``<thinking>`` / ``<speak>`` text + exactly one native function call
(``wait()`` is the explicit no-op), surfaces speak text to a callback
(terminal in the chat harness) and
hands action calls to the main-thread :class:`ToolDispatcher`. The
dispatcher just owns a per-Spot pending slot; the main loop polls it
each tick to decide whether to install a new controller in the existing
``primitive_controllers[spot_id]`` slot, abort the running one, or
fulfil a stop request. Tool callbacks (progress + result) push events
back onto the bus so the agent stays in the loop. The whole thing is
single-Spot-aware via ``spot_id`` but the dispatcher is keyed by spot
id so the same module drives the future two-spot wiring.

Output contract (parsed from the model's text + function calls)
---------------------------------------------------------------
::

    <thinking>
    short reasoning trace
    </thinking>
    <speak>
    optional message to the human
    </speak>
    [function_call: one of goto / move / search / find / recall / stop / wait / done]

Tools available to the model (see :data:`TOOL_DECLS`):

- ``goto(target)`` -- drive to a sector label (e.g. ``"C2"``) or a
  ``"x,z"`` setpoint string.
- ``move(forward_m, lateral_m, dyaw_deg)`` -- body-frame nudge.
- ``search(sector)`` -- run :class:`SearchSectorController` on the
  named sector. Streams ``ToolProgress`` per visited viewpoint.
- ``find(label, sector)`` -- run :class:`FindLabelController` for a
  YOLOE label inside the named sector. Streams ``ToolProgress`` on
  each detection tick.
- ``recall(question)`` -- LLM query over THIS spot's perception log.
- ``stop()`` -- abort whatever primitive is currently running for me.
- ``wait()`` -- explicit no-op; only valid when a primitive is
  running and the agent has no new instruction this turn.
- ``done(answer)`` -- terminal: agent goes idle after writing a final
  message; chat history is preserved so a follow-up :class:`UserMessage`
  resumes context.

Threading invariants
--------------------
- Only the **main thread** touches ``habitat_sim`` / ``SpotTeleop`` /
  ``ControllerCtx``. Agent threads never instantiate, start, or step
  controllers themselves; they hand a constructed ``Controller``
  instance to the dispatcher and the main loop installs + ticks it.
- The dispatcher is the only piece read by both. Its mutators
  (``submit``, ``request_stop``, ``report_done``) are guarded by a
  single ``threading.Lock``.
- :class:`EventBus` is a bounded ``queue.Queue`` -- safe for cross-
  thread put/get on its own.
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .coverage import CoverageMap
from .detection import OnDemandDetector
from .perception import OnDemandCaptioner
from .recall import OnDemandRecaller
from .tools import (
    Controller,
    FindLabelController,
    FindResult,
    GotoController,
    MoveController,
    PrimitiveResult,
    RecallController,
    RecallResult,
    SearchResult,
    SearchSectorController,
    resolve_goto_target,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_AGENT_MODEL = "gemini-3.1-flash-lite-preview"


def _import_genai():
    """Lazy import so the module is unit-testable without google-genai
    installed. Same pattern as :mod:`mumt_sim.agent.recall`."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    return genai, types


def _api_key_from_env() -> Optional[str]:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    return None


# ---------------------------------------------------------------------------
# AgentEvent hierarchy
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """Base for everything that wakes up the agent. Carries a wall
    timestamp so the formatter can render relative ages if needed.
    Subclasses override :meth:`render` to produce the per-event
    ``<event ...>...</event>`` block sent to the LLM."""

    t_wall: float = field(default_factory=time.time)

    def render(self) -> str:  # pragma: no cover - subclasses override
        return f"<event type=\"{type(self).__name__}\"></event>"


@dataclass
class UserMessage(AgentEvent):
    """User typed a line into the chat. Delivered to the bus
    regardless of whether a tool is running -- the agent decides."""

    text: str = ""

    def render(self) -> str:
        return f"<event type=\"UserMessage\">\n{self.text.strip()}\n</event>"


@dataclass
class ToolStarted(AgentEvent):
    """Main loop just installed and started a controller for us."""

    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        args_str = json.dumps(self.args, separators=(",", ":"))
        return (
            f"<event type=\"ToolStarted\" tool=\"{self.name}\">\n"
            f"args={args_str}\n"
            f"</event>"
        )


@dataclass
class ToolProgress(AgentEvent):
    """Mid-flight progress update from a long-running controller.
    ``payload`` is a small JSON-serialisable dict the controller
    chose to share. We render it inside the ``<event>`` block so the
    model sees the structure verbatim."""

    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        body = json.dumps(self.payload, separators=(",", ": "))
        return (
            f"<event type=\"ToolProgress\" tool=\"{self.name}\">\n"
            f"{body}\n"
            f"</event>"
        )


@dataclass
class ToolResult(AgentEvent):
    """A controller finished cleanly. ``summary`` is the human/LLM-
    facing one-liner; controllers may also stash a structured payload
    in ``details`` for richer recall."""

    name: str = ""
    status: str = "success"
    summary: str = ""
    details: Optional[Dict[str, Any]] = None

    def render(self) -> str:
        det = ""
        if self.details:
            det = "\n" + json.dumps(self.details, separators=(",", ": "))
        return (
            f"<event type=\"ToolResult\" tool=\"{self.name}\" "
            f"status=\"{self.status}\">\n"
            f"{self.summary}{det}\n"
            f"</event>"
        )


@dataclass
class ToolFailed(AgentEvent):
    """A controller errored, timed out, or returned a non-success
    status. Rendered with the same shape as ToolResult but distinct
    type so the model can branch on it."""

    name: str = ""
    status: str = "blocked"
    reason: str = ""

    def render(self) -> str:
        return (
            f"<event type=\"ToolFailed\" tool=\"{self.name}\" "
            f"status=\"{self.status}\">\n"
            f"{self.reason}\n"
            f"</event>"
        )


@dataclass
class ToolStopped(AgentEvent):
    """Controller was aborted -- either by an explicit ``stop()``
    call from the agent, or by the user (e.g. ``X`` hotkey in mixed
    chat/teleop scenarios). Note: submitting a new action while one
    is running does NOT auto-stop -- it is rejected with a
    :class:`ToolFailed` ``status="busy"`` instead."""

    name: str = ""
    reason: str = ""

    def render(self) -> str:
        return (
            f"<event type=\"ToolStopped\" tool=\"{self.name}\">\n"
            f"{self.reason}\n"
            f"</event>"
        )


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Bounded queue of :class:`AgentEvent`s. Drop-oldest on overflow
    so a hung agent thread can never blow the heap. ``drain(timeout)``
    blocks up to ``timeout`` seconds for the first event then drains
    anything else available immediately."""

    def __init__(self, maxsize: int = 64) -> None:
        self.maxsize = int(maxsize)
        self._q: "queue.Queue[AgentEvent]" = queue.Queue(maxsize=self.maxsize)
        self._lock = threading.Lock()

    def put(self, event: AgentEvent) -> None:
        with self._lock:
            if self._q.full():
                # Drop one oldest to make room.
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(event)
            except queue.Full:  # pragma: no cover - guarded above
                pass

    def drain(self, timeout: float = 0.5) -> List[AgentEvent]:
        events: List[AgentEvent] = []
        try:
            first = self._q.get(timeout=timeout)
            events.append(first)
        except queue.Empty:
            return events
        while True:
            try:
                events.append(self._q.get_nowait())
            except queue.Empty:
                break
        return events

    def clear(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return


# ---------------------------------------------------------------------------
# State + parser helpers
# ---------------------------------------------------------------------------


def format_state_block(
    spot_id: int,
    pose_xz: Tuple[float, float],
    yaw_rad: float,
    sector: Optional[str],
    sim_t: float,
    coverage_summary: Optional[str] = None,
    running_tool: Optional[str] = None,
    user_pose_xz: Optional[Tuple[float, float]] = None,
    user_sector: Optional[str] = None,
) -> str:
    """One ``<state>`` block. Always the LAST thing in the user-side
    message so the model reads it as the freshest context.

    ``user_pose_xz`` / ``user_sector`` describe the human user's
    location in the scene (the embodied operator). Letting the agent
    see this enables natural-language commands like "come to me",
    "follow me", "what is in my room" to be grounded spatially. ``None``
    means we don't know where the user is (e.g. headless test).
    """
    yaw_deg = int(round(math.degrees(yaw_rad)))
    lines = [
        f"t_sim={sim_t:6.1f}s  spot{spot_id}  "
        f"pose=({pose_xz[0]:+.2f}, {pose_xz[1]:+.2f}, yaw {yaw_deg:+d}deg)  "
        f"sector={sector or 'NA'}",
    ]
    if user_pose_xz is not None:
        ux, uz = user_pose_xz
        usect = user_sector or "NA"
        lines.append(
            f"user: pose=({ux:+.2f}, {uz:+.2f})  sector={usect}"
        )
    if coverage_summary:
        lines.append(coverage_summary)
    if running_tool:
        lines.append(f"running: {running_tool}")
    body = "\n".join(lines)
    return f"<state>\n{body}\n</state>"


_THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_SPEAK_RE = re.compile(r"<speak>(.*?)</speak>", re.DOTALL | re.IGNORECASE)


def parse_thinking_speak(text: str) -> Tuple[str, Optional[str]]:
    """Pull ``<thinking>`` and ``<speak>`` out of the model's free
    text. Tolerant of missing tags: any text that is not inside
    ``<speak>`` is treated as the thinking trace. Returns
    ``(thinking, speak_or_None)``; both are stripped, ``speak`` is
    None if no ``<speak>`` block was found."""
    if not text:
        return "", None
    speak_match = _SPEAK_RE.search(text)
    speak = speak_match.group(1).strip() if speak_match else None
    # Thinking is everything inside <thinking>, OR everything that's
    # left after stripping <speak>.
    think_matches = _THINK_RE.findall(text)
    if think_matches:
        thinking = "\n".join(t.strip() for t in think_matches).strip()
    else:
        thinking = _SPEAK_RE.sub("", text).strip()
    return thinking, speak


# ---------------------------------------------------------------------------
# Tool schemas (Gemini FunctionDeclarations)
# ---------------------------------------------------------------------------


def _build_tool_decls():
    """Construct the list of ``FunctionDeclaration`` objects exposed
    to the model. Pulled into a function so we can call it lazily,
    after the SDK has been imported."""
    _, t = _import_genai()

    def s(type_, description, **extra):
        return t.Schema(type=type_, description=description, **extra)

    decls = [
        t.FunctionDeclaration(
            name="goto",
            description=(
                "Drive to a navmesh point. `target` is either a "
                "chess-style sector label like 'C2' or a 'x,z' string "
                "in world metres (e.g. '1.20,-3.40'). Blocks until the "
                "spot has arrived, the navmesh path is unreachable, or "
                "progress stalls."
            ),
            parameters=t.Schema(
                type=t.Type.OBJECT,
                properties={"target": s(t.Type.STRING, "Sector label or 'x,z' setpoint.")},
                required=["target"],
            ),
        ),
        t.FunctionDeclaration(
            name="move",
            description=(
                "Body-frame relative motion. Rotates by `dyaw_deg`, "
                "then translates by (`forward_m`, `lateral_m`) in the "
                "rotated body frame. Use for small nudges when goto "
                "is overkill."
            ),
            parameters=t.Schema(
                type=t.Type.OBJECT,
                properties={
                    "forward_m": s(t.Type.NUMBER, "Metres forward (positive) or backward (negative)."),
                    "lateral_m": s(t.Type.NUMBER, "Metres right (positive) or left (negative)."),
                    "dyaw_deg": s(t.Type.NUMBER, "Rotation in degrees, positive = CCW / left."),
                },
                required=[],
            ),
        ),
        t.FunctionDeclaration(
            name="search",
            description=(
                "Plan + execute a sector search. Drives to ~4-6 viewpoints "
                "inside `sector` (and its 8 neighbours) chosen by a greedy "
                "set-cover planner, asks the captioner what it sees at each, "
                "and returns a structured summary. Emits a ToolProgress event "
                "per visited viewpoint so you can decide to stop early."
            ),
            parameters=t.Schema(
                type=t.Type.OBJECT,
                properties={"sector": s(t.Type.STRING, "Chess-style sector label, e.g. 'C2'.")},
                required=["sector"],
            ),
        ),
        t.FunctionDeclaration(
            name="find",
            description=(
                "Hunt for `label` inside `sector` using the YOLOE detector. "
                "Tours the same viewpoints `search` would, but fires the "
                "Jetson detector every ~5 ticks instead of captioning. On "
                "the first detection it walks to ~1m from the target and "
                "faces it, then ends with status=success. Emits a "
                "ToolProgress event on every detection tick."
            ),
            parameters=t.Schema(
                type=t.Type.OBJECT,
                properties={
                    "label": s(t.Type.STRING, "Open-vocabulary class string, e.g. 'human', 'red bag'."),
                    "sector": s(t.Type.STRING, "Chess-style sector label to search inside."),
                },
                required=["label", "sector"],
            ),
        ),
        t.FunctionDeclaration(
            name="recall",
            description=(
                "Query MY perception log via a memory-LLM. Pose-stamped, "
                "sector-stamped captions of everything I've seen are "
                "dumped into the prompt; the LLM answers in prose. Use "
                "for 'have I seen X', 'where did I last see Y', 'what's "
                "in this room' style questions. Other spots' memory is "
                "NOT visible to me."
            ),
            parameters=t.Schema(
                type=t.Type.OBJECT,
                properties={"question": s(t.Type.STRING, "Free-form natural-language question.")},
                required=["question"],
            ),
        ),
        t.FunctionDeclaration(
            name="stop",
            description=(
                "Abort whatever primitive is currently running. No-op "
                "if nothing is running. The main loop will end the "
                "controller this tick and you'll receive a ToolStopped "
                "event before your next turn."
            ),
            parameters=t.Schema(type=t.Type.OBJECT, properties={}, required=[]),
        ),
        t.FunctionDeclaration(
            name="wait",
            description=(
                "Explicit no-op for this turn. Use ONLY when a primitive "
                "is currently running, you have no new instruction to "
                "issue, and you simply want to let it continue (e.g. "
                "after a ToolProgress event that does not require any "
                "change in plan). Never use `wait` after a UserMessage "
                "or after a ToolResult/Failed/Stopped event -- those "
                "always require a real action or `done`."
            ),
            parameters=t.Schema(type=t.Type.OBJECT, properties={}, required=[]),
        ),
        t.FunctionDeclaration(
            name="done",
            description=(
                "Mark the current task as finished. `answer` is your "
                "final report to the user. After this call the agent "
                "goes idle; the chat history is preserved, so the "
                "user's next message will resume with full context."
            ),
            parameters=t.Schema(
                type=t.Type.OBJECT,
                properties={"answer": s(t.Type.STRING, "Final reply to the user.")},
                required=["answer"],
            ),
        ),
    ]
    return decls


TOOL_NAMES = ("goto", "move", "search", "find", "recall", "stop", "wait", "done")


# ---------------------------------------------------------------------------
# Result -> LLM summary
# ---------------------------------------------------------------------------


def format_result_for_llm(
    name: str, result: PrimitiveResult,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Render a finished controller's result as ``(summary, details)``
    for the next ``<event type="ToolResult">``. ``summary`` is a
    one-liner the model reads first; ``details`` is an optional
    structured payload for grounding richer reasoning."""
    fx, fz, fyaw = result.final_pose
    base = (
        f"{result.status} in {result.t_elapsed_s:.1f}s; "
        f"final pose ({fx:+.2f},{fz:+.2f}, yaw {math.degrees(fyaw):+.0f}deg). "
        f"reason: {result.reason}"
    )
    details: Optional[Dict[str, Any]] = None

    if isinstance(result, SearchResult):
        n_obs = len(result.observations)
        n_ok = sum(1 for o in result.observations if o.caption is not None)
        rooms: List[str] = []
        objs: List[str] = []
        people = 0
        for o in result.observations:
            if o.caption is None:
                continue
            people += int(o.caption.people_visible or 0)
            for x in (o.caption.objects_of_interest or [])[:3]:
                if x and x not in objs:
                    objs.append(x)
        # Each search caption already carries a per-vp summary; pull
        # the most informative ones for the agent.
        per_vp = []
        for k, o in enumerate(result.observations):
            if o.caption is None:
                per_vp.append({"vp": k + 1, "error": o.error})
                continue
            per_vp.append({
                "vp": k + 1,
                "summary": (o.caption.summary or "")[:160],
                "objects": (o.caption.objects_of_interest or [])[:5],
                "people": int(o.caption.people_visible or 0),
            })
        summary = (
            f"search {result.sector}: {n_ok}/{n_obs} viewpoints captioned, "
            f"objects seen include {', '.join(objs[:6]) or '-'}, "
            f"people_visible={people}. {base}"
        )
        details = {
            "sector": result.sector,
            "n_planned": result.n_viewpoints_planned,
            "viewpoints": per_vp,
        }
        return summary, details

    if isinstance(result, FindResult):
        if result.found and result.target_world_xz is not None:
            tx, tz = result.target_world_xz
            rng = float(result.target_range_m or 0.0)
            summary = (
                f"find '{result.target_label}' in {result.sector}: FOUND at "
                f"({tx:+.2f},{tz:+.2f}) range~{rng:.2f}m. "
                f"approached={result.approached} centered={result.centered}. "
                f"{base}"
            )
        else:
            summary = (
                f"find '{result.target_label}' in {result.sector}: "
                f"NOT FOUND after {result.n_detections_run} detection ticks "
                f"across {result.n_viewpoints_planned} viewpoints. {base}"
            )
        details = {
            "label": result.target_label,
            "sector": result.sector,
            "found": bool(result.found),
            "n_detections_run": int(result.n_detections_run),
            "n_detections_failed": int(result.n_detections_failed),
        }
        return summary, details

    if isinstance(result, RecallResult):
        ans = (result.answer or "").strip() or "(empty answer)"
        summary = (
            f"recall over {result.n_rows_in_context} rows "
            f"({result.t_call_s:.1f}s): {ans}"
        )
        return summary, {"question": result.question, "answer": ans}

    return f"{name}: {base}", None


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------


@dataclass
class _PendingRequest:
    """Internal: one queued tool request from an agent thread."""
    spot_id: int
    name: str
    args: Dict[str, Any]
    controller: Controller


class ToolDispatcher:
    """Main-thread-side coordinator between agents and the per-spot
    primitive slots.

    Per spot id we keep at most one pending request. The agent is
    required to keep its own primitive slot exclusive: if a primitive
    is already installed (or another request is already queued), a new
    :meth:`submit` is REJECTED and a :class:`ToolFailed` with
    ``status="busy"`` is pushed onto the agent's bus. The agent must
    explicitly call ``stop()`` and wait for the resulting
    :class:`ToolStopped` event before submitting the next action.

    All public methods are thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Dict[int, _PendingRequest] = {}
        self._stop_requested: Dict[int, str] = {}  # spot_id -> reason
        # Once a controller is installed by the main loop we remember
        # which agent owns it so we know which bus to push results to.
        self._installed_name: Dict[int, str] = {}
        self._buses: Dict[int, EventBus] = {}

    # ----- agent registration --------------------------------------------

    def register_bus(self, spot_id: int, bus: EventBus) -> None:
        with self._lock:
            self._buses[int(spot_id)] = bus

    # ----- agent-thread API ----------------------------------------------

    def submit(
        self,
        spot_id: int,
        name: str,
        args: Dict[str, Any],
        controller: Controller,
    ) -> bool:
        """Queue a controller for the main loop to install. Returns
        ``True`` if the request was accepted, ``False`` if the slot is
        busy. On rejection a :class:`ToolFailed` with ``status="busy"``
        is pushed onto the agent's bus so the model gets explicit
        feedback to call ``stop()`` first."""
        spot_id = int(spot_id)
        with self._lock:
            running_name = self._installed_name.get(spot_id)
            pending = self._pending.get(spot_id)
            bus = self._buses.get(spot_id)
            if running_name is not None or pending is not None:
                busy_name = running_name or (pending.name if pending else "?")
                if bus is not None:
                    bus.put(ToolFailed(
                        name=name, status="busy",
                        reason=(
                            f"primitive '{busy_name}' is currently "
                            f"running for this spot. Call stop() and "
                            f"wait for ToolStopped before submitting "
                            f"a new action."
                        ),
                    ))
                return False
            self._pending[spot_id] = _PendingRequest(
                spot_id=spot_id, name=name, args=args, controller=controller,
            )
            return True

    def has_pending(self, spot_id: int) -> bool:
        with self._lock:
            return int(spot_id) in self._pending

    def installed_name(self, spot_id: int) -> Optional[str]:
        with self._lock:
            return self._installed_name.get(int(spot_id))

    def request_stop(self, spot_id: int, reason: str = "agent stop()") -> None:
        with self._lock:
            self._stop_requested[int(spot_id)] = reason

    # ----- main-thread API -----------------------------------------------

    def try_start_pending(self, spot_id: int) -> Optional[_PendingRequest]:
        """If there's a queued request for this spot, hand it back to
        the caller (which is responsible for installing + start()-ing
        it). The caller MUST also call :meth:`note_started` once the
        controller is in place. Submission policy guarantees we never
        have a pending request when something is already installed."""
        spot_id = int(spot_id)
        with self._lock:
            req = self._pending.pop(spot_id, None)
            return req

    def note_started(self, spot_id: int, name: str) -> None:
        """Record that a controller is now installed for this spot.
        Pushes a ToolStarted event onto the bus so the agent knows."""
        spot_id = int(spot_id)
        with self._lock:
            self._installed_name[spot_id] = name
            bus = self._buses.get(spot_id)
        if bus is not None:
            # We don't have the args here -- caller posts richer event
            # if needed. For now we just signal the start.
            bus.put(ToolStarted(name=name))

    def consume_stop(self, spot_id: int) -> Optional[str]:
        """Main loop calls this each tick. If a stop has been
        requested, returns the reason and clears the flag."""
        spot_id = int(spot_id)
        with self._lock:
            return self._stop_requested.pop(spot_id, None)

    def report_done(
        self, spot_id: int, name: str, result: PrimitiveResult,
    ) -> None:
        """Main loop calls this when a controller finishes. We
        translate the result into the right event variant and push
        onto the agent's bus."""
        spot_id = int(spot_id)
        with self._lock:
            self._installed_name.pop(spot_id, None)
            bus = self._buses.get(spot_id)
        if bus is None:
            return
        status = result.status
        if status == "aborted":
            bus.put(ToolStopped(name=name, reason=result.reason))
            return
        if status in ("unreachable", "blocked", "timeout"):
            bus.put(ToolFailed(name=name, status=status, reason=result.reason))
            return
        summary, details = format_result_for_llm(name, result)
        bus.put(ToolResult(
            name=name, status=status, summary=summary, details=details,
        ))

    def push_progress(
        self, spot_id: int, name: str, payload: Dict[str, Any],
    ) -> None:
        """Controllers call this through a bound callback set by the
        main loop. Cheap put on the bus."""
        spot_id = int(spot_id)
        with self._lock:
            bus = self._buses.get(spot_id)
        if bus is not None:
            bus.put(ToolProgress(name=name, payload=payload))


# ---------------------------------------------------------------------------
# AgentClient (Gemini chat with native function calling)
# ---------------------------------------------------------------------------


class AgentClient:
    """Thin wrapper around ``google-genai``'s chat API with native
    function calling enabled. One instance is meant to be shared
    across all per-Spot :class:`AgentLoop`s in the process; each loop
    owns its own ``Chat`` session (history is per-spot)."""

    def __init__(
        self,
        model: str = _DEFAULT_AGENT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 1024,
    ) -> None:
        api_key = api_key or _api_key_from_env()
        if not api_key:
            raise RuntimeError(
                "no Gemini API key found; set GEMINI_API_KEY (or "
                "GOOGLE_API_KEY) before launching"
            )
        genai, types_ = _import_genai()
        self._client = genai.Client(api_key=api_key)
        self._types = types_
        self._tool_decls = _build_tool_decls()
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)

    def make_chat(self, system_prompt: str):
        """Return a fresh :class:`google.genai.chats.Chat` configured
        with the agent's system prompt + tool declarations.

        We force ``function_calling_config.mode = "ANY"`` so the model
        is REQUIRED to emit a function call on every turn -- without
        this, Gemini's default ``AUTO`` mode lets it reply with text
        only, which silently breaks our every-turn-needs-a-call
        contract and sends the agent into a feedback spiral when its
        synthetic protocol-error event hits the loop."""
        t = self._types
        cfg = t.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[t.Tool(function_declarations=self._tool_decls)],
            tool_config=t.ToolConfig(
                function_calling_config=t.FunctionCallingConfig(
                    mode=t.FunctionCallingConfigMode.ANY,
                    allowed_function_names=list(TOOL_NAMES),
                ),
            ),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return self._client.chats.create(model=self.model, config=cfg)


# ---------------------------------------------------------------------------
# AGENT_SYSTEM_PROMPT
# ---------------------------------------------------------------------------


AGENT_SYSTEM_PROMPT = """\
You are the autonomy controller for a quadruped robot ("Spot") exploring an
indoor scene. A human user talks to you over a chat interface and may also
appear in the scene; another robot exists in the world but its memory and
sensors are not visible to you. Your job is to interpret the user's natural-
language requests, plan, drive yourself around, look at things, and report
back.

Output format
-------------
Each response has THREE parts: a thinking trace, an optional speak
block, and a function call. Use these tags inside your text part
exactly as shown:

<thinking>
your private reasoning, planning, and uncertainty
</thinking>
<speak>
short message to the user, plain prose, only when there is something to say
</speak>

Then, in the SAME response, emit EXACTLY ONE native function call
to one of the tools below. There is no such thing as a turn without
a function call -- if you have nothing new to do, call `wait()`
explicitly. The text response alone is never an action; only the
function call moves the world.

CRITICAL RULES:

1. EVERY turn must include exactly one function call. No exceptions.
2. After a UserMessage event you MUST call an action tool (goto /
   move / search / find / recall) or `done`. Do NOT call `wait`
   here. Do NOT just narrate intent. If you say "I'll search the
   kitchen", call search() right now -- the act of calling the tool
   IS the action. Saying you will do something without calling the
   tool means nothing happens.
3. After a ToolStarted / ToolProgress event, if the running
   primitive is doing the right thing and you have no new
   instruction, call `wait()` explicitly to mean "let it keep
   going". If you want to do something different, call `stop()`
   first; submitting a new action while one is running is rejected
   with `ToolFailed status=busy` and nothing happens.
4. After a ToolResult / ToolFailed / ToolStopped event you MUST
   either issue the next action or call `done(answer)`. Never call
   `wait` here -- the previous primitive is finished.

Tools you can call:

- goto(target)         drive to a sector or 'x,z' setpoint
- move(forward_m, lateral_m, dyaw_deg)   small body-frame nudge
- search(sector)       captioned tour of a 5m sector
- find(label, sector)  YOLOE-driven hunt for an object
- recall(question)     memory-LLM query over your own past observations
- stop()               abort the primitive currently running for you
- done(answer)         finish this task; preserves chat for the next message

Sectors and the world
---------------------
The world is divided into 5m chess-style sectors labelled by letter+digit
(A1, B2, C3, ...). `goto`, `search`, and `find` accept either a sector
label or an explicit `"x,z"` string in world metres. The user may also
refer to rooms loosely (kitchen, hallway, bedroom); look those up with
`recall` if you are not sure which sector they live in.

Memory
------
A captioner runs in the background ~ every 2 seconds, writing one row
per observation into your perception log: timestamp, sector, body pose,
inferred room name, objects, scene description. `recall(question)`
sends the whole log to a memory-LLM and returns its prose answer. Other
robots have their OWN logs which you cannot read. If you need
information from another robot you must wait until that capability is
exposed -- for now, only your own observations exist.

Events you will receive
-----------------------
Each turn the user-side message contains a list of `<event ...>`
blocks followed by a `<state>` snapshot. Event types:

- UserMessage: the human typed something. Always react to it -- at
  minimum acknowledge with `<speak>`. The first user message after
  done() starts a new task.
- ToolStarted / ToolProgress / ToolResult / ToolFailed / ToolStopped:
  the lifecycle of a primitive you ran. ToolProgress arrives mid-
  flight (e.g. between viewpoints in a search) so you can decide to
  stop() and try something else.

The `<state>` block at the end is a fresh snapshot of where you are
and what you are doing right now. Use it -- it is more current than
any memory row. The `user:` line in `<state>`, if present, is the
human user's body pose in the same world frame as yours. That is
what to plug into `goto` for instructions like "come to me",
"follow me here", or "what's near me" -- pass the user's `x,z` as a
`"x,z"` setpoint string, or use the `user: sector=...` value as the
sector argument.

Behaviour rules
---------------
1. Be terse with the user. One short sentence per `<speak>` is plenty.
2. Plan before acting. Long thinking is fine; long speech is not.
3. Prefer `recall` over `search` when you suspect you have already seen
   what the user is asking about. Only `search` if you have no
   memory or memory says you have not been there.
4. Use `find` when you know what visual object to look for and the
   sector to search in. Stop the find as soon as you have a useful
   detection by responding to the FOUND ToolProgress; do not redundantly
   wait for the controller to finish if you have what you need.
5. Only one primitive can run at a time per spot. If something is
   already running (`<state>` shows `running_tool=...`) and you want
   to do something different, you MUST call `stop()` first, wait for
   the `ToolStopped` event, and then submit the new action on the
   next turn. Submitting a new action while a primitive is running
   is rejected with a `ToolFailed status=busy` event -- nothing
   changes in the world.
6. Always end a task with `done(answer)`. Do not just stop replying.
7. If the user gives an ambiguous sector reference like "B" or "the
   kitchen", pick the best concrete sector based on `<state>` (your
   current sector, the user's sector) or memory; then act on that
   choice. Do NOT ask the user to disambiguate -- announce your
   choice in `<speak>` and proceed.

Examples
--------
Example A -- recall-only.
USER: "have we seen any humans?"
You:
<thinking>recall over my log first; if nothing, plan a search</thinking>
<speak>checking my memory</speak>
recall(question="have I observed any humans, and if so where and when?")

(LATER, after ToolResult comes back saying yes at C2 ~30s ago)
You:
<thinking>memory says yes at C2; respond and finish</thinking>
<speak>yes -- saw a person in C2 about 30 seconds ago</speak>
done(answer="yes, last seen in sector C2 ~30s ago")

Example B -- search then find.
USER: "find a person in the kitchen"
You:
<thinking>recall first to learn which sector the kitchen is</thinking>
<speak>looking up where the kitchen is</speak>
recall(question="which sector is the kitchen in?")

(ToolResult: "kitchen seen in sector C2 around t=42s")
You:
<thinking>kitchen=C2; hunt for a human there</thinking>
<speak>heading to C2 to look for someone</speak>
find(label="human", sector="C2")

(ToolProgress: detection True, range 1.05m)
You:
<thinking>got them; let the find finish so I am positioned, then report.
no new instruction needed -- wait for the approach to complete.</thinking>
<speak>found someone, walking up to them</speak>
wait()

(ToolResult: status=success, FOUND at (1.2,-3.4) range 1.05m, approached=true)
You:
<thinking>done</thinking>
<speak>got them in C2</speak>
done(answer="found a person in sector C2 at (1.20, -3.40)")

Example C -- mid-task user override.
You are in the middle of a search, ToolProgress just said vp 2/4.
USER: "never mind, go check the bathroom instead"
You:
<thinking>user changed task; abort and pivot</thinking>
<speak>switching</speak>
stop()

(ToolStopped: aborted by stop())
You:
<thinking>find out which sector is the bathroom</thinking>
recall(question="which sector contains the bathroom?")
"""


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


@dataclass
class AgentTraceEntry:
    """One LLM step worth keeping for the HUD ring buffer."""
    t_wall: float
    thinking: str
    speak: Optional[str]
    action: Optional[str]


class AgentLoop:
    """Per-Spot event-driven agent. Construct one per Spot; call
    :meth:`start` once at sim setup and :meth:`stop` on shutdown.
    Public mutators (:meth:`post_user_message`) are thread-safe."""

    def __init__(
        self,
        spot_id: int,
        client: AgentClient,
        dispatcher: ToolDispatcher,
        bus: EventBus,
        coverage: CoverageMap,
        get_state: Callable[[], Dict[str, Any]],
        on_demand_captioner: Optional[OnDemandCaptioner],
        on_demand_detector: Optional[OnDemandDetector],
        on_demand_recaller: Optional[OnDemandRecaller],
        on_speak: Optional[Callable[[int, str], None]] = None,
        on_thinking: Optional[Callable[[int, str], None]] = None,
        on_action: Optional[Callable[[int, str], None]] = None,
        max_steps_per_task: int = 30,
        per_call_timeout_s: float = 30.0,
        history_cap_turns: int = 40,
    ) -> None:
        self.spot_id = int(spot_id)
        self.client = client
        self.dispatcher = dispatcher
        self.bus = bus
        self.coverage = coverage
        self.get_state = get_state
        self.on_demand_captioner = on_demand_captioner
        self.on_demand_detector = on_demand_detector
        self.on_demand_recaller = on_demand_recaller
        self.on_speak = on_speak
        self.on_thinking = on_thinking
        self.on_action = on_action
        self.max_steps_per_task = int(max_steps_per_task)
        self.per_call_timeout_s = float(per_call_timeout_s)
        self.history_cap_turns = int(history_cap_turns)

        self._chat = None
        self._chat_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running_task = False
        self._task_steps = 0
        # Trace ring buffer for the HUD; bounded.
        self._trace: List[AgentTraceEntry] = []
        self._trace_lock = threading.Lock()
        # Last action description for the HUD.
        self._last_action: Optional[str] = None
        # Register our bus with the dispatcher.
        self.dispatcher.register_bus(self.spot_id, self.bus)

    # ----- public API ----------------------------------------------------

    def post_user_message(self, text: str) -> None:
        """Push a user message onto the bus regardless of whether a
        tool is running. Always succeeds (drop-oldest if full)."""
        text = (text or "").strip()
        if not text:
            return
        self.bus.put(UserMessage(text=text))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"AgentLoop-{self.spot_id}", daemon=True,
        )
        self._thread.start()

    def stop(self, wait: bool = False, timeout: float = 1.0) -> None:
        self._stop_event.set()
        # Wake the bus drain.
        self.bus.put(UserMessage(text=""))  # synthetic poke; ignored in loop
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)

    def trace_snapshot(self, n: int = 5) -> List[AgentTraceEntry]:
        with self._trace_lock:
            return list(self._trace[-n:])

    @property
    def is_running_task(self) -> bool:
        return self._running_task

    @property
    def last_action(self) -> Optional[str]:
        return self._last_action

    # ----- worker thread -------------------------------------------------

    def _ensure_chat(self):
        with self._chat_lock:
            if self._chat is None:
                self._chat = self.client.make_chat(AGENT_SYSTEM_PROMPT)
            return self._chat

    def _trim_history(self) -> None:
        """Keep the chat history under ``history_cap_turns`` Content
        objects. Drops the OLDEST user/model pair(s) until we are
        under the cap. The system instruction lives in the config,
        not the history, so trimming history doesn't strip identity."""
        with self._chat_lock:
            if self._chat is None:
                return
            hist = self._chat.get_history(curated=False)
            if len(hist) <= self.history_cap_turns:
                return
            drop = len(hist) - self.history_cap_turns
            del hist[:drop]

    def _run(self) -> None:
        while not self._stop_event.is_set():
            events = self.bus.drain(timeout=0.5)
            # Filter out the synthetic empty poke we use to wake on stop.
            events = [
                e for e in events
                if not (isinstance(e, UserMessage) and not e.text)
            ]
            if not events:
                continue
            # If we receive a UserMessage we are entering / continuing
            # a task; reset the per-task step counter on the FIRST
            # user message after going idle.
            for e in events:
                if isinstance(e, UserMessage) and not self._running_task:
                    self._running_task = True
                    self._task_steps = 0
                    break

            try:
                self._handle_events(events)
            except Exception as exc:  # pragma: no cover - belt and braces
                msg = f"agent error: {type(exc).__name__}: {exc}"
                if self.on_speak:
                    try:
                        self.on_speak(self.spot_id, f"[error] {msg}")
                    except Exception:
                        pass
                # Best-effort: surface to terminal so the user knows.
                print(f"[agent spot{self.spot_id}] {msg}", flush=True)

            if self._task_steps >= self.max_steps_per_task and self._running_task:
                self._running_task = False
                if self.on_speak:
                    try:
                        self.on_speak(
                            self.spot_id,
                            f"[step cap reached: {self.max_steps_per_task} steps]",
                        )
                    except Exception:
                        pass

    def _handle_events(self, events: List[AgentEvent]) -> None:
        chat = self._ensure_chat()
        # Format user-side message: events first, then state stamp.
        rendered = "\n".join(e.render() for e in events)
        state = self._render_state()
        message = f"{rendered}\n\n{state}"

        # LLM call (blocking).
        t0 = time.monotonic()
        try:
            response = chat.send_message(message)
        except Exception as exc:
            print(
                f"[agent spot{self.spot_id}] LLM call failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        t_call = time.monotonic() - t0
        self._task_steps += 1

        # Pull text + function calls out of the response.
        text = (response.text or "").strip()
        thinking, speak = parse_thinking_speak(text)
        fcs = list(getattr(response, "function_calls", []) or [])

        action_str: Optional[str] = None
        for fc in fcs:
            fn_name = (fc.name or "").strip()
            try:
                fn_args = dict(fc.args or {})
            except Exception:
                fn_args = {}
            try:
                handled = self._dispatch_function_call(fn_name, fn_args)
            except Exception as exc:
                self.bus.put(ToolFailed(
                    name=fn_name or "?",
                    status="unreachable",
                    reason=f"bad args ({type(exc).__name__}): {exc}",
                ))
                handled = f"{fn_name}(...) -- bad args"
            if handled is not None and action_str is None:
                action_str = handled

        # Hard contract: every turn must include a function call. If
        # the model emitted text-only, push a synthetic ToolFailed so
        # it gets corrective feedback on the next turn. Use a
        # distinct status string so the model never confuses this
        # protocol error with a navigation/tool failure (e.g. a
        # genuine `goto` 'unreachable').
        if not fcs:
            self.bus.put(ToolFailed(
                name="(none_emitted)",
                status="protocol_error",
                reason=(
                    "PROTOCOL ERROR: your previous response contained "
                    "NO function call. This is not a tool failure -- "
                    "no tool ran, no sector is unreachable, nothing in "
                    "the world changed. You broke the output contract. "
                    "Re-emit your previous intent as an actual function "
                    "call this turn (goto/move/search/find/recall), or "
                    "call wait() if a primitive is already running, or "
                    "done(answer) if the task is finished."
                ),
            ))
            # Also expose to the chat trace so the user can see the
            # protocol error inline rather than just "(no action)".
            if action_str is None:
                action_str = "(no function call -- protocol error)"

        # Surface to callbacks + terminal.
        if thinking and self.on_thinking:
            try:
                self.on_thinking(self.spot_id, thinking)
            except Exception:
                pass
        if speak and self.on_speak:
            try:
                self.on_speak(self.spot_id, speak)
            except Exception:
                pass
        if action_str and self.on_action:
            try:
                self.on_action(self.spot_id, action_str)
            except Exception:
                pass
        # Console trace.
        action_disp = action_str or "(no action)"
        print(
            f"[agent spot{self.spot_id}] step {self._task_steps} "
            f"({t_call:.1f}s): {action_disp}",
            flush=True,
        )
        if thinking:
            think_oneline = thinking.replace("\n", " ").strip()
            if len(think_oneline) > 200:
                think_oneline = think_oneline[:197] + "..."
            print(f"  [think] {think_oneline}", flush=True)
        if speak:
            speak_oneline = speak.replace("\n", " ").strip()
            print(f"  [speak] {speak_oneline}", flush=True)

        # Record trace.
        with self._trace_lock:
            self._trace.append(AgentTraceEntry(
                t_wall=time.time(),
                thinking=thinking,
                speak=speak,
                action=action_str,
            ))
            if len(self._trace) > 16:
                del self._trace[:-16]
        if action_str:
            self._last_action = action_str

        self._trim_history()

    def _render_state(self) -> str:
        """Fetch a fresh state snapshot from the main thread via the
        ``get_state`` callback. The callback returns a dict with
        ``pose_xz``, ``yaw_rad``, ``sector``, ``sim_t``, optional
        ``coverage_summary``, optional ``running_tool``, and optional
        ``user_pose_xz`` / ``user_sector`` (the embodied human's pose;
        omit if unknown)."""
        try:
            st = self.get_state() or {}
        except Exception:
            st = {}
        return format_state_block(
            spot_id=self.spot_id,
            pose_xz=st.get("pose_xz", (0.0, 0.0)),
            yaw_rad=float(st.get("yaw_rad", 0.0)),
            sector=st.get("sector"),
            sim_t=float(st.get("sim_t", 0.0)),
            coverage_summary=st.get("coverage_summary"),
            running_tool=st.get("running_tool"),
            user_pose_xz=st.get("user_pose_xz"),
            user_sector=st.get("user_sector"),
        )

    def _dispatch_function_call(
        self, name: str, args: Dict[str, Any],
    ) -> Optional[str]:
        """Translate a single Gemini function call into either a
        controller submission, a stop request, or a task-end. Returns
        a short ``"name(args)"`` string for the trace, or None if the
        call was unrecognised / malformed."""
        name = name.strip()
        if name == "done":
            answer = str(args.get("answer", "")).strip()
            self._running_task = False
            if answer and self.on_speak:
                try:
                    self.on_speak(self.spot_id, answer)
                except Exception:
                    pass
            return f"done(answer={answer!r})"

        if name == "stop":
            self.dispatcher.request_stop(self.spot_id, reason="agent stop()")
            return "stop()"

        if name == "wait":
            return "wait()"

        if name == "goto":
            target = str(args.get("target", "")).strip()
            if not target:
                return None
            try:
                target_xz = self._resolve_target(target)
            except ValueError as exc:
                self.bus.put(ToolFailed(
                    name="goto", status="unreachable", reason=str(exc),
                ))
                return f"goto(target={target!r}) -- bad target"
            ctl = GotoController(target_xz=target_xz)
            ok = self.dispatcher.submit(
                self.spot_id, "goto", {"target": target}, ctl,
            )
            suffix = "" if ok else " -- busy"
            return f"goto(target={target!r}){suffix}"

        if name == "move":
            forward_m = float(args.get("forward_m", 0.0) or 0.0)
            lateral_m = float(args.get("lateral_m", 0.0) or 0.0)
            dyaw_deg = float(args.get("dyaw_deg", 0.0) or 0.0)
            ctl = MoveController(
                forward_m=forward_m, lateral_m=lateral_m,
                dyaw_rad=math.radians(dyaw_deg),
            )
            ok = self.dispatcher.submit(
                self.spot_id, "move",
                {"forward_m": forward_m, "lateral_m": lateral_m, "dyaw_deg": dyaw_deg},
                ctl,
            )
            suffix = "" if ok else " -- busy"
            return (
                f"move(forward_m={forward_m:+.2f}, lateral_m={lateral_m:+.2f}, "
                f"dyaw_deg={dyaw_deg:+.0f}){suffix}"
            )

        if name == "search":
            sector = str(args.get("sector", "")).strip()
            if not sector or self.on_demand_captioner is None:
                self.bus.put(ToolFailed(
                    name="search", status="unreachable",
                    reason="captioner unavailable" if self.on_demand_captioner is None
                    else "missing sector",
                ))
                return f"search(sector={sector!r}) -- skipped"
            ctl = SearchSectorController(sector, self.on_demand_captioner)
            ok = self.dispatcher.submit(
                self.spot_id, "search", {"sector": sector}, ctl,
            )
            suffix = "" if ok else " -- busy"
            return f"search(sector={sector!r}){suffix}"

        if name == "find":
            label = str(args.get("label", "")).strip()
            sector = str(args.get("sector", "")).strip()
            if not label or not sector or self.on_demand_detector is None:
                self.bus.put(ToolFailed(
                    name="find", status="unreachable",
                    reason="detector unavailable" if self.on_demand_detector is None
                    else "missing label/sector",
                ))
                return f"find(label={label!r}, sector={sector!r}) -- skipped"
            ctl = FindLabelController(
                sector=sector, target_label=label,
                on_demand=self.on_demand_detector,
            )
            ok = self.dispatcher.submit(
                self.spot_id, "find",
                {"label": label, "sector": sector}, ctl,
            )
            suffix = "" if ok else " -- busy"
            return f"find(label={label!r}, sector={sector!r}){suffix}"

        if name == "recall":
            question = str(args.get("question", "")).strip()
            if not question or self.on_demand_recaller is None:
                self.bus.put(ToolFailed(
                    name="recall", status="unreachable",
                    reason="recaller unavailable" if self.on_demand_recaller is None
                    else "missing question",
                ))
                return f"recall(question={question!r}) -- skipped"
            ctl = RecallController(question, self.on_demand_recaller)
            ok = self.dispatcher.submit(
                self.spot_id, "recall", {"question": question}, ctl,
            )
            suffix = "" if ok else " -- busy"
            return f"recall(question={question!r}){suffix}"

        # Unknown tool name -- record as a failure so the model gets
        # feedback on the next turn.
        self.bus.put(ToolFailed(
            name=name or "?", status="unreachable",
            reason=f"unknown tool '{name}'",
        ))
        return None

    def _resolve_target(self, target: str) -> Tuple[float, float]:
        """Accept either a sector label or a 'x,z' string; raises
        ValueError on neither shape. Sector lookup goes through the
        coverage map's read-only API which is safe to call off the
        main thread."""
        target = target.strip()
        if "," in target:
            try:
                xs, zs = target.split(",", 1)
                return float(xs.strip()), float(zs.strip())
            except Exception as exc:
                raise ValueError(f"goto target {target!r} not a 'x,z' pair: {exc}")
        return resolve_goto_target(target, self.coverage)


# ---------------------------------------------------------------------------
# Stdin goal reader
# ---------------------------------------------------------------------------


class StdinChatReader:
    """Daemon thread that reads lines off ``sys.stdin`` and routes
    each non-empty line to a callback.

    The chat harness binds the callback to ``agent.post_user_message``.
    Special two-character lines starting with ``:`` are treated as
    commands (``:abort``, ``:quit``); everything else is forwarded as
    a user message. Empty lines are ignored.

    Stops when ``stop()`` is called -- but stdin reads are blocking,
    so the thread will exit only after the next line lands or the
    process dies. We mark the thread daemon so process exit kills it
    cleanly.
    """

    def __init__(
        self,
        on_message: Callable[[str], None],
        on_command: Optional[Callable[[str], None]] = None,
        prompt: str = "you> ",
    ) -> None:
        self.on_message = on_message
        self.on_command = on_command
        self.prompt = prompt
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="StdinChatReader", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # Print the initial prompt; subsequent prompts are reprinted
        # by the caller after each response (caller knows when to do
        # it without racing the agent's terminal output).
        try:
            sys.stdout.write(self.prompt)
            sys.stdout.flush()
        except Exception:
            pass
        for line in sys.stdin:
            if self._stop.is_set():
                break
            text = line.rstrip("\n").strip()
            if not text:
                # Reprint prompt for empty enters.
                try:
                    sys.stdout.write(self.prompt)
                    sys.stdout.flush()
                except Exception:
                    pass
                continue
            if text.startswith(":") and self.on_command is not None:
                try:
                    self.on_command(text[1:].strip())
                except Exception as exc:
                    print(f"[stdin] command failed: {exc}", flush=True)
            else:
                try:
                    self.on_message(text)
                except Exception as exc:
                    print(f"[stdin] message dropped: {exc}", flush=True)
            # Reprint the prompt after each line so the user can
            # type the next thing while the agent is still working.
            try:
                sys.stdout.write(self.prompt)
                sys.stdout.flush()
            except Exception:
                pass


__all__ = [
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


def format_event_for_llm(ev: AgentEvent) -> str:
    """Public helper: render any :class:`AgentEvent` as the same
    ``<event>`` block the agent loop sends to the model. Useful for
    tests + tracing."""
    return ev.render()
