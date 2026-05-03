"""Tiny LLM-driven orchestrator that fans user messages out to per-Spot agents.

Architecture in one paragraph
-----------------------------
The orchestrator is a single LLM that sits between the user's chat
prompt and N per-Spot :class:`mumt_sim.agent.loop.AgentLoop` instances.
It owns its own :class:`mumt_sim.agent.loop.EventBus`; the chat reader
posts each user line onto that bus. A worker thread drains the bus
and asks Gemini "who is this for?" via two tools:

- ``tell(spot_ids=[...], message="...")``  -- forward ``message`` (verbatim
  or paraphrased) to one or more spots. The orchestrator invokes a
  caller-supplied ``on_route`` callback per ``tell`` call; the chat
  harness binds that to ``spot_buses[id].put(UserMessage(text))``.
- ``ask_user(message="...")`` -- the orchestrator addresses the user
  directly (only when it cannot route). Bound to a terminal printer
  in the harness.

Spot speech (``AgentLoop.on_speak``) is **not** routed through the
orchestrator: it goes straight to the terminal with a ``spotN>``
prefix. That keeps the orchestrator strictly user-to-spot and lets
spot replies appear with zero added latency.

Like :class:`mumt_sim.agent.loop.AgentClient` we force Gemini's
function-calling mode to ``ANY`` so every turn produces at least one
tool call -- text-only replies from the model would silently drop
user instructions.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .loop import (
    EventBus,
    UserMessage,
    _api_key_from_env,
    _import_genai,
)


_DEFAULT_ORCHESTRATOR_MODEL = "gemini-3.1-flash-lite-preview"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


def _build_router_decls(num_spots: int):
    """Build the tool declarations for the orchestrator.

    Two tools only:

    - ``tell(spot_ids: list[int], message: str)`` -- forward a message
      to the listed spots. ``spot_ids`` is constrained to the integers
      ``0 .. num_spots-1``; ``message`` is the natural-language string
      to deliver (can be a paraphrase or the user's text verbatim).
    - ``ask_user(message: str)`` -- speak directly to the user. Use
      this only when routing is impossible (ambiguous addressee that
      cannot be resolved from context).
    """
    _, t = _import_genai()

    def s(type_, desc, **extra):
        return t.Schema(type=type_, description=desc, **extra)

    valid_ids = list(range(num_spots))
    valid_ids_str = ", ".join(str(i) for i in valid_ids)

    decls = [
        t.FunctionDeclaration(
            name="tell",
            description=(
                f"Forward an instruction to one or more spots. "
                f"`spot_ids` is a list of integers chosen from "
                f"[{valid_ids_str}]; pass a single id like [0] to "
                f"address one spot, or multiple ids like [0, 1] when "
                f"the same instruction applies to several. `message` "
                f"is the natural-language instruction the spot will "
                f"receive as if the user typed it directly to that "
                f"spot. Pass the user's message verbatim unless you "
                f"need to rephrase (e.g. when splitting work across "
                f"spots, drop the addressee prefix and rewrite for "
                f"the recipient)."
            ),
            parameters=t.Schema(
                type=t.Type.OBJECT,
                properties={
                    "spot_ids": t.Schema(
                        type=t.Type.ARRAY,
                        description=(
                            f"Recipient spot ids; each must be one of "
                            f"[{valid_ids_str}]."
                        ),
                        items=s(t.Type.INTEGER, "Spot id."),
                    ),
                    "message": s(
                        t.Type.STRING,
                        "Instruction text the spot's agent will read.",
                    ),
                },
                required=["spot_ids", "message"],
            ),
        ),
        t.FunctionDeclaration(
            name="ask_user",
            description=(
                "Speak to the user directly. Use only when you cannot "
                "route the message (truly ambiguous addressee that "
                "context does not resolve). Prefer `tell` whenever "
                "possible -- the user wants results, not chatter."
            ),
            parameters=t.Schema(
                type=t.Type.OBJECT,
                properties={
                    "message": s(t.Type.STRING, "Plain prose to the user."),
                },
                required=["message"],
            ),
        ),
    ]
    return decls


ROUTER_TOOL_NAMES = ("tell", "ask_user")


# ---------------------------------------------------------------------------
# OrchestratorClient
# ---------------------------------------------------------------------------


class OrchestratorClient:
    """Gemini chat wrapper for the orchestrator.

    Mirrors :class:`mumt_sim.agent.loop.AgentClient` but with a much
    smaller tool set and a different system prompt. One instance is
    meant to be shared across the process; the loop owns the chat
    session so history is per-orchestrator (there is only one)."""

    def __init__(
        self,
        num_spots: int = 2,
        model: str = _DEFAULT_ORCHESTRATOR_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
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
        self.num_spots = int(num_spots)
        self._tool_decls = _build_router_decls(self.num_spots)
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)

    def make_chat(self, system_prompt: str):
        """Return a fresh chat with forced ``mode=ANY`` and the
        router tool declarations bound."""
        t = self._types
        cfg = t.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[t.Tool(function_declarations=self._tool_decls)],
            tool_config=t.ToolConfig(
                function_calling_config=t.FunctionCallingConfig(
                    mode=t.FunctionCallingConfigMode.ANY,
                    allowed_function_names=list(ROUTER_TOOL_NAMES),
                ),
            ),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return self._client.chats.create(model=self.model, config=cfg)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _render_system_prompt(num_spots: int) -> str:
    ids = ", ".join(str(i) for i in range(num_spots))
    return ROUTER_SYSTEM_PROMPT_TEMPLATE.format(num_spots=num_spots, ids=ids)


ROUTER_SYSTEM_PROMPT_TEMPLATE = """\
You are the dispatcher for a multi-robot search-and-rescue team. The
team is {num_spots} quadruped Spot robots, ids: {ids}. Each spot is
its own LLM agent that can navigate, search a sector, find labelled
objects with a YOLO detector, and recall its own perception memory.
Spots cannot see each other's memory. A human user gives you natural-
language instructions; you decide which spot(s) should hear each one
and forward the instruction. You are a router, not a planner. Do not
add reasoning of your own about how the spots should accomplish the
task -- that is the spots' job.

Output contract
---------------
EVERY turn must call exactly one or more of the tools below. The
SDK is configured so a text-only reply is impossible; you have no
"stay silent" option. If the user is not asking anything actionable
(small talk, "thanks"), use `ask_user` to acknowledge briefly.

Tools:

- tell(spot_ids=[...], message="..."):
    forward `message` as a fresh user-side instruction to each spot
    in `spot_ids`. Spot ids must come from [{ids}]. The message text
    is what that spot's agent will see, verbatim. Pass the user's
    message through unchanged unless you need to rewrite (e.g. when
    splitting work, drop the addressee prefix and reword for the
    recipient).

- ask_user(message="..."):
    speak to the user directly. Use only when routing is impossible
    (e.g. user references "the other one" with no prior context).

Routing rules
-------------
1. Explicit addressee. If the user says "spot 0", "spot zero",
   "the first one", "robot 1", etc., route to that spot only:
       tell(spot_ids=[0], message="<the rest of the instruction>")

2. "Both" / "everyone" / "all of you" / a generic command with no
   addressee -> route to ALL spots:
       tell(spot_ids=[{ids}], message="<verbatim user message>")

3. Split work. If the user explicitly assigns different sub-tasks
   per spot ("spot 0 take the kitchen, spot 1 take the bedroom"),
   call `tell` ONCE per spot with the rewritten instruction, e.g.:
       tell(spot_ids=[0], message="search the kitchen")
       tell(spot_ids=[1], message="search the bedroom")

4. Implicit split. If the user gives a coverage task like "search
   the whole house" or "find the missing person fast", split the
   work across both spots with paraphrased instructions, e.g.:
       tell(spot_ids=[0], message="search the western half of the house for any signs of a person")
       tell(spot_ids=[1], message="search the eastern half of the house for any signs of a person")
   Use your best judgement; do not stall asking for clarification
   on this kind of generic command.

5. Ambiguity that you cannot resolve -> `ask_user(...)`.

Style
-----
Keep messages short and instruction-shaped. Don't add commentary,
reasoning, or "got it"-style chatter to the spots; you are pretending
to be the user typing to them. The user sees the spots' replies in
the same terminal, prefixed with `spotN>`, so you don't need to
relay anything in the spot->user direction -- that path is fully
transparent.
"""


# ---------------------------------------------------------------------------
# OrchestratorLoop
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorTraceEntry:
    """One orchestrator turn, for HUD / debugging."""
    t_wall: float
    user_text: str
    routed: List[tuple]  # list of (spot_ids, message)
    asked_user: Optional[str] = None


class OrchestratorLoop:
    """Worker thread that drains the orchestrator's bus, asks the
    LLM where each :class:`UserMessage` should go, and fires the
    routing callbacks. Public mutator (:meth:`post_user_message`) is
    thread-safe."""

    def __init__(
        self,
        client: OrchestratorClient,
        bus: EventBus,
        on_route: Callable[[Sequence[int], str], None],
        on_ask_user: Optional[Callable[[str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        per_call_timeout_s: float = 20.0,
        history_cap_turns: int = 30,
    ) -> None:
        self.client = client
        self.bus = bus
        self.on_route = on_route
        self.on_ask_user = on_ask_user
        self.on_thinking = on_thinking
        self.per_call_timeout_s = float(per_call_timeout_s)
        self.history_cap_turns = int(history_cap_turns)

        self._chat = None
        self._chat_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._trace: List[OrchestratorTraceEntry] = []
        self._trace_lock = threading.Lock()

    # ----- public API ----------------------------------------------------

    def post_user_message(self, text: str) -> None:
        """Push a user message onto the bus. Empty lines are ignored."""
        text = (text or "").strip()
        if not text:
            return
        self.bus.put(UserMessage(text=text))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="OrchestratorLoop", daemon=True,
        )
        self._thread.start()

    def stop(self, wait: bool = False, timeout: float = 1.0) -> None:
        self._stop_event.set()
        # Wake any pending bus drain.
        self.bus.put(UserMessage(text=""))
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)

    def trace_snapshot(self, n: int = 5) -> List[OrchestratorTraceEntry]:
        with self._trace_lock:
            return list(self._trace[-n:])

    # ----- worker thread -------------------------------------------------

    def _ensure_chat(self):
        with self._chat_lock:
            if self._chat is None:
                prompt = _render_system_prompt(self.client.num_spots)
                self._chat = self.client.make_chat(prompt)
            return self._chat

    def _trim_history(self) -> None:
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
            # Filter the synthetic empty pokes used to wake on stop.
            user_msgs = [
                e for e in events
                if isinstance(e, UserMessage) and e.text
            ]
            if not user_msgs:
                continue
            # Concatenate batched user messages (rare but possible).
            joined = "\n\n".join(
                f"<event type=\"UserMessage\">\n{e.text}\n</event>"
                for e in user_msgs
            )
            chat = self._ensure_chat()
            try:
                response = chat.send_message(joined)
            except Exception as exc:
                print(
                    f"[orch] LLM call failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            fcs = list(getattr(response, "function_calls", []) or [])
            text = (response.text or "").strip()
            if text and self.on_thinking:
                try:
                    self.on_thinking(text)
                except Exception:
                    pass

            routed: List[tuple] = []
            asked: Optional[str] = None

            if not fcs:
                # SDK should make this impossible with mode=ANY but
                # belt-and-suspenders: if it ever happens, ask the
                # user to repeat (and skip routing this turn).
                msg = (
                    "[orch] no tool call emitted; the dispatcher LLM "
                    "broke contract. Please repeat your instruction."
                )
                print(msg, flush=True)
                if self.on_ask_user:
                    try:
                        self.on_ask_user(msg)
                    except Exception:
                        pass
                continue

            for fc in fcs:
                name = (fc.name or "").strip()
                try:
                    args = dict(fc.args or {})
                except Exception:
                    args = {}
                if name == "tell":
                    spot_ids_raw = args.get("spot_ids") or []
                    try:
                        spot_ids = [int(x) for x in spot_ids_raw]
                    except Exception:
                        spot_ids = []
                    spot_ids = [
                        i for i in spot_ids
                        if 0 <= i < self.client.num_spots
                    ]
                    message = str(args.get("message") or "").strip()
                    if not spot_ids or not message:
                        continue
                    try:
                        self.on_route(spot_ids, message)
                    except Exception as exc:
                        print(
                            f"[orch] on_route raised "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    routed.append((tuple(spot_ids), message))
                elif name == "ask_user":
                    message = str(args.get("message") or "").strip()
                    if not message:
                        continue
                    asked = message
                    if self.on_ask_user:
                        try:
                            self.on_ask_user(message)
                        except Exception as exc:
                            print(
                                f"[orch] on_ask_user raised "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )

            with self._trace_lock:
                self._trace.append(OrchestratorTraceEntry(
                    t_wall=time.time(),
                    user_text="\n".join(e.text for e in user_msgs),
                    routed=routed,
                    asked_user=asked,
                ))
                if len(self._trace) > 16:
                    del self._trace[:-16]

            self._trim_history()


__all__ = [
    "OrchestratorClient",
    "OrchestratorLoop",
    "OrchestratorTraceEntry",
    "ROUTER_SYSTEM_PROMPT_TEMPLATE",
    "ROUTER_TOOL_NAMES",
]
