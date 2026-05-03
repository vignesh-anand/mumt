"""LLM-backed recall over the perception memory table.

The agent's "what have I seen?" tool. Caller hands in a free-form
question; we dump every row of :class:`mumt_sim.agent.memory.MemoryTable`
into a Gemini 3.1 Flash Lite call and the model answers in prose.

Three pieces:

- :class:`RecallClient` -- text-only ``google-genai`` wrapper, parallel
  shape to :class:`mumt_sim.agent.perception.GeminiClient` but no image
  part. Defaults to ``gemini-3.1-flash-lite-preview``; overridable via
  ``MUMT_RECALL_MODEL``.
- :class:`OnDemandRecaller` -- :class:`concurrent.futures.ThreadPoolExecutor`
  wrapper so the teleop main loop can fire a recall without blocking
  the tick. Same shape as :class:`OnDemandCaptioner` /
  :class:`OnDemandDetector`.
- :func:`format_memory_dump` -- compact one-line-per-row textual
  serialisation of :class:`MemoryRow` snapshots. No truncation,
  Flash Lite has plenty of context.

Performance notes:

- Flash Lite is fast (~2-5 s per call typical) so a single shared
  pool with ``max_workers=2`` is plenty for the 2-spot demo. Recalls
  are also rare events (hotkey / explicit agent tool calls), so
  contention with detection / captioning is irrelevant.
- The whole memory snapshot goes to the model every call by design
  -- this is the "no text search" tool. Revisit only when row count
  exceeds ~10k (current 0.5 Hz x 2-spot capture is ~30 rows / minute).
"""
from __future__ import annotations

import concurrent.futures as _cf
import os
from typing import Iterable, Optional

from .memory import MemoryRow


_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def _import_genai():
    """Lazy import so ``import recall`` works without google-genai
    installed (e.g. in unit tests). Mirrors perception._import_genai."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    return genai, types


def _api_key_from_env() -> Optional[str]:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        val = os.environ.get(name)
        if val:
            return val
    return None


# ---------------------------------------------------------------------------
# Memory dump formatting
# ---------------------------------------------------------------------------


RECALL_SYSTEM_PROMPT = (
    "You are the MEMORY of a multi-robot autonomy stack. Two Spot robots "
    "(spot0, spot1) explore an indoor scene; every ~2 seconds each robot "
    "captions what its head camera sees and writes one row into a shared "
    "perception memory table.\n\n"
    "The user will paste the entire memory table below as plain text, then "
    "ask a question about what has been observed. Each row has the format:\n"
    "  t=<sim_seconds>s spot<id> sect=<sector_or_NA> pose=(x,z,yawDeg) "
    "room=<single_word> objs=[<noun>,<noun>,...] | <one-sentence scene description>\n\n"
    "`sect` is a chess-style coarse-grid label (letter + digit, e.g. C2, B4) "
    "covering a 5 m square of the scene. `NA` means the spot was outside the "
    "navigated area when the row was captured. `pose` is the robot's body "
    "pose at frame-grab time: world XZ in metres and yaw in degrees (yaw=0 "
    "means facing world +X; positive yaw turns left / CCW from above). "
    "`room` is a one-word room label inferred by the captioning model. "
    "Rows are ordered by capture time (oldest first).\n\n"
    "Answer the user's question concisely (1-3 sentences typical), grounded "
    "ONLY in what the rows actually contain. When relevant, cite specific "
    "sectors and rough timestamps (e.g. 'kitchen seen in C2 around t=42s'). "
    "If the answer is not supported by the rows, say so plainly -- do NOT "
    "invent observations. If the table is empty, say the robots have not "
    "captured anything yet."
)


def format_memory_dump(rows: Iterable[MemoryRow]) -> str:
    """Render an iterable of :class:`MemoryRow` as a compact text
    block, one row per line, in capture order.

    Format per row::

        t=  12.3s spot0 sect=C2 pose=(+1.20,-3.40,+45deg) room=kitchen objs=[chair,table,fridge] | a small kitchen with a wooden table

    Empty / missing fields collapse cleanly: empty objects -> ``[]``,
    missing sector -> ``NA``, empty scene description -> ``-``. Yaw
    is rendered in degrees (rounded to int) so the model's reasoning
    about heading is in human-friendly units. The ``raw_response``
    field is intentionally dropped (it duplicates the parsed fields
    and is huge).
    """
    import math

    rows = list(rows)
    if not rows:
        return "(memory table is empty -- no observations have been captured yet)"
    lines: list[str] = []
    for row in rows:
        sect = row.sector if row.sector else "NA"
        objs = ",".join(row.objects) if row.objects else ""
        scene = (row.scene_description or "").strip().replace("\n", " ")
        if not scene:
            scene = "-"
        yaw_deg = int(round(math.degrees(float(row.pose_yaw_rad))))
        pose = (
            f"({float(row.pose_x):+.2f},{float(row.pose_z):+.2f},"
            f"{yaw_deg:+d}deg)"
        )
        lines.append(
            f"t={row.t_sim:6.1f}s spot{int(row.spot_id)} "
            f"sect={sect} pose={pose} room={row.room_name or 'unknown'} "
            f"objs=[{objs}] | {scene}"
        )
    return "\n".join(lines)


def build_recall_user_prompt(question: str, dump: str) -> str:
    """Pair a question with a memory dump into a single user-side
    prompt. Question first so the model knows what to look for as it
    scans the rows; dump second, fenced for clarity."""
    q = (question or "").strip() or "(no question provided)"
    return (
        f"Question: {q}\n\n"
        f"Memory table (oldest -> newest):\n"
        f"-----BEGIN MEMORY-----\n"
        f"{dump}\n"
        f"-----END MEMORY-----"
    )


# ---------------------------------------------------------------------------
# RecallClient
# ---------------------------------------------------------------------------


class RecallClient:
    """Text-only ``google-genai`` wrapper for memory-recall calls.

    Cheap to construct, holds an SDK-internal connection pool, so
    instantiate once per process. Thread-safe enough for concurrent
    ``query`` calls because the SDK manages its own session pool.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_output_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        api_key = api_key or _api_key_from_env()
        if not api_key:
            raise RuntimeError(
                "no Gemini API key found; set GEMINI_API_KEY (or "
                "GOOGLE_API_KEY) before launching"
            )
        genai, _types = _import_genai()
        self._client = genai.Client(api_key=api_key)
        self._types = _types
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)

    def query(self, system_prompt: str, user_prompt: str) -> str:
        """Block-call the recall model with a system + user prompt
        pair. Returns the raw text response (stripped). Raises on
        transport failures so the calling Future carries the exception.

        We pass the system prompt via ``GenerateContentConfig.system_instruction``
        (the canonical Gemini field for it) so it doesn't eat the user
        turn's token budget. Fall back to prepending it to the user
        prompt if the SDK version doesn't expose ``system_instruction``.
        """
        try:
            cfg = self._types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                system_instruction=system_prompt,
            )
            contents = [user_prompt]
        except TypeError:
            # Older SDK without system_instruction kwarg.
            cfg = self._types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
            contents = [system_prompt + "\n\n" + user_prompt]
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=cfg,
        )
        return (resp.text or "").strip()


# ---------------------------------------------------------------------------
# OnDemandRecaller
# ---------------------------------------------------------------------------


class OnDemandRecaller:
    """Tiny thread-pool wrapper around :class:`RecallClient`. Lets a
    controller fire ``query`` from the teleop main loop without
    blocking the tick: ``submit(system, user)`` returns a
    :class:`concurrent.futures.Future` whose result is the raw text
    response.

    Default ``max_workers=2`` -- recall is a rare-but-bursty
    operation, two concurrent calls handles "both spots query at the
    same time" without overcommitting the API.
    """

    def __init__(
        self,
        client: RecallClient,
        max_workers: int = 2,
        name: str = "OnDemandRecaller",
    ) -> None:
        self.client = client
        self._executor = _cf.ThreadPoolExecutor(
            max_workers=int(max_workers), thread_name_prefix=name
        )

    def submit(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> "_cf.Future[str]":
        def _work() -> str:
            return self.client.query(system_prompt, user_prompt)

        return self._executor.submit(_work)

    def stop(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
