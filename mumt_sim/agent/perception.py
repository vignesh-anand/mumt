"""Background perception worker for the autonomy harness.

One ``CaptionWorker`` runs per Spot. Every ``period`` seconds it grabs
the latest head-cam RGB frame (snapshotted from the main thread) plus
the Spot's body XZ + sim time, sends the frame to Gemma 4 via the
``google-genai`` SDK, asks for a strict JSON ``{room_name, objects,
scene_description}`` response, and appends a :class:`MemoryRow` to the
shared :class:`MemoryTable`.

Threading model:
- One worker thread per Spot. Workers never touch the simulator
  directly; they read snapshots posted by the main loop via
  :meth:`CaptionWorker.post_observation` (cheap, just stores the latest
  frame + metadata under a lock).
- The main loop is non-blocking: it just posts the latest snapshot and
  carries on. The HTTP call to Gemma happens entirely off-thread.
- ``MemoryTable.append`` is the only point of cross-thread mutation,
  and it is itself lock-guarded.

Failure handling:
- Network errors / parse errors are logged once per failure and never
  block the worker; the worker continues with the next tick. Memory
  rows from a failed call are still written but with an empty room /
  objects payload and the raw response (or error) in
  ``raw_response``, so debugging is easy.
"""
from __future__ import annotations

import concurrent.futures as _cf
import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .memory import MemoryRow, MemoryTable


# google-genai is imported lazily so the rest of the module can be
# inspected (and the workers stubbed out) without an API key being set.
def _import_genai():
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    return genai, types


def _api_key_from_env() -> Optional[str]:
    """Look for an API key in the env vars google-genai natively reads.
    Returns the key string or None."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        val = os.environ.get(name)
        if val:
            return val
    return None


_DEFAULT_MODEL = "gemma-4-26b-a4b-it"

# ---------------------------------------------------------------------------
# Prompts + parsers
# ---------------------------------------------------------------------------

AMBIENT_CAPTION_PROMPT = (
    "You are the perception system of an indoor robot. Look at this "
    "image from the robot's head camera and respond with STRICT JSON "
    "(no prose, no markdown fences) in this exact schema:\n"
    "{\n"
    '  "room_name": "<short single-word room label, e.g. kitchen, bedroom, hallway, living_room, bathroom, unknown>",\n'
    '  "objects": ["<noun>", "<noun>", ...],\n'
    '  "scene_description": "<one short sentence describing what is in view>"\n'
    "}\n"
    "Keep `objects` to at most 8 entries, lower-case nouns. If you "
    "are not sure, use \"unknown\" for the room and an empty list for "
    "objects."
)


SEARCH_VIEWPOINT_PROMPT = (
    "You are the perception system of an indoor robot that is actively "
    "SEARCHING this area. You are parked at a deliberately-chosen search "
    "viewpoint. Catalogue what is in view from this angle and respond with "
    "STRICT JSON (no prose, no markdown fences) in this exact schema:\n"
    "{\n"
    '  "summary": "<one sentence describing what is in view at this viewpoint>",\n'
    '  "objects_of_interest": ["<noun>", "<noun>", ...],\n'
    '  "people_visible": <integer count>,\n'
    '  "anomalies": ["<short description>", "<short description>", ...]\n'
    "}\n"
    "objects_of_interest: lower-case nouns, at most 8, focus on items a "
    "search-and-rescue or security operator would care about (bags, tools, "
    "furniture, doorways, anything dropped on the floor, signs of people).\n"
    "people_visible: integer count of humans / human-shaped figures visible. "
    "0 if none.\n"
    "anomalies: open doors, dropped items, spilled liquids, broken glass, "
    "unusual mess, signs of disturbance. Empty list if nothing notable. "
    "Keep each entry short (one phrase)."
)


@dataclass
class _Snapshot:
    rgb: np.ndarray  # (H, W, 3) uint8 BGR or RGB
    rgb_is_bgr: bool
    t_sim: float
    sector: Optional[str]


@dataclass
class SearchViewpointCaption:
    """Parsed Gemma response for a single search-mode viewpoint."""

    summary: str
    objects_of_interest: list[str]
    people_visible: int
    anomalies: list[str]
    raw_response: Optional[str] = field(default=None, repr=False)


def _encode_jpeg(rgb: np.ndarray, rgb_is_bgr: bool, quality: int = 80) -> bytes:
    """Encode ``rgb`` to JPEG bytes. We use cv2.imencode because the
    teleop pipeline is already in BGR uint8; PIL would force a copy +
    colour-channel swap. ``rgb_is_bgr=True`` matches OpenCV's native
    layout, which is what the teleop's head-cam frames are."""
    import cv2  # local import; cv2 is already a hard dep
    if not rgb_is_bgr:
        rgb = rgb[:, :, ::-1]  # RGB -> BGR for cv2
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def _extract_json_object(raw: str) -> Optional[dict]:
    """Pull the first ``{...}`` JSON object out of ``raw``, tolerant of
    markdown fences and leading/trailing prose. Returns ``None`` if no
    object could be parsed."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    lo = text.find("{")
    hi = text.rfind("}")
    if lo == -1 or hi == -1 or hi <= lo:
        return None
    try:
        obj = json.loads(text[lo : hi + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_ambient_caption(raw: str) -> tuple[str, list[str], str]:
    """Pull ``(room_name, objects, scene_description)`` out of a raw
    Gemma response; falls back to ``("unknown", [], <truncated raw>)``
    if JSON parsing fails."""
    obj = _extract_json_object(raw)
    if obj is None:
        return "unknown", [], raw.strip()[:200]
    room = str(obj.get("room_name", "unknown")).strip().lower() or "unknown"
    objs_raw = obj.get("objects", [])
    if isinstance(objs_raw, list):
        objs = [str(o).strip().lower() for o in objs_raw if str(o).strip()]
    else:
        objs = []
    scene = str(obj.get("scene_description", "")).strip()
    return room, objs[:8], scene


def parse_search_caption(raw: str) -> SearchViewpointCaption:
    """Pull a :class:`SearchViewpointCaption` out of a raw Gemma
    response; falls back to a stub with the raw text in ``summary`` if
    JSON parsing fails."""
    obj = _extract_json_object(raw)
    if obj is None:
        return SearchViewpointCaption(
            summary=raw.strip()[:200],
            objects_of_interest=[],
            people_visible=0,
            anomalies=[],
            raw_response=raw,
        )
    summary = str(obj.get("summary", "")).strip()
    objs_raw = obj.get("objects_of_interest", [])
    if isinstance(objs_raw, list):
        objs = [str(o).strip().lower() for o in objs_raw if str(o).strip()][:8]
    else:
        objs = []
    pv_raw = obj.get("people_visible", 0)
    try:
        people = int(pv_raw)
    except Exception:
        people = 0
    anom_raw = obj.get("anomalies", [])
    if isinstance(anom_raw, list):
        anoms = [str(a).strip() for a in anom_raw if str(a).strip()][:5]
    else:
        anoms = []
    return SearchViewpointCaption(
        summary=summary,
        objects_of_interest=objs,
        people_visible=people,
        anomalies=anoms,
        raw_response=raw,
    )


class GemmaClient:
    """Tiny wrapper around the ``google-genai`` SDK. Schema-agnostic:
    callers pass a prompt and get back the raw text response, then run
    their own parser (``parse_ambient_caption``,
    ``parse_search_caption``, etc).

    The client is cheap to construct but does hold a TCP connection
    pool inside the SDK, so reuse one instance across workers.

    ``max_output_tokens`` and ``temperature`` are forwarded to every
    ``generate_content`` call. We default to a small token cap because
    our schemas all want short JSON; this is the single biggest knob
    on per-call latency for a 26B-class model."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_output_tokens: int = 256,
        temperature: float = 0.3,
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

    def caption(
        self,
        rgb: np.ndarray,
        prompt: str,
        rgb_is_bgr: bool = True,
        jpeg_quality: int = 80,
    ) -> str:
        """Block-call Gemma with one image + one text prompt. Returns
        the raw response text (stripped). Raises on transport
        failures."""
        jpeg = _encode_jpeg(rgb, rgb_is_bgr=rgb_is_bgr, quality=jpeg_quality)
        # Image-before-text is the recommended order for Gemma 4.
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                self._types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                prompt,
            ],
            config=self._types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
            ),
        )
        return (resp.text or "").strip()

    def caption_ambient(
        self,
        rgb: np.ndarray,
        rgb_is_bgr: bool = True,
        jpeg_quality: int = 80,
    ) -> tuple[str, list[str], str, str]:
        """Convenience: ambient prompt + parser. Returns
        ``(room_name, objects, scene_description, raw_response)``."""
        raw = self.caption(rgb, AMBIENT_CAPTION_PROMPT, rgb_is_bgr, jpeg_quality)
        room, objs, scene = parse_ambient_caption(raw)
        return room, objs, scene, raw

    def caption_search(
        self,
        rgb: np.ndarray,
        rgb_is_bgr: bool = True,
        jpeg_quality: int = 80,
    ) -> SearchViewpointCaption:
        """Convenience: search-viewpoint prompt + parser."""
        raw = self.caption(rgb, SEARCH_VIEWPOINT_PROMPT, rgb_is_bgr, jpeg_quality)
        return parse_search_caption(raw)


class OnDemandCaptioner:
    """Small thread-pool executor wrapping a :class:`GemmaClient`.
    Lets the main loop fire off a Gemma call from a controller without
    blocking the tick: ``submit(rgb, prompt, parser)`` returns a
    :class:`concurrent.futures.Future` whose result is whatever
    ``parser(raw_text)`` returns.

    ``max_workers`` controls how many Gemma calls can be in-flight
    concurrently. We default to 2 so that if a controller times out a
    call and abandons its future, the underlying HTTP call can keep
    running on its own worker without blocking the next ``submit``
    from the same (or another) controller. Set to 1 to fully
    serialise.

    The TCP pool inside the wrapped ``GemmaClient`` is shared with
    whatever ambient ``CaptionWorker``s use the same client.
    """

    def __init__(
        self,
        client: GemmaClient,
        max_workers: int = 6,
        name: str = "OnDemandCaptioner",
    ) -> None:
        self.client = client
        self._executor = _cf.ThreadPoolExecutor(
            max_workers=int(max_workers), thread_name_prefix=name
        )

    def submit(
        self,
        rgb: np.ndarray,
        prompt: str,
        parser: Callable[[str], Any],
        rgb_is_bgr: bool = True,
        jpeg_quality: int = 80,
    ) -> "_cf.Future[Any]":
        """Queue a Gemma call. The submitted ``rgb`` must outlive the
        call (we copy a reference, not the buffer); the teleop loop
        already produces fresh ``obs[...]`` arrays each tick so that's
        satisfied. ``parser`` is run on the worker thread so the
        Future yields the parsed result directly."""

        def _work() -> Any:
            raw = self.client.caption(
                rgb, prompt, rgb_is_bgr=rgb_is_bgr, jpeg_quality=jpeg_quality
            )
            return parser(raw)

        return self._executor.submit(_work)

    def stop(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


class CaptionWorker:
    """Per-Spot background captioner.

    Construct one per Spot, call :meth:`start` once at sim setup and
    :meth:`stop` at tear-down. The main loop posts the latest head-cam
    snapshot via :meth:`post_observation` every tick (cheap); the
    worker thread wakes every ``period`` seconds, grabs the most
    recent snapshot, calls Gemma, appends a row to ``memory``.

    The worker survives Gemma errors -- on failure it appends a row
    with empty payload and the error message in ``raw_response``, then
    sleeps until the next period and retries. This keeps the rest of
    the pipeline running and the failures auditable.
    """

    def __init__(
        self,
        spot_id: int,
        client: GemmaClient,
        memory: MemoryTable,
        period_s: float = 2.0,
        warmup_s: float = 0.5,
    ) -> None:
        self.spot_id = int(spot_id)
        self.client = client
        self.memory = memory
        self.period_s = float(period_s)
        self.warmup_s = float(warmup_s)
        self._latest: Optional[_Snapshot] = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----- main thread API -------------------------------------------------

    def post_observation(
        self,
        rgb: np.ndarray,
        t_sim: float,
        sector: Optional[str],
        rgb_is_bgr: bool = True,
    ) -> None:
        """Stash the latest head-cam RGB so the worker can read it. The
        frame is NOT copied here -- the caller must hand in a frame
        whose backing storage will not be overwritten before the
        worker grabs it. The teleop main loop already produces a fresh
        ``obs[...]`` ndarray per tick, which satisfies this."""
        snap = _Snapshot(
            rgb=rgb, rgb_is_bgr=rgb_is_bgr, t_sim=float(t_sim), sector=sector
        )
        with self._latest_lock:
            self._latest = snap

    # ----- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"CaptionWorker-{self.spot_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ----- worker thread ---------------------------------------------------

    def _take_latest(self) -> Optional[_Snapshot]:
        with self._latest_lock:
            snap = self._latest
            self._latest = None
            return snap

    def _run(self) -> None:
        # Stagger workers slightly so they don't both call Gemma at the
        # same instant -- offset by ``warmup_s * spot_id`` keeps the
        # API load smooth.
        if self.warmup_s > 0:
            self._stop.wait(self.warmup_s * self.spot_id)
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_t:
                # Wake either when it's time, or when stop is set.
                if self._stop.wait(next_t - now):
                    break
            next_t = max(now, next_t) + self.period_s

            snap = self._take_latest()
            if snap is None:
                continue

            t_wall = time.time()
            try:
                room, objs, scene, raw = self.client.caption_ambient(
                    snap.rgb, rgb_is_bgr=snap.rgb_is_bgr
                )
                row = MemoryRow(
                    t_sim=snap.t_sim,
                    t_wall=t_wall,
                    spot_id=self.spot_id,
                    sector=snap.sector,
                    room_name=room,
                    objects=objs,
                    scene_description=scene,
                    raw_response=raw,
                )
            except Exception as exc:
                row = MemoryRow(
                    t_sim=snap.t_sim,
                    t_wall=t_wall,
                    spot_id=self.spot_id,
                    sector=snap.sector,
                    room_name="error",
                    objects=[],
                    scene_description="",
                    raw_response=f"<gemma error: {type(exc).__name__}: {exc}>",
                )
            self.memory.append(row)
