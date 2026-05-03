"""Append-only perception-memory table for the autonomy harness.

One row per multimodal caption: ``(t, spot_id, sector, room_name, objects,
scene_description)``. Rows are written by background ``CaptionWorker``
threads (one per Spot) and consumed by the agent loop / HUD on the main
thread; ``MemoryTable`` is the synchronisation point and the on-disk
JSONL writer.

This is the agent's perception log. The coverage map answers "where
have I been"; this answers "what did I see, when, in which sector".
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class MemoryRow:
    """One captioner observation, sector-stamped + pose-stamped.

    ``t_sim`` is the in-sim wall time at the moment the frame was
    grabbed (so rows replay against the coverage map correctly even if
    the caption call took longer than the sim tick). ``t_wall`` is the
    real-world unix time the row was written, useful for debugging
    pipeline latency. ``sector`` is the chess-style coarse-grid label
    of the Spot's body XZ at frame-grab time (e.g. ``"C2"``); ``None``
    if the spot was outside the navmesh AABB.

    ``pose_x``, ``pose_z`` are the body XZ in world metres at frame-
    grab time. ``pose_yaw_rad`` is the body yaw in radians (same
    convention as the rest of the codebase: yaw=0 means body forward
    is along world +X, ``forward(yaw) = (cos yaw, -sin yaw)``). Pose
    is captured alongside the sector so the agent can recreate the
    exact viewpoint a caption came from -- richer than sector alone,
    which is only 5 m precise.
    """

    t_sim: float
    t_wall: float
    spot_id: int
    sector: Optional[str]
    pose_x: float
    pose_z: float
    pose_yaw_rad: float
    room_name: str
    objects: list[str]
    scene_description: str
    raw_response: Optional[str] = field(default=None, repr=False)

    def to_jsonl(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)


class MemoryTable:
    """Thread-safe append-only table of ``MemoryRow``s with optional
    JSONL persistence.

    Writers (``CaptionWorker`` threads) call :meth:`append`. Readers
    (main loop, HUD, future agent loop) call :meth:`latest_for_spot`,
    :meth:`latest_for_sector`, or :meth:`recent`. Iteration is
    snapshot-safe -- it copies the underlying list once under the lock.

    If ``jsonl_path`` is given, every appended row is also flushed to
    disk as a single line so a crash never loses observations. The file
    is opened in append mode so multiple sessions can share a single
    log.
    """

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._rows: list[MemoryRow] = []
        self._lock = threading.Lock()
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_fh = self._jsonl_path.open("a", encoding="utf-8")
        else:
            self._jsonl_fh = None

    def close(self) -> None:
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.flush()
                self._jsonl_fh.close()
            finally:
                self._jsonl_fh = None

    def append(self, row: MemoryRow) -> None:
        line = row.to_jsonl()
        with self._lock:
            self._rows.append(row)
            if self._jsonl_fh is not None:
                self._jsonl_fh.write(line + "\n")
                self._jsonl_fh.flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def snapshot(self) -> list[MemoryRow]:
        """Return a copy of all rows ordered by insertion (≈ time)."""
        with self._lock:
            return list(self._rows)

    def latest_for_spot(self, spot_id: int) -> Optional[MemoryRow]:
        with self._lock:
            for row in reversed(self._rows):
                if row.spot_id == spot_id:
                    return row
        return None

    def latest_for_sector(self, sector: str) -> Optional[MemoryRow]:
        with self._lock:
            for row in reversed(self._rows):
                if row.sector == sector:
                    return row
        return None

    def recent(self, n: int = 10) -> list[MemoryRow]:
        with self._lock:
            return list(self._rows[-n:])

    def filter_by_spot(self, spot_id: int) -> Iterable[MemoryRow]:
        for row in self.snapshot():
            if row.spot_id == spot_id:
                yield row


def default_jsonl_path(run_dir: Optional[Path] = None) -> Path:
    """Return ``<run_dir>/memory_<unix_ts>.jsonl`` (default
    ``outputs/memory_<unix_ts>.jsonl``)."""
    base = Path(run_dir) if run_dir else Path("outputs")
    ts = int(time.time())
    return base / f"memory_{ts}.jsonl"
