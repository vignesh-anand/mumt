"""HTTP client for the Jetson YOLO26 server.

Mirrors the captioner pair in :mod:`mumt_sim.agent.perception`:

- :class:`YoloeClient` -- synchronous wrapper around
  ``POST /detect/open`` on the FastAPI server documented in the project
  README. Open-vocabulary, uses YOLOE-26-L. Steady-state ~90 ms per
  call as long as the class list does not change between calls.
- :class:`OnDemandDetector` -- small ThreadPoolExecutor wrapping a
  :class:`YoloeClient` so the teleop main loop can fire detections
  without blocking the tick. Returns
  :class:`concurrent.futures.Future` whose result is a
  :class:`DetectionResponse`.

Conventions:

- ``rgb`` arrays from habitat-sim are RGB; ``rgb_is_bgr=False`` is the
  default. The JPEG encoder swaps channels for OpenCV.
- ``bbox`` is ``[x1, y1, x2, y2]`` in pixel coordinates of the original
  uploaded image, matching the server's response.

Important performance note from the server docs: rotating the
``classes`` list between calls drops the server to <1 FPS because
YOLOE has to re-run its text encoder. Keep the same classes for the
duration of a single ``find()`` primitive (we always pass the target
label string, never a rotating list).
"""
from __future__ import annotations

import concurrent.futures as _cf
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np


_DEFAULT_BASE_URL = "http://192.168.50.1:8000"


def _default_base_url() -> str:
    """Honour ``MUMT_YOLOE_URL`` so the server can be moved without
    touching code."""
    return os.environ.get("MUMT_YOLOE_URL", _DEFAULT_BASE_URL)


@dataclass
class Detection:
    """One bounding box from a detector response."""

    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in image px
    class_id: int = -1

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


@dataclass
class DetectionResponse:
    """Parsed JSON body from ``POST /detect`` or ``POST /detect/open``.

    ``image_size_wh`` is ``(width, height)`` of the image as the server
    decoded it, NOT necessarily the model's inference resolution; the
    bboxes in ``detections`` are scaled back to this frame.
    """

    mode: str  # "closed" | "open"
    inference_ms: float
    total_ms: float
    image_size_wh: tuple[int, int]
    detections: list[Detection] = field(default_factory=list)

    def best_for_label(self, label: str) -> Optional[Detection]:
        """Highest-confidence detection whose label exactly equals
        ``label`` (case-insensitive). ``None`` if not present."""
        lab = label.strip().lower()
        best: Optional[Detection] = None
        for det in self.detections:
            if det.label.strip().lower() != lab:
                continue
            if best is None or det.confidence > best.confidence:
                best = det
        return best

    @classmethod
    def from_dict(cls, body: dict) -> "DetectionResponse":
        size = body.get("image_size", [0, 0])
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            size = (0, 0)
        dets = []
        for d in body.get("detections", []):
            try:
                xyxy = tuple(float(v) for v in d["xyxy"])
            except (KeyError, TypeError, ValueError):
                continue
            if len(xyxy) != 4:
                continue
            dets.append(
                Detection(
                    label=str(d.get("class_name", "")).strip(),
                    confidence=float(d.get("conf", 0.0)),
                    xyxy=xyxy,  # type: ignore[arg-type]
                    class_id=int(d.get("class_id", -1)),
                )
            )
        return cls(
            mode=str(body.get("mode", "")),
            inference_ms=float(body.get("inference_ms", 0.0)),
            total_ms=float(body.get("total_ms", 0.0)),
            image_size_wh=(int(size[0]), int(size[1])),
            detections=dets,
        )


def _encode_jpeg(rgb: np.ndarray, rgb_is_bgr: bool, quality: int = 80) -> bytes:
    """Encode ``rgb`` to JPEG bytes via cv2 (already a hard dep)."""
    import cv2

    if not rgb_is_bgr:
        rgb = rgb[:, :, ::-1]  # RGB -> BGR for cv2
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def _normalise_classes(classes: Sequence[str] | str) -> list[str]:
    """Accept ``["human"]``, ``("human","backpack")``, or a single
    comma-separated string. Strips whitespace, drops empties."""
    if isinstance(classes, str):
        items: Iterable[str] = classes.split(",")
    else:
        items = classes
    out: list[str] = []
    for it in items:
        s = str(it).strip()
        if s:
            out.append(s)
    if not out:
        raise ValueError("classes must contain at least one non-empty label")
    return out


class YoloeClient:
    """Synchronous client for the Jetson YOLO26 server's
    ``POST /detect/open`` (open-vocabulary YOLOE-26).

    One client per process is fine; it holds a ``requests.Session`` so
    the TCP connection to the Jetson is reused across calls. Pass
    ``timeout_s`` generously enough to absorb a first call's text-
    encoder cost (~1.5 s) the very first time a new class list is
    seen; subsequent calls with the same list are <100 ms.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: float = 5.0,
        jpeg_quality: int = 80,
        default_conf: float = 0.25,
        default_iou: float = 0.7,
        default_imgsz: Optional[int] = None,
    ) -> None:
        # requests is a runtime dep; lazy import keeps this module
        # importable in environments without it (e.g. unit tests on a
        # fresh wheelhouse).
        import requests  # noqa: WPS433

        self._requests = requests
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.jpeg_quality = int(jpeg_quality)
        self.default_conf = float(default_conf)
        self.default_iou = float(default_iou)
        self.default_imgsz = int(default_imgsz) if default_imgsz else None
        self._session = requests.Session()

    def detect_open(
        self,
        rgb: np.ndarray,
        classes: Sequence[str] | str,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        rgb_is_bgr: bool = False,
    ) -> DetectionResponse:
        """Block-call ``/detect/open``. Raises on transport / HTTP
        errors so the caller's ``Future`` carries the exception."""
        cls_list = _normalise_classes(classes)
        jpeg = _encode_jpeg(rgb, rgb_is_bgr=rgb_is_bgr, quality=self.jpeg_quality)
        files = {"image": ("frame.jpg", jpeg, "image/jpeg")}
        # Server accepts repeated form fields with the same name.
        data: list[tuple[str, str]] = [("classes", c) for c in cls_list]
        data.append(("conf", str(self.default_conf if conf is None else float(conf))))
        data.append(("iou", str(self.default_iou if iou is None else float(iou))))
        eff_imgsz = imgsz if imgsz is not None else self.default_imgsz
        if eff_imgsz:
            data.append(("imgsz", str(int(eff_imgsz))))

        url = f"{self.base_url}/detect/open"
        resp = self._session.post(
            url, files=files, data=data, timeout=self.timeout_s
        )
        resp.raise_for_status()
        return DetectionResponse.from_dict(resp.json())

    def health(self) -> dict:
        """``GET /health`` -- useful for startup diagnostics."""
        resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass


class OnDemandDetector:
    """Small thread-pool wrapper around a :class:`YoloeClient`. Lets a
    controller fire ``detect_open`` from the teleop main loop without
    blocking the tick: ``submit(rgb, classes)`` returns a
    :class:`concurrent.futures.Future`.

    Default ``max_workers=4`` keeps two parallel ``find`` controllers
    fed without overwhelming the Jetson (which serialises per model
    anyway and tops out around ~10 FPS on YOLOE).
    """

    def __init__(
        self,
        client: YoloeClient,
        max_workers: int = 4,
        name: str = "OnDemandDetector",
    ) -> None:
        self.client = client
        self._executor = _cf.ThreadPoolExecutor(
            max_workers=int(max_workers), thread_name_prefix=name
        )

    def submit(
        self,
        rgb: np.ndarray,
        classes: Sequence[str] | str,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        rgb_is_bgr: bool = False,
    ) -> "_cf.Future[DetectionResponse]":
        """Queue a detection. ``rgb`` must outlive the call: the
        teleop loop already produces fresh ``obs[...]`` arrays per
        tick so passing a reference is safe (Python ref-keeps the
        buffer alive until the worker is done)."""

        def _work() -> DetectionResponse:
            return self.client.detect_open(
                rgb, classes,
                conf=conf, iou=iou, imgsz=imgsz,
                rgb_is_bgr=rgb_is_bgr,
            )

        return self._executor.submit(_work)

    def stop(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
        self.client.close()
