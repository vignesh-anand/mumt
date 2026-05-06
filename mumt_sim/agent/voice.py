"""Push-to-talk audio capture + ElevenLabs Scribe speech-to-text.

Two pieces, intentionally independent so callers can swap either side:

- :class:`PushToTalkRecorder` -- subprocess-backed mono PCM recorder.
  Spawns ``arecord`` (or ``ffmpeg`` as a fallback) writing a WAV file
  to a tempfile, terminates it on ``stop()``, and returns the captured
  bytes. Running audio I/O in a separate process keeps PortAudio /
  PulseAudio segfaults from killing the HITL sim.

- :class:`ElevenLabsSTT` -- thin wrapper over the ``elevenlabs`` SDK's
  ``speech_to_text.convert`` endpoint. Submits audio off-thread so the
  HITL render loop never blocks on a network round-trip; consumers poll
  the returned ``concurrent.futures.Future``.

Both modules import their heavy deps lazily so importing this file is
free even when STT is disabled.
"""
from __future__ import annotations

import io
import os
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

# Default to 16 kHz mono -- ElevenLabs Scribe handles 16-48 kHz; 16 kHz
# keeps WAV bytes small (~32 kB/s) which is plenty for short voice
# commands and minimises round-trip size.
_DEFAULT_SAMPLE_RATE: int = 16000
_DEFAULT_CHANNELS: int = 1


# ----------------------------------------------------------------------
# Audio capture
# ----------------------------------------------------------------------

class PushToTalkRecorder:
    """Edge-triggered mono PCM recorder backed by an ``arecord`` subprocess.

    Lifecycle::

        rec = PushToTalkRecorder()
        rec.start()         # PTT held
        ...                  # main loop continues; samples accumulate
        wav_bytes = rec.stop()  # PTT released; returns 16-bit PCM WAV

    ``stop()`` returns ``b""`` if no samples were captured (e.g. user
    released within one frame). Calling ``start()`` again resets the
    buffer and re-opens the input stream.

    Subprocess-based on purpose: previous in-process ``sounddevice``
    backends segfaulted on the first ``InputStream`` open under certain
    PulseAudio + habitat-sim combinations, killing the entire HITL sim.
    Driving ``arecord`` (or ``ffmpeg`` as a fallback) in a child process
    isolates audio I/O so a backend crash exits with a non-zero status
    rather than taking down the parent.
    """

    def __init__(
        self,
        *,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        channels: int = _DEFAULT_CHANNELS,
        blocksize: int = 0,  # noqa: ARG002 - kept for backwards compat
        device: Optional[Union[int, str]] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        # ALSA/PulseAudio device hint:
        #   * None  -> arecord's default device ("default")
        #   * str   -> passed as ``-D <device>`` to arecord (e.g.
        #     "plughw:1,0", "pulse", "default")
        #   * int   -> resolved to "plughw:<idx>,0"
        self.device = device
        self._proc: Optional[subprocess.Popen] = None
        self._tempfile: Optional[str] = None
        self._start_t: float = 0.0

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_arecord_device(device) -> Optional[str]:
        if device is None:
            return None
        if isinstance(device, int):
            return f"plughw:{device},0"
        return str(device)

    def _build_command(self, out_wav: str) -> list[str]:
        """Pick ``arecord`` if present, else ``ffmpeg``. Both produce a
        16-bit PCM mono WAV at ``self.sample_rate`` to ``out_wav``."""
        if shutil.which("arecord"):
            cmd = [
                "arecord",
                "-q",
                "-f", "S16_LE",
                "-r", str(self.sample_rate),
                "-c", str(self.channels),
                "-t", "wav",
            ]
            dev = self._resolve_arecord_device(self.device)
            if dev is not None:
                cmd.extend(["-D", dev])
            cmd.append(out_wav)
            return cmd
        if shutil.which("ffmpeg"):
            # PulseAudio capture via ffmpeg as a fallback for systems
            # without ALSA tools installed.
            cmd = [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "pulse",
                "-i", str(self.device) if self.device else "default",
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-acodec", "pcm_s16le",
                out_wav,
            ]
            return cmd
        raise RuntimeError(
            "No audio recorder found. Install 'arecord' (apt install "
            "alsa-utils) or 'ffmpeg' for push-to-talk capture."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Spawn arecord/ffmpeg writing to a fresh tempfile."""
        if self.is_active():
            return
        # Ensure any lingering process from a previous PTT is gone.
        self._kill_proc()
        fd, path = tempfile.mkstemp(prefix="mumt_ptt_", suffix=".wav")
        os.close(fd)
        self._tempfile = path
        cmd = self._build_command(path)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._start_t = time.monotonic()

    def stop(self) -> bytes:
        """Stop the subprocess and return the captured WAV bytes.

        Returns ``b""`` when nothing was captured."""
        if self._proc is None:
            return b""
        try:
            # arecord finalises the WAV header on SIGINT (ffmpeg too).
            # Falling back to SIGTERM if the recorder ignores SIGINT.
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
        finally:
            self._proc = None

        path = self._tempfile
        self._tempfile = None
        if not path or not os.path.isfile(path):
            return b""
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            data = b""
        try:
            os.remove(path)
        except OSError:
            pass
        # Guard against degenerate captures (PTT released sub-frame).
        if len(data) < 64:
            return b""
        return data

    def close(self) -> None:
        self._kill_proc()
        path = self._tempfile
        self._tempfile = None
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=0.5)
        except Exception:  # noqa: BLE001
            pass
        self._proc = None


# ----------------------------------------------------------------------
# Speech-to-text (ElevenLabs Scribe)
# ----------------------------------------------------------------------

# Scribe's first GA model. ElevenLabs may add new variants; expose the
# model name on the constructor so the YAML can pick one without code
# changes.
_DEFAULT_STT_MODEL: str = "scribe_v1"


class ElevenLabsSTT:
    """Async wrapper around ElevenLabs ``speech_to_text.convert``.

    >>> stt = ElevenLabsSTT()       # reads ELEVENLABS_API_KEY
    >>> fut = stt.transcribe(wav_bytes)
    >>> text = fut.result(timeout=10.0)

    The thread pool is small (default ``max_workers=2``) because PTT
    requests are bursty and the transcribe-on-release pattern won't
    stack many concurrent calls.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_STT_MODEL,
        max_workers: int = 2,
    ) -> None:
        if not api_key:
            api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. Pass api_key= or export "
                "the env var before launching mumt_hitl_app.",
            )
        self.api_key = api_key
        self.model = str(model)
        self._client = None
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="elevenlabs-stt",
        )

    def _get_client(self):
        # Lazy: only import / construct on first use so the module can
        # be imported even when ``elevenlabs`` is not installed.
        if self._client is None:
            try:
                from elevenlabs.client import ElevenLabs
            except Exception as exc:  # pragma: no cover - optional dep
                raise RuntimeError(
                    f"The 'elevenlabs' SDK is required for ElevenLabsSTT "
                    f"but failed to import ({exc}). Install via "
                    f"'pip install elevenlabs'.",
                ) from exc
            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    def transcribe(self, wav_bytes: bytes) -> Future:
        """Submit audio for transcription. Returns a ``Future[str]``.

        ``wav_bytes`` should be a complete RIFF/WAV byte string (the
        output of :meth:`PushToTalkRecorder.stop`). An empty / very
        short clip resolves to an empty string without hitting the API.
        """
        if not wav_bytes:
            f: Future = Future()
            f.set_result("")
            return f
        return self._pool.submit(self._transcribe_blocking, wav_bytes)

    def _transcribe_blocking(self, wav_bytes: bytes) -> str:
        client = self._get_client()
        # SDK accepts a file-like object on .convert. Wrap the bytes so
        # we don't pay for a tempfile round-trip.
        buf = io.BytesIO(wav_bytes)
        # ElevenLabs SDK expects a name attribute on the file-like
        # because internally it sniffs the extension; give it a .wav
        # to match the byte payload.
        buf.name = "ptt.wav"
        try:
            resp = client.speech_to_text.convert(
                file=buf,
                model_id=self.model,
            )
        except Exception as exc:  # noqa: BLE001 - surface to caller
            raise RuntimeError(
                f"ElevenLabs STT call failed: {exc}",
            ) from exc

        # SDK returns either an object with .text or a dict; handle both
        # for forwards-compatibility with newer SDK releases.
        text = getattr(resp, "text", None)
        if text is None and isinstance(resp, dict):
            text = resp.get("text")
        return (text or "").strip()

    def stop(self, wait: bool = False) -> None:
        try:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)
        except TypeError:
            # Python <3.9 fallback (no cancel_futures arg).
            self._pool.shutdown(wait=wait)


# ----------------------------------------------------------------------
# Speech-to-text (Gemini)
# ----------------------------------------------------------------------

# Gemini natively accepts inline audio under generate_content via
# ``Part.from_bytes(data=wav, mime_type="audio/wav")``. We use a short
# transcribe-only prompt so the model returns the spoken text and
# nothing else. Default model matches the rest of the autonomy stack
# (gemini-3.1-flash-lite-preview is fine for short PTT clips, but we
# default to ``flash`` here since transcribe quality matters more
# than latency for the user experience).
_DEFAULT_GEMINI_STT_MODEL: str = "gemini-3.1-flash-lite-preview"

_GEMINI_STT_PROMPT: str = (
    "Transcribe the spoken audio verbatim. Output ONLY the transcript "
    "text, with no preamble, no quotation marks, and no commentary. "
    "If the audio is silent or unintelligible, output an empty string."
)


def _gemini_api_key_from_env() -> Optional[str]:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )


class GeminiSTT:
    """Async Gemini-backed STT with the same surface as ``ElevenLabsSTT``.

    Drop-in replacement: exposes ``transcribe(wav_bytes) -> Future[str]``
    and ``stop(wait=False)``. Reuses the project's existing
    ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` env vars so no extra
    credentials are needed when the rest of the autonomy stack is
    already pointed at Gemini.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_GEMINI_STT_MODEL,
        max_workers: int = 2,
        prompt: str = _GEMINI_STT_PROMPT,
    ) -> None:
        api_key = api_key or _gemini_api_key_from_env()
        if not api_key:
            raise RuntimeError(
                "no Gemini API key found; set GEMINI_API_KEY (or "
                "GOOGLE_API_KEY) before launching mumt_hitl_app."
            )
        self.api_key = api_key
        self.model = str(model)
        self.prompt = str(prompt)
        self._client = None
        self._types = None
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="gemini-stt",
        )

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai  # type: ignore  # noqa: PLC0415
                from google.genai import types  # type: ignore  # noqa: PLC0415
            except Exception as exc:  # pragma: no cover - optional dep
                raise RuntimeError(
                    f"The 'google-genai' SDK is required for GeminiSTT "
                    f"but failed to import ({exc}). Install via "
                    f"'pip install google-genai'.",
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
            self._types = types
        return self._client, self._types

    def transcribe(self, wav_bytes: bytes) -> Future:
        if not wav_bytes:
            f: Future = Future()
            f.set_result("")
            return f
        return self._pool.submit(self._transcribe_blocking, wav_bytes)

    def _transcribe_blocking(self, wav_bytes: bytes) -> str:
        client, types = self._get_client()
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=wav_bytes, mime_type="audio/wav",
                    ),
                    self.prompt,
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=256,
                    temperature=0.0,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - surface to caller
            raise RuntimeError(
                f"Gemini STT call failed: {exc}",
            ) from exc

        text = getattr(resp, "text", None) or ""
        # Strip stray quotes / model-padding that some responses still
        # include despite the prompt asking for raw text.
        text = text.strip().strip('"').strip("'").strip()
        return text

    def stop(self, wait: bool = False) -> None:
        try:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)
        except TypeError:
            self._pool.shutdown(wait=wait)


__all__ = [
    "PushToTalkRecorder",
    "ElevenLabsSTT",
    "GeminiSTT",
]
