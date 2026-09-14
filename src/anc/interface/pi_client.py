"""Pi 5 HTTP client for remote file-based audio processing.

Sends WAV audio to a Raspberry Pi 5 running inference_server.py,
receives enhanced audio + telemetry, and returns a structured result.

Connection Configuration
------------------------
Pi host/port defaults to mDNS hostname ``raspberrypi.local:8090``,
overridable via constructor args, environment variables (``PI5_HOST``,
``PI5_PORT``), or the dashboard UI. No hardcoded IP addresses.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# Named constant — short timeout for direct Ethernet link (fail-fast)
PI5_REQUEST_TIMEOUT_SECONDS = 10


@dataclass
class PiConnectionConfig:
    """Connection parameters for the Pi 5 inference server.

    Defaults to mDNS hostname (available via avahi-daemon on Raspberry Pi OS)
    so the demo works without a static IP. Override via env vars or explicit args.
    """
    host: str = ""
    port: int = 0
    timeout_seconds: int = PI5_REQUEST_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        # Allow env-var overrides
        if not self.host:
            self.host = os.environ.get("PI5_HOST", "raspberrypi.local")
        if not self.port:
            self.port = int(os.environ.get("PI5_PORT", "8090"))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class PiClient:
    """HTTP client for Pi 5 inference server.

    All methods handle connection errors gracefully — they return error dicts
    or None instead of raising exceptions, so the dashboard never crashes
    if the Pi is offline or unreachable.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.config = PiConnectionConfig(
            host=host or "",
            port=port or 0,
            timeout_seconds=timeout_seconds or PI5_REQUEST_TIMEOUT_SECONDS,
        )

    def send_for_processing(self, wav_bytes: bytes) -> dict[str, Any]:
        """Send WAV audio to the Pi for DTLN enhancement.

        Parameters
        ----------
        wav_bytes : bytes
            Raw WAV file bytes (complete file including header).

        Returns
        -------
        dict
            On success: ``{ "success": True, "enhanced_audio_base64": str,
                "sample_rate": int, "processing_time_ms": float,
                "realtime_ratio": float, "telemetry": dict }``
            On failure: ``{ "success": False, "error": str }``
        """
        url = f"{self.config.base_url}/process"
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload = json.dumps({"audio_base64": audio_b64}).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data

        except urllib.error.URLError as e:
            reason = str(e.reason) if hasattr(e, "reason") else str(e)
            return {
                "success": False,
                "error": (
                    f"Pi 5 unreachable at {self.config.base_url}: {reason}. "
                    f"Check that the inference server is running and the Pi is "
                    f"connected via Ethernet."
                ),
            }
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode("utf-8"))
                return body
            except Exception:
                return {
                    "success": False,
                    "error": f"Pi 5 returned HTTP {e.code}: {e.reason}",
                }
        except TimeoutError:
            return {
                "success": False,
                "error": (
                    f"Pi 5 request timed out after {self.config.timeout_seconds}s. "
                    f"The Pi at {self.config.base_url} may be overloaded or "
                    f"unreachable."
                ),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error communicating with Pi 5: {str(e)}",
            }

    def check_health(self) -> dict[str, Any] | None:
        """Check if the Pi 5 inference server is alive and model is loaded.

        Returns
        -------
        dict or None
            Health status dict on success, None if unreachable.
        """
        url = f"{self.config.base_url}/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
