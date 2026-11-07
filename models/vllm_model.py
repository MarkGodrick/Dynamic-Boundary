"""vLLM-backed target model with automatic server lifecycle management.

Starts a vLLM server in a separate conda environment via subprocess,
waits for it to be ready, then delegates generate() to APIModel.
Keeps boundary env clean — no vllm import needed.
"""

import os
import signal
import subprocess
import tempfile
import time
import urllib.request
import urllib.error

from logger import get_logger
from .api_model import APIModel
from .response_parser import get_parser

logger = get_logger(__name__)


class VLLMModel:
    """Target model served by a local vLLM server.

    Manages the full lifecycle: start server → generate → shutdown.
    """

    def __init__(
        self,
        model_name: str,
        conda_env: str = "vllm085",
        cuda_devices: str = "0",
        port: int = 8100,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        extra_args: list[str] | None = None,
        server_timeout: int = 600,
    ):
        self.model_name = model_name
        self.conda_env = conda_env
        self.cuda_devices = cuda_devices
        self.port = port
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_args = extra_args or []
        self.server_timeout = server_timeout

        self._proc: subprocess.Popen | None = None
        self._api_model: APIModel | None = None

        self._start_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        """Start vLLM server and wait until healthy."""
        triton_libcuda_path = os.environ.get(
            "TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu",
        )
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "env", f"CUDA_VISIBLE_DEVICES={self.cuda_devices}",
            f"TRITON_LIBCUDA_PATH={triton_libcuda_path}",
            # "http_proxy=", "https_proxy=", "all_proxy=",
            # "HF_HUB_OFFLINE=1",
            "vllm", "serve", self.model_name,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ] + self.extra_args

        logger.info("Starting vLLM server: %s", " ".join(cmd))
        # Redirect stderr to a temp file, NOT a PIPE: vLLM logs every request
        # to stderr, and an unread PIPE fills its ~64KB OS buffer and blocks
        # (freezes) the server mid-inference. A file drains without blocking
        # and still lets _read_stderr() surface startup errors on failure.
        self._stderr_file = tempfile.NamedTemporaryFile(
            prefix=f"vllm_{self.port}_", suffix=".stderr", delete=False,
        )
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
            preexec_fn=os.setsid,
        )

        # Wait for health
        if not self._wait_for_health():
            stderr = self._read_stderr()
            self.shutdown()
            raise RuntimeError(
                f"vLLM server failed to start within {self.server_timeout}s. "
                f"stderr: {stderr[-1000:]}"
            )

        logger.info("vLLM server ready on port %d", self.port)

        # Build internal APIModel
        os.environ.setdefault("VLLM_API_KEY", "dummy")
        parser = get_parser(self.model_name)
        self._api_model = APIModel(
            model=self.model_name,
            api_key_env="VLLM_API_KEY",
            base_url=f"http://localhost:{self.port}/v1",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_parser=parser,
        )

    def _wait_for_health(self) -> bool:
        """Poll /health until ready or timeout. Returns False if server dies."""
        url = f"http://localhost:{self.port}/health"
        start = time.time()
        while time.time() - start < self.server_timeout:
            # Check if process already died
            if self._proc.poll() is not None:
                logger.error("vLLM server process died during startup")
                return False
            try:
                resp = urllib.request.urlopen(url, timeout=3)
                if resp.status == 200:
                    return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(5)
        return False

    def _read_stderr(self) -> str:
        """Read whatever stderr the server process has written to its file."""
        path = getattr(self, "_stderr_file", None)
        if path is not None:
            try:
                with open(path.name, "rb") as f:
                    return f.read().decode(errors="replace")
            except Exception:
                pass
        return ""

    def shutdown(self) -> None:
        """Stop the vLLM server process."""
        if self._proc is None or self._proc.poll() is not None:
            return
        logger.info("Shutting down vLLM server (pid=%d)", self._proc.pid)
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            self._proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                self._proc.wait(timeout=5)
            except Exception:
                pass
        self._proc = None

    def force_kill(self) -> None:
        """Synchronous SIGKILL on the vllm pgroup; returns immediately, no wait.

        Used by hard-exit paths (second Ctrl-C) where we must guarantee no
        orphan vllm process but have no patience for graceful shutdown. The
        kernel reaps the pgroup; init / systemd reaps any zombies. Safe to
        call from a signal handler — no Python locks, no I/O.
        """
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def __del__(self):
        self.shutdown()

    # ------------------------------------------------------------------
    # Server restart
    # ------------------------------------------------------------------

    def _restart_server(self) -> None:
        """Shutdown and restart the vLLM server."""
        logger.warning("Restarting vLLM server...")
        self.shutdown()
        time.sleep(2)
        self._start_server()

    def _is_server_alive(self) -> bool:
        """Quick health check — returns True if /health responds 200."""
        url = f"http://localhost:{self.port}/health"
        try:
            resp = urllib.request.urlopen(url, timeout=3)
            return resp.status == 200
        except (urllib.error.URLError, ConnectionError, OSError):
            return False

    # ------------------------------------------------------------------
    # TargetModel interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, seed: int | None = None) -> str:
        """Generate a response via the vLLM server (single-turn).

        If the server has crashed (api_model raises after retries),
        attempt to restart the server once and retry the request.
        """
        try:
            return self._api_model.generate(prompt, seed=seed)
        except RuntimeError:
            if self._is_server_alive():
                raise
            logger.error("vLLM server appears dead, attempting restart")
            self._restart_server()
            return self._api_model.generate(prompt, seed=seed)

    def generate_conversation(self, messages: list[dict], seed: int | None = None) -> str:
        """Generate a response for a multi-turn conversation via the vLLM server."""
        try:
            return self._api_model.generate_conversation(messages, seed=seed)
        except RuntimeError:
            if self._is_server_alive():
                raise
            logger.error("vLLM server appears dead, attempting restart")
            self._restart_server()
            return self._api_model.generate_conversation(messages, seed=seed)

    def get_last_meta(self) -> dict:
        """Pass through APIModel's thread-local meta so the trial-runner can
        read finish_reason / truncated / token counts for vLLM-served calls.

        Without this pass-through, `_accumulate_meta`'s
        `getattr(model, 'get_last_meta', None)` check returns None and all
        per-turn meta / trial-level `truncated` fields stay None for vLLM
        runs — defeating the Phase 1 truncation-tracking infra.
        """
        if self._api_model is None:
            return {}
        return self._api_model.get_last_meta()
