"""Job-level vs worker-level execution errors through the runner loop.

A job that fails for its own reasons (e.g. the local LibreTranslate has no
model loaded for the requested language pair) must be reported FAILED to the
coordinator while the worker keeps running and claims the next job. Only
worker-level faults (GPU context corruption, model server down, plugin
release failures, model load failures from other plugins) still drain.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest
from sparehour import agent as agent_mod
from sparehour.agent import ExecutionError

from sparehour_worker.runner import (
    JOB_LEVEL_EXECUTION_ERRORS,
    IdleGatedRunner,
    classify_execution_error,
)

from test_runtime import FakeClient, _job, _runner, _slot, _wait_for

# -- classification -----------------------------------------------------------


def _plugin_job(handler, *, job_id="job-1", payload=None):
    job = _job(
        executor={
            "type": "plugin",
            "handler": handler,
            "timeout_seconds": 30,
            "environment": {},
        },
        job_id=job_id,
    )
    job["payload"] = payload or {"text": "Hello world.", "source": "en", "target": "fr"}
    return job


_LT = "sparehour_workloads.libretranslate"
_LT_BATCH = "sparehour_workloads.libretranslate:LibreTranslateBatchExecutor"
_SDXL = "sparehour_workloads.sdxl:SDXLExecutor"


def test_libretranslate_model_load_failed_is_reclassified():
    job = _plugin_job(f"{_LT}:LibreTranslateExecutor")
    original = ExecutionError(
        "model_load_failed", "LibreTranslate has no loaded model for 'en' -> 'fr'"
    )
    reported = classify_execution_error(job, original)
    assert reported is not original
    assert reported.code == "unsupported_language_pair"
    assert "no loaded model" in str(reported)
    assert reported.__cause__ is original


def test_libretranslate_batch_model_load_failed_is_reclassified():
    job = _plugin_job(_LT_BATCH, payload={"segments": ["Hello"], "source": "en", "target": "fr"})
    reported = classify_execution_error(
        job, ExecutionError("model_load_failed", "LibreTranslate has no loaded model")
    )
    assert reported.code == "unsupported_language_pair"


def test_sdxl_model_load_failed_is_unchanged():
    job = _plugin_job(_SDXL)
    original = ExecutionError("model_load_failed", "sdxl weights missing")
    assert classify_execution_error(job, original) is original


def test_worker_level_codes_are_unchanged():
    job = _plugin_job(f"{_LT}:LibreTranslateExecutor")
    for code in ("cuda_context_failure", "model_server_unavailable", "plugin_release_failed"):
        original = ExecutionError(code, "boom")
        assert classify_execution_error(job, original) is original


def test_job_level_code_passes_through_unchanged():
    job = _plugin_job(f"{_LT}:LibreTranslateExecutor")
    original = ExecutionError("unsupported_language_pair", "pair not served")
    assert classify_execution_error(job, original) is original


def test_job_level_errors_are_disjoint_from_draining_errors():
    # Upstream drains on DRAINING_EXECUTION_ERRORS; a code in both sets would
    # be ambiguous (report job-level but drain). They must stay disjoint.
    assert not (JOB_LEVEL_EXECUTION_ERRORS & agent_mod.DRAINING_EXECUTION_ERRORS)


# -- end-to-end: real executor, stub service, real slot loop -------------------


class _StubLibreTranslate(http.server.BaseHTTPRequestHandler):
    """Minimal loopback LibreTranslate: /languages, /translate."""

    LANGUAGES = [
        {"code": "en", "name": "English", "targets": ["de"]},
        {"code": "de", "name": "German", "targets": ["en"]},
    ]

    def _respond(self, document):
        body = json.dumps(document).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/languages":
            self._respond(self.LANGUAGES)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/translate":
            self._respond({"translatedText": "Hallo Welt."})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


class DrainingRecorderClient(FakeClient):
    """FakeClient that records draining heartbeats (drain requests)."""

    def __init__(self):
        super().__init__()
        self.draining_heartbeats: list[str] = []

    def heartbeat_agent(self, agent_id, *, draining=None):
        if draining:
            self.draining_heartbeats.append(agent_id)
        return super().heartbeat_agent(agent_id, draining=draining)


def _lt_job(job_id, target):
    return {
        "id": job_id,
        "lease_token": "tok",
        "workload": "translation",
        "route": "translate",
        "payload": {"text": "Hello world.", "source": "en", "target": target},
        "executor": {
            "type": "plugin",
            "handler": "sparehour_workloads.libretranslate:LibreTranslateExecutor",
            "timeout_seconds": 30,
            "environment": {
                "LIBRETRANSLATE_ENDPOINT": "",  # filled with the stub URL
                "LIBRETRANSLATE_FORMAT": "text",
            },
        },
        "output_contract": [{"name": "translation", "max_bytes": 2097152}],
    }


def test_unsupported_pair_fails_job_but_worker_takes_next_job(tmp_path):
    import sparehour_workloads.libretranslate as lt

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubLibreTranslate)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    lt._READY_PAIRS.clear()
    try:
        stub_url = f"http://127.0.0.1:{server.server_port}/translate"
        jobs = [_lt_job("A", "fr"), _lt_job("B", "de")]
        for job in jobs:
            job["executor"]["environment"]["LIBRETRANSLATE_ENDPOINT"] = stub_url

        client = DrainingRecorderClient()
        client.jobs = jobs
        runner = _runner(client, tmp_path)
        slot = runner.config.registration.slots[0]
        loop = threading.Thread(
            target=runner._slot_loop, args=(slot, False), name="slot-test", daemon=True
        )
        loop.start()
        try:
            # Job B (supported pair) completing proves the loop survived
            # job A's unsupported-pair failure and claimed again.
            assert _wait_for(lambda: "B" in client.completed, timeout=30)
            # No drain was requested before the test itself stops the runner.
            assert not runner.stop.is_set()
            assert client.draining_heartbeats == []
            assert client.failed == [("A", "unsupported_language_pair")]
            assert client.uploaded == ["translation"]
            assert runner.status.failed == 1
            assert runner.status.completed == 1
            notes = "\n".join(runner.status.notes)
            assert "unsupported_language_pair (worker keeps running)" in notes
        finally:
            runner.stop.set()
            loop.join(timeout=10)
            assert not loop.is_alive()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=10)
        lt._READY_PAIRS.clear()


# -- worker-level fault still drains -------------------------------------------


def test_cuda_context_failure_still_drains(tmp_path, monkeypatch):
    def broken_execute(self, slot, job, workdir, lease):
        raise ExecutionError("cuda_context_failure", "CUDA context is corrupted")

    monkeypatch.setattr(agent_mod.AgentRunner, "_execute", broken_execute)
    client = DrainingRecorderClient()
    client.jobs = [_job(job_id="job-1")]
    runner = _runner(client, tmp_path)
    slot = runner.config.registration.slots[0]
    loop = threading.Thread(
        target=runner._slot_loop, args=(slot, False), name="slot-test", daemon=True
    )
    loop.start()
    try:
        assert _wait_for(lambda: bool(client.failed), timeout=30)
        # Worker-level fault: the job is failed with the original code and
        # the runner requested a drain (stop set + draining heartbeat).
        assert client.failed == [("job-1", "cuda_context_failure")]
        assert _wait_for(runner.stop.is_set)
        assert client.draining_heartbeats == [runner.config.registration.agent_id]
        assert runner.status.failed == 1
    finally:
        runner.stop.set()
        loop.join(timeout=10)
        assert not loop.is_alive()


def test_classified_execute_passthrough_preserves_object(tmp_path, monkeypatch):
    """_execute re-raises the very same error object when unchanged, so
    upstream drain classification still sees the original code."""
    sentinel = ExecutionError("plugin_release_failed", "release failed")

    def broken_execute(self, slot, job, workdir, lease):
        raise sentinel

    monkeypatch.setattr(agent_mod.AgentRunner, "_execute", broken_execute)
    runner: IdleGatedRunner = _runner(FakeClient(), tmp_path)

    with pytest.raises(ExecutionError) as caught:
        runner._execute(_slot(), _job(), tmp_path / "wd", _FakeLease())
    assert caught.value is sentinel


class _FakeLease:
    """Minimal lease stand-in for direct _execute invocation."""

    job_id = "job-1"

    def __init__(self):
        self._stop = threading.Event()
        self._thread = threading.current_thread()
