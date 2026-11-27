# Test the local HTTP service

The default service intentionally starts without a model bundle.
A successful Uvicorn startup means the HTTP process is running. It does not mean the model is ready.
`bundle_dir: null` produces readiness 503 and `{"error":"not_ready"}` for event requests.
That is the expected default behavior. Do not change the readiness guard to make the request succeed.

Use curl directly. uv manages the Python server environment; it is not required for an HTTP client.
The `%` printed after a curl response can be the zsh no-final-newline marker. It is not part of the JSON.
The commands below use `-i` to show status headers and `-w '\n'` to add a final newline.

## 1. Install the service dependencies

Run these commands from the project checkout:

```bash
cd /Users/timotei.bidila/Dev/Various/sleep-classifier
uv sync --locked --inexact --group service --extra cpu
```

`--extra cpu` selects CPU dependencies on Linux. On macOS, it retains the installed platform's PyTorch build.
`--inexact` keeps existing notebook tools while installing the selected dependencies.
`--no-sync` in later commands uses that prepared environment; it does not install missing dependencies.

## 2. Start the default service in terminal A

```bash
unset SLEEP_SERVICE_CONFIG
uv run --locked --no-sync python -m uvicorn \
  sleep_classifier.http_service:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 16
```

Expect a startup warning that no evaluation bundle is configured.
Keep this process running. Use terminal B for the following requests.

## 3. Check health before submitting events

```bash
curl -i -sS -w '\n' http://127.0.0.1:8000/healthz
curl -i -sS -w '\n' http://127.0.0.1:8000/readyz
```

Expected results:

| Route | HTTP status | Body |
| --- | --- | --- |
| `/healthz` | 200 | `{"status":"alive"}` |
| `/readyz` | 503 | `{"ready":false,"reason":"no_bundle"}` |

The startup log and readiness response explain why event submission is disabled.
A configured bundle that fails loading reports `bundle_unavailable` instead of `no_bundle`.
Check the bundle's exact runtime revision, file integrity, path, and validated device in that case.
Restart after changing the configuration; the service loads the bundle at startup.

## 4. Confirm the expected no-bundle rejection

```bash
curl -i -sS -w '\n' http://127.0.0.1:8000/v1/events \
  -H 'Content-Type: application/json' -H 'X-Event-Sequence: 0' \
  -d '{"schema_version":"1.0.0","type":"start","session_id":"demo","participant_id":"demo","session_start_seconds":0}'
```

Expect HTTP 503 and `{"error":"not_ready"}`. This confirms the readiness guard works.
It does not indicate a curl error or malformed start event.
Open `http://127.0.0.1:8000/docs` to inspect the health and session route groups.

## 5. Test predictions without datasets, training, or GPU use

Stop terminal A with Ctrl+C when finished.
Run the synthetic HTTP prediction test:

```bash
uv run --locked --no-sync python -m pytest -q \
  tests/test_http_service.py::test_http_synthetic_session_matches_direct_runner
```

This test uses random-weight fixtures and a test-only CPU resolver.
It submits all 76 synthetic example records through FastAPI.
Two predictions, one coverage skip, and the lifecycle records must match the direct runner exactly.
It does not load the saved research model or weaken production device checks.

Run the full focused CI selection when needed:

```bash
uv run --locked --no-sync python -m pytest -q \
  tests/test_bundle_loader.py tests/test_bundle_runtime.py \
  tests/test_session_protocol.py tests/test_raw_session_runner.py \
  tests/test_raw_session_cli.py tests/test_http_service.py \
  tests/test_service_sessions.py tests/test_ci_contract.py
uv run --locked --no-sync ruff check .
```

These checks exercise the CPU-only engineering scope. They do not train or assess sleep-stage accuracy.

## 6. Optional: serve the fixed research model on the Mac

This procedure uses MPS inference and existing development artifacts. It performs no training and reads no final-test observations.
Skip this section when keeping the no-GPU constraint. Linux CPU serving still needs a separately validated bundle profile.
The default service and synthetic tests above remain usable without a real model.

### Export a fresh bundle at the current committed revision

Stop the server. Update to the intended PR commit before exporting.
Keep runtime source files committed and clean. Check the revision:

```bash
git rev-parse HEAD
git status --short
uv run --locked --no-sync python -m sleep_classifier.bundle_export \
  --config configs/service-bundle.yaml --export
```

The exporter uses the existing frozen seed-42 fold-01 checkpoint and its fitted transforms.
It verifies the saved 4,587 development predictions before publication. It does not fit a model.
The prepared destination is `artifacts/evaluation-bundles/motion-seed42-fold01-http-v1`.
Neither this destination nor its verification record is overwritten.
If they already exist at another runtime revision, use a new bundle identifier and matching service config.
Preserve old bundles. Never edit their manifests to bypass runtime compatibility.

### Start with the explicit model configuration

In terminal A:

```bash
SLEEP_SERVICE_CONFIG="$PWD/configs/service-model.yaml" \
uv run --locked --no-sync python -m uvicorn \
  sleep_classifier.http_service:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 16
```

In terminal B:

```bash
curl -i -sS -w '\n' http://127.0.0.1:8000/readyz
```

Proceed only after HTTP 200 and `{"ready":true,"reason":null}`.
Reusing an older notebook-47 bundle with new source code does not satisfy the exact-runtime requirement.
A squash merge also changes the revision. Export a new identifier or use the old bundle's matching runtime wheel.

### Replay the synthetic session over HTTP

Run from the checkout in terminal B:

```bash
uv run --locked --no-sync python - <<'PY'
import json
import urllib.request
from pathlib import Path

base = "http://127.0.0.1:8000"
with urllib.request.urlopen(base + "/readyz", timeout=10) as response:
    assert json.load(response)["ready"]

token = None
records = []
for sequence, line in enumerate(Path("examples/raw-session/session.jsonl").read_text().splitlines()):
    headers = {"Content-Type": "application/json", "X-Event-Sequence": str(sequence)}
    if token is not None:
        headers["X-Session-Token"] = token
    request = urllib.request.Request(base + "/v1/events", data=line.encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    token = result["session_token"]
    assert result["next_sequence"] == sequence + 1
    records.extend(result["records"])

assert records[-1]["type"] == "session_finished"
assert records[-1]["predicted_epochs"] == 2
assert records[-1]["skipped_epochs"] == 1
print(json.dumps(records, indent=2))
PY
```

This client uses uv for its Python environment only. curl could send the same requests.
Do not run the manual start request before this replay; it would occupy the single active session.
Finish releases the session. Cancellation requires the current owner token and expected next sequence.
Do not retry an event automatically after losing a successful response; this version has no response replay cache.
Synthetic class predictions have no accuracy interpretation because the example contains no stage targets.

## 7. Optional: check the Docker wrapper

Follow the build commands in the [container guide](local-service-and-ci.md#build-the-container).
The default container has no bundle. Expect the same health 200 and readiness 503 as section 3.
The MPS evaluation bundle cannot become a Linux CPU bundle through a mount or configuration change.
This test checks packaging and HTTP availability only.
