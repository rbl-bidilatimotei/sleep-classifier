# Local HTTP service, economical CI, and Docker

This milestone adds transport and package checks around the fixed raw-session runner.
It does not change features, weights, class policies, or the reserved final test.
All automated inference tests use random-weight synthetic fixtures on CPU.

## CI scope

One Ubuntu job runs for pull requests and pushes to `main`.
Feature-branch pushes do not create a second run. New commits cancel stale runs for the same PR.
The job has read-only repository permissions and a fifteen-minute timeout.
It caches uv downloads and installs locked dependencies with Python 3.11.
Linux resolves PyTorch from its explicit CPU wheel index, without CUDA packages.
The Mac PyTorch version stays 2.15.0. This dependency selection does not certify a Linux inference bundle.

The automatic job checks:

- Ruff across the repository
- An explicit list of synthetic bundle, protocol, runner, HTTP, session, and package tests
- A wheel built from the exact checkout revision
- Embedded runtime identity, five schemas, package metadata, and every wheel RECORD hash
- Package paths that exclude datasets, checkpoints, artifacts, notebooks, and unexpected resources

The clean checkout must contain neither `data/` nor `artifacts/` before tests.
The workflow never downloads a dataset, calls a training command, or uses a GPU runner.
This is a restricted test selection, not an operating-system sandbox.
Docker build and health checks require the manual `check_docker` workflow input.
The workflow does not publish images or deploy a service.

## Start locally

Install the locked service dependencies:

```bash
uv sync --locked --group service
uv run --locked --no-sync python -m uvicorn \
  sleep_classifier.http_service:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 16
```

The default settings contain no bundle. `/healthz` returns 200, while `/readyz` and event submission return 503.
Open `http://127.0.0.1:8000/docs` for the route list.
The event body uses the existing JSON schema, linked below; the generic Request handler does not generate its fields in OpenAPI.

To select a bundle, copy `configs/service.yaml` to a private local config.
Set its `bundle_dir` relative to that config file.
Set `SLEEP_SERVICE_CONFIG` to the copied config path before launching the server.
The service never accepts a bundle path from an HTTP request.

The loader requires the exact export runtime revision and validated device.
The current evaluation bundle requires MPS. Older exported bundles require their retained matching wheel or a new export identifier.
A failed runtime or device check leaves readiness false. The service does not fall back to CPU.
No bundle is exported or fitted automatically.

## Submit one session

| Route | Purpose |
| --- | --- |
| `GET /healthz` | Confirm process liveness |
| `GET /readyz` | Check the fixed bundle and device readiness |
| `POST /v1/events` | Submit one event and receive its resulting records |
| `DELETE /v1/session` | Cancel the current owner's session without claiming completion |

Send each event as `application/json`, using the existing [event contract](../schemas/raw-session/v1/event.schema.json).
Extra fields, duplicate JSON keys, nonfinite values, compressed bodies, and oversized bodies are rejected.
The default body limit is 4,096 bytes. Streaming bodies obey the same bound without relying on Content-Length.

Send `X-Event-Sequence: 0` with the start event. Omit `X-Session-Token` on that request.
A successful reply returns `session_token`, `next_sequence`, and `records`.
Send the token and the returned next sequence in subsequent request headers.
A raw observation normally returns an empty `records` list. Watermarks and finish can emit epoch responses.
Use the returned sequence after every successful request. Rejected events do not advance it.

```bash
curl -sS http://127.0.0.1:8000/v1/events \
  -H 'Content-Type: application/json' -H 'X-Event-Sequence: 0' \
  -d '{"schema_version":"1.0.0","type":"start","session_id":"demo","participant_id":"demo","session_start_seconds":0}'
```

A configured and compatible bundle is required for this request to succeed.
Keep the returned token private. Do not put tokens or sensor values in access-log URLs.
The ownership token separates session callers; it is not user authentication or a complete public-service security design.
Bind the service to loopback. Public hosting remains outside this version.

The instance owns one active session. Another start returns 409 until finish, cancellation, or idle expiry.
Use exactly one Uvicorn worker and one instance. Multiple workers do not share session state.
Concurrent mutations return 409 instead of queuing inference.
There is no automatic retry cache. A repeated sequence is rejected before state changes.
If a successful response is lost, the server cannot reproduce it through this interface.
Cancel with the owner token and expected next sequence, or let the idle timeout expire before restarting.
A lost start response can require idle expiry because the client did not receive its ownership token.

Idle expiry occurs on the next event, cancellation, or readiness check. No background cleanup timer runs.
Successful events refresh activity after processing. Invalid requests do not extend ownership.
The event budget includes start. After exhaustion, only finish or cancellation can release the session explicitly.
Finish still obeys the runner's per-event epoch-output cap.
A failed model operation releases the session and returns a generic error. Partial progress is not rolled back.
Shutdown waits for the current inference call before releasing state.
Cancellation and expiry do not produce a false `session_finished` integrity claim.

## Error policy

| Status | Meaning |
| --- | --- |
| 400 | Missing or malformed sequence or length header |
| 403 | Incorrect session ownership |
| 409 | Active, missing, expired, busy, or out-of-sequence session |
| 413 | Request body exceeds its byte limit |
| 415 | Unsupported content type or encoding |
| 422 | Invalid event, late observation, or conflicting sensor record |
| 429 | Pending-event or session-event limit |
| 500 | Terminal runner failure; session is released |
| 503 | No compatible, available bundle/device |

Error bodies contain stable codes. They do not echo sensor records or ownership tokens.
The event schemas, watermark rules, coverage profile, and prediction semantics remain unchanged.
See the [raw-session guide](raw-session-inference.md) for input and completion rules.

## Build the container

Build the wheel from a clean committed checkout first:

```bash
uv build --wheel --out-dir dist
uv run --locked --no-sync python scripts/check_wheel.py \
  --wheel dist/sleep_classifier-0.1.0-py3-none-any.whl \
  --expected-revision "$(git rev-parse HEAD)"
docker build --tag sleep-classifier:local .
docker run --rm --read-only --tmpfs /tmp \
  --publish 127.0.0.1:8000:8000 sleep-classifier:local
```

The image installs locked service dependencies and the built wheel. It runs as an unprivileged numeric user.
The Docker context allowlist contains only the Dockerfile, project metadata, lockfile, and wheels.
It excludes Git history, local environments, sensor data, model artifacts, and credentials.
Model bundles are not baked into the image. Mount any future validated bundle read-only through operator configuration.
Do not publish this evaluation image as a clinical service.

The default container has no bundle. Expect liveness 200 and readiness 503.
The current MPS profile cannot run in a Linux CPU container.
A ready Linux inference service requires a separately validated CPU profile and matching bundle contract.
Changing a manifest device field is not validation. Do not disable the loader guard.
The image's HEALTHCHECK measures liveness, not inference readiness.

## Source documentation

- [uv CPU PyTorch indexes](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/)
- [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
- [FastAPI lifespan tests](https://fastapi.tiangolo.com/advanced/testing-events/)
