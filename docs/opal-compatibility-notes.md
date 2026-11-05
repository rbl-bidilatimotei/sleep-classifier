# OPAL and Codex compatibility investigation

Inspected on 2026-11-05. No live agent call or model training ran during this investigation.

## Failure evidence

The first live pilot used OPAL 0.3.10 and the app-bundled Codex CLI 0.166.0-alpha.6.
The provider rollout contains a response_item with type additional_tools and role developer.
It declares these tool surfaces:

| Namespace | Declared tools |
| --- | --- |
| functions | exec, wait, request_user_input, request_user_input_async |
| collaboration | followup_task, interrupt_agent, list_agents, send_message, spawn_agent, wait_agent |

These are tool definitions, not proof that those tools were called.
They are also not empty metadata.
The rollout's world_state.top_level_tools records the same namespaces.
The provider session originator is opal_sol and its recorded source is vscode.
This is consistent with a CLI/provider tool-envelope change, but does not establish its cause.
It does not establish malicious injection.

OPAL disables multi-agent and code-mode feature flags, but explicitly enables code_mode_host for code-mode-only models.
The alpha provider still emitted the additional tool definitions.
OPAL's context auditor accepts only known message, reasoning, call, and output records.
It rejects additional_tools with Unexpected provider non-message item: additional_tools.
The saved decision response states that the edits were not applied.
The events end on program version zero; the pilot continued with the original Adam updates.

Do not bypass this audit by ignoring all additional_tools records.
The payload includes collaboration and user-input capabilities outside the intended optimizer tool surface.
A safe schema change would need explicit validation and upstream agreement about the allowed capabilities.
No audit guard or installed OPAL file was changed.

## Upstream status

The latest PyPI opal-optimizer release was still 0.3.10 when checked.
The current upstream _codex_client.py still rejects unknown provider response-item types.
No newer verified upstream release fixes this failure.
The upstream guide describes Codex CLI 0.160.1 in its experimental-backend notes.
That reference motivates a compatibility pin; it does not prove this exact pilot works with that version.

Sources:

- [Current OPAL context audit](https://github.com/xingjian-bai/opal/blob/main/opal/backends/_codex_client.py)
- [Experimental Codex backend guide](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#experimental-codex-backend)
- [OPAL package metadata](https://pypi.org/pypi/opal-optimizer/json)

## Prepared stable CLI alternative

The official @openai/codex 0.160.1 package is pinned by tracked npm manifests:

```text
tools/opal-codex/node_modules/.bin/codex
```

Initial installation checks used only --version and app-server --help.
The project readiness check also verified the existing isolated login without inference.
The executable reports codex-cli 0.160.1 and supports the app-server command.
No login, inference, API request, or credential copy was performed.
The existing isolated OPAL home remains unchanged.
The bundled alpha executable and desktop app remain unchanged.

Reproduce the installation from the repository root:

```bash
npm ci --prefix tools/opal-codex --no-audit --no-fund
```

The tracked tools/opal-codex/package-lock.json records official registry URLs and integrity hashes.
The installed node_modules directory is ignored.
The @openai/codex archive integrity is:

```text
sha512-f1yrJhwgimKQI1kYQlxdPJcFwkNZxZrbz7Hf89EAcnLqkQ7TiESzr2FgSZn13Ga5RuHjlVUfurzwq10wk9zw2g==
```

Run commands from the repository root so the configured relative executable path resolves correctly.
Prefer an explicit configured binary so a later desktop update cannot silently change this experiment's CLI.
uv continues to manage Python dependencies; npm supplies this external executable only.

## Validation boundary

The stable CLI installation and executable check are complete.
Live response compatibility remains unverified until an authorized run produces an accepted decision.
An offline replay of the saved failed rollout still rejects the unknown tool envelope.
A synthetic regression test also verifies that this guard fails closed.
A successful experiment must distinguish requested, failed, kept, applied, and reverted decisions.
It must not report an unchanged fallback run as evidence that reactive optimization has no effect.
