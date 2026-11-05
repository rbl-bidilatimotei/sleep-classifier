import json
from pathlib import Path

import pytest


def test_unknown_provider_tool_envelope_still_fails_closed(tmp_path):
    client_module = pytest.importorskip("opal.backends._codex_client")
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "collaboration",
                            "tools": [
                                {"type": "function", "name": "spawn_agent"},
                            ],
                        },
                    ],
                },
            }
        )
        + "\n"
    )
    client = object.__new__(client_module.Client)
    client.request = {}
    client.rollout = str(rollout)
    with pytest.raises(client_module.IsolationError, match="additional_tools"):
        client.audit(successful=False)


def test_external_cli_dependency_is_pinned_with_registry_integrity():
    root = Path(__file__).resolve().parents[1] / "tools/opal-codex"
    manifest = json.loads((root / "package.json").read_text())
    lock = json.loads((root / "package-lock.json").read_text())
    assert manifest["dependencies"]["@openai/codex"] == "0.160.1"
    package = lock["packages"]["node_modules/@openai/codex"]
    assert package["version"] == "0.160.1"
    assert package["resolved"].startswith("https://registry.npmjs.org/@openai/codex/")
    assert package["integrity"].startswith("sha512-")
