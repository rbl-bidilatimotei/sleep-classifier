from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sleep_classifier.codex_runtime import CodexRuntime


def test_explicit_binary_wins_over_environment(monkeypatch):
    monkeypatch.setenv("CODEX_BINARY", "/env/codex")
    lookup = Mock(return_value="/custom/codex")
    monkeypatch.setattr("sleep_classifier.codex_runtime.shutil.which", lookup)
    assert CodexRuntime.resolve_binary("/custom/codex") == "/custom/codex"
    lookup.assert_called_once_with("/custom/codex")


def test_missing_explicit_binary_does_not_fall_back(monkeypatch):
    monkeypatch.setattr("sleep_classifier.codex_runtime.shutil.which", Mock(return_value=None))
    with pytest.raises(RuntimeError, match="Configured Codex executable"):
        CodexRuntime.resolve_binary("/missing/codex")


def test_bundled_executable_is_found_without_path(monkeypatch):
    monkeypatch.delenv("CODEX_BINARY", raising=False)
    monkeypatch.setattr("sleep_classifier.codex_runtime.shutil.which", Mock(return_value=None))
    monkeypatch.setattr("sleep_classifier.codex_runtime.Path.is_file", Mock(return_value=True))
    monkeypatch.setattr("sleep_classifier.codex_runtime.os.access", Mock(return_value=True))
    assert CodexRuntime.resolve_binary().startswith("/Applications/ChatGPT.app/")


def test_readiness_checks_login_without_inference(tmp_path, monkeypatch):
    backend = Mock(version="codex-cli test", codex_home=tmp_path)
    backend.login_status.return_value = True
    factory = Mock(return_value=backend)
    monkeypatch.setattr(CodexRuntime, "resolve_binary", Mock(return_value="/test/codex"))
    monkeypatch.setattr(
        "sleep_classifier.codex_runtime.import_module",
        Mock(return_value=SimpleNamespace(CodexBackend=factory)),
    )
    result = CodexRuntime.check()
    backend.preflight.assert_called_once_with()
    backend.login_status.assert_called_once_with()
    backend.decide.assert_not_called()
    assert result["logged_in"] is True
    backend.login_status.return_value = False
    with pytest.raises(RuntimeError, match="login --device-auth") as failure:
        CodexRuntime.check()
    assert f"mkdir -p {tmp_path}" in str(failure.value)
    assert str(failure.value).index("mkdir -p") < str(failure.value).index("login --device-auth")
