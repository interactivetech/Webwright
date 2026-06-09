import importlib
from pathlib import Path
import sys
import types


def _load_doctor_module():
    fake_rich_console = types.SimpleNamespace(Console=lambda *args, **kwargs: object())

    class _FakeTable:
        def __init__(self, *args, **kwargs):
            pass

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *args, **kwargs):
            pass

    fake_rich_table = types.SimpleNamespace(Table=_FakeTable)
    sys.modules["rich.console"] = fake_rich_console
    sys.modules["rich.table"] = fake_rich_table
    sys.modules.pop("webwright.run.doctor", None)
    return importlib.import_module("webwright.run.doctor")


def test_check_python():
    doctor = _load_doctor_module()
    ok, message = doctor.check_python()

    assert isinstance(ok, bool)
    assert isinstance(message, str)


def test_check_playwright():
    doctor = _load_doctor_module()
    ok, message = doctor.check_playwright()

    assert isinstance(ok, bool)
    assert isinstance(message, str)


def test_check_chromium():
    doctor = _load_doctor_module()
    ok, message = doctor.check_chromium()

    assert isinstance(ok, bool)
    assert isinstance(message, str)


def test_check_screenshot():
    doctor = _load_doctor_module()
    ok, message = doctor.check_screenshot()

    assert isinstance(ok, bool)
    assert isinstance(message, str)


def test_check_openai_key_exists(monkeypatch):
    doctor = _load_doctor_module()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    ok, message = doctor.check_openai_key()

    assert ok is True
    assert "found" in message


def test_check_openai_key_missing(monkeypatch):
    doctor = _load_doctor_module()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_ENDPOINT", raising=False)

    ok, message = doctor.check_openai_key()

    assert ok is False
    assert "missing" in message


def test_check_openai_key_missing_but_local_endpoint_configured(monkeypatch):
    doctor = _load_doctor_module()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://192.168.1.171:8000/v1")

    ok, message = doctor.check_openai_key()

    assert ok is True
    assert "local OpenAI-compatible endpoint configured" in message


def test_plugin_manifests_exist(tmp_path, monkeypatch):
    doctor = _load_doctor_module()
    monkeypatch.chdir(tmp_path)

    claude_dir = tmp_path / ".claude-plugin"
    codex_dir = tmp_path / ".codex-plugin"

    claude_dir.mkdir()
    codex_dir.mkdir()

    (claude_dir / "plugin.json").write_text("{}")
    (codex_dir / "plugin.json").write_text("{}")

    ok, message = doctor.check_plugin_manifests()

    assert ok is True
    assert "found" in message


def test_plugin_manifests_missing(tmp_path, monkeypatch):
    doctor = _load_doctor_module()
    monkeypatch.chdir(tmp_path)

    ok, message = doctor.check_plugin_manifests()

    assert ok is False
    assert "missing" in message


def test_screenshot_file_cleanup():
    doctor = _load_doctor_module()
    screenshot_path = Path("doctor_test.png")

    if screenshot_path.exists():
        screenshot_path.unlink()

    doctor.check_screenshot()

    assert not screenshot_path.exists()
