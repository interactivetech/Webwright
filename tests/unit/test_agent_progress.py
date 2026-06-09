from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 10):
    pytest.skip("project requires Python 3.10+", allow_module_level=True)

from webwright.agents.default import DefaultAgent


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def format_message(self, **kwargs):
        return {"role": kwargs["role"], "content": kwargs.get("content", ""), "extra": kwargs.get("extra", {})}

    def query(self, messages):
        self.calls += 1
        return {
            "role": "assistant",
            "content": "thought",
            "extra": {
                "actions": [{"bash_command": "echo hi", "command": "echo hi"}],
                "done": False,
                "final_response": "",
                "raw_response": {},
                "usage": {},
            },
        }

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [{"role": "user", "content": "observation", "extra": {"observation": outputs[0]["observation"]}}]

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {"model": {}}


class _FakeEnv:
    def get_template_vars(self, **kwargs):
        return {}

    def execute(self, action, cwd=""):
        return {"observation": {"success": True, "command": action.get("command", ""), "returncode": 0}}

    def serialize(self):
        return {"environment": {}}

    def close(self):
        return None


def test_agent_emits_progress_lines(capsys, tmp_path: Path) -> None:
    model = _FakeModel()
    env = _FakeEnv()
    agent = DefaultAgent(
        model,
        env,
        system_template="system",
        instance_template="instance",
        output_path=tmp_path / "trajectory.json",
        step_limit=5,
    )
    agent.messages = [
        model.format_message(role="system", content="system"),
        model.format_message(role="user", content="instance"),
    ]

    message = agent.query()
    agent.execute_actions(message)

    captured = capsys.readouterr().out
    assert "[webwright] Step 1/5: querying model" in captured
    assert "[webwright] Step 1: executing action 1/1 (echo hi)" in captured
