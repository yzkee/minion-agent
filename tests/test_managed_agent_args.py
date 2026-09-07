"""A managed sub-agent's `agent_args` must reach the agent that is built for it.

The parent splats `self.config.agent_args` into its own constructor, so settings
such as `max_steps` and `additional_authorized_imports` are honoured there. The
same keys on a managed sub-agent are only honoured if they are splatted into the
sub-agent's constructor too.
"""

import asyncio

import pytest
from smolagents import CodeAgent

from minion_agent import AgentConfig, AgentFramework, MinionAgent


class _OfflineModel:
    """A model stand-in: constructed, never called, so the test needs no network."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("the test never runs the agent")


def _config(name, agent_args=None):
    return AgentConfig(
        model_id="offline/none",
        name=name,
        description=f"{name} agent",
        agent_type=CodeAgent,
        model_type=lambda **kwargs: _OfflineModel(**kwargs),
        tools=[],
        agent_args=agent_args,
    )


def test_managed_agent_args_reach_the_sub_agent():
    agent = asyncio.run(
        MinionAgent.create_async(
            AgentFramework.SMOLAGENTS,
            _config("supervisor"),
            managed_agents=[
                _config(
                    "worker",
                    {"max_steps": 3, "additional_authorized_imports": ["json"]},
                )
            ],
        )
    )
    child = agent._agent.managed_agents["worker"]
    assert child.max_steps == 3
    assert "json" in child.additional_authorized_imports


def test_parent_agent_args_are_unchanged():
    agent = asyncio.run(
        MinionAgent.create_async(
            AgentFramework.SMOLAGENTS,
            _config("supervisor", {"max_steps": 5}),
        )
    )
    assert agent._agent.max_steps == 5


def test_agent_args_for_another_framework_are_not_passed_on():
    """A managed config that targets a different framework keeps its own args.

    `example_deep_research.py` reuses one AgentConfig both to build a
    DEEP_RESEARCH agent and as a managed agent under a SMOLAGENTS parent. Its
    `agent_args` name deep-research models, which the smolagents agent class
    does not accept.
    """
    research = _config("research_assistant", {"planning_model": "some/model"})
    research.framework = AgentFramework.DEEP_RESEARCH

    agent = asyncio.run(
        MinionAgent.create_async(
            AgentFramework.SMOLAGENTS, _config("supervisor"), managed_agents=[research]
        )
    )
    assert "research_assistant" in agent._agent.managed_agents
