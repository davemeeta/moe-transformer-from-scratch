"""Minimal agent interface shared by every stage of the RouteLens pipeline.

Deliberately not a framework: each agent reads what upstream agents already
wrote into `state` and returns the updated state. Phase 4's Orchestrator is
a plain loop over a list of these, with one conditional edge (Judge can
route back to the Red-Team agent) -- writing that ourselves rather than
adopting LangGraph/AutoGen keeps the multi-agent control flow legible, which
is the point of a learning project about multi-agent systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

AgentState = dict[str, Any]


class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError
