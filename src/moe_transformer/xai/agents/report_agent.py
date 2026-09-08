"""Report Agent: the last stage of the pipeline. Renders everything the
upstream agents computed into one markdown report and writes it into
state, same as every other agent -- the CLI decides whether/where to save
it to disk.
"""

from __future__ import annotations

from moe_transformer.xai.agents.base import Agent, AgentState
from moe_transformer.xai.report import render_markdown_report


class ReportAgent(Agent):
    name = "report"

    def run(self, state: AgentState) -> AgentState:
        state["report_markdown"] = render_markdown_report(state)
        return state
