"""
LangGraph agent definition.

Graph layout:

    [START] ──► scan ──► confirm ──► (approved?) ──► goal ──► opportunities ──► proposal ──► (approved?) ──► [END]
                                         │                                            │
                                         └──(not approved)──► confirm  (loop)        └──(not approved)──► proposal  (loop)
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from confirm_node import confirm_node
from goal_node import goal_node
from models import Opportunity, Proposal, ProjectGoal, ProjectUnderstanding
from opportunities_node import opportunities_node
from proposal_node import proposal_node
from scan_node import scan_node


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    project_path: str
    project_understanding: Optional[ProjectUnderstanding]
    user_summary: str
    approved: bool
    conversation_history: list[dict]
    project_goal: Optional[ProjectGoal]
    opportunities: Optional[list[Opportunity]]
    proposals: Optional[list[Proposal]]
    proposals_approved: bool
    proposal_conversation_history: list[dict]


# ──────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────

def _route_confirm(state: AgentState) -> str:
    return "goal" if state["approved"] else "confirm"


def _route_proposals(state: AgentState) -> str:
    return END if state["proposals_approved"] else "proposal"


def build_agent():
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("scan", scan_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("goal", goal_node)
    graph.add_node("opportunities", opportunities_node)
    graph.add_node("proposal", proposal_node)

    graph.set_entry_point("scan")
    graph.add_edge("scan", "confirm")
    graph.add_conditional_edges("confirm", _route_confirm)
    graph.add_edge("goal", "opportunities")
    graph.add_edge("opportunities", "proposal")
    graph.add_conditional_edges("proposal", _route_proposals)

    return graph.compile()
