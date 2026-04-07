"""
LangGraph agent definition.

Graph layout:

    [START] ──► scan ──► confirm ──► (approved?) ──► goal ──► opportunities ──► proposal ──► (approved?) ──► [END]
                                         │                                            │
                                         └──(not approved)──► confirm  (loop)        └──(not approved)──► proposal  (loop)
"""

from __future__ import annotations

from typing import Literal, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from confirm_node import confirm_node
from goal_node import goal_node
from models import Contract, Opportunity, Proposal, ProjectGoal, ProjectUnderstanding, Task
from opportunities_node import opportunities_node
from proposal_node import proposal_node
from scan_node import scan_node


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class AgentState(TypedDict):

    # ── Existing Fields（現有 nodes 使用）──────────────────────────────
    project_path                 : str
    project_understanding        : Optional[ProjectUnderstanding]
    user_summary                 : str
    project_goal                 : Optional[ProjectGoal]
    opportunities                : Optional[list[Opportunity]]
    proposals                    : Optional[list[Proposal]]
    conversation_history         : list[dict]
    proposal_conversation_history: list[dict]

    # TODO: remove after confirm/proposal node refactor
    approved          : bool
    proposals_approved: bool

    # ── Routing Fields（conditional edges 讀這些，node 完成後 reset 為 None）──
    pm_decision      : Optional[Literal["approved", "rejected"]]
    ce_sufficient    : Optional[bool]
    tp_status        : Optional[Literal["ready", "needs_context", "proposal_mismatch"]]
    explore_status   : Optional[Literal["has_task", "done", "scope_exceeded"]]
    implement_status : Optional[Literal["complete", "error"]]
    eh_decision      : Optional[Literal["recoverable", "task_replan", "macro_replan", "halt"]]
    verify_passed    : Optional[bool]
    milestone_passed : Optional[bool]

    # ── Task Tracking──────────────────────────────────────────────────
    pending_tasks    : list[Task]        # TP 生成，Explore 逐個抽取
    current_task     : Optional[Task]    # Explore 設，Verify pass 後清
    completed_tasks  : list[Task]        # Verify pass 後移入
    contracts        : list[Contract]    # TP 生成，Verify 執行

    # ── Checkpoint（Rollback 用）──────────────────────────────────────
    milestone_checkpoints: dict[str, str]   # milestone_id → git commit hash

    # ── Guidance──────────────────────────────────────────────────────
    opportunity_guidance: Optional[str]     # PM reject 時寫，Opportunity node 讀
    replan_guidance     : Optional[str]     # EH / macro replan 時帶給下個 node
    pm_replan_count     : int               # 防 PM ↔ Opportunity 無限 loop

    # ── Node Artifacts（建 node 時逐步加入）──────────────────────────
    # pm_decision_detail : PMDecision        ← 建 PM node 時加
    # wcd                : WorkingContextDoc ← 建 CE node 時加
    # change_manifest    : ChangeManifest    ← 建 Explore node 時加
    # execution_log      : ExecutionLog      ← 建 Implement node 時加
    # verify_result      : VerifyResult      ← 建 Verify node 時加


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
