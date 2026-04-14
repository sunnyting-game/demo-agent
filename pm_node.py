"""
PM node — evaluates proposals across 5 stages and produces a CEHandoff or rejection.

Stages
------
1. Value Judgment      — per proposal, single LLM call
2. Assumption Verify   — per proposal, agentic loop
3. Feasibility         — per proposal, agentic loop (reference only, no filtering)
4. Selection           — single LLM call across all surviving proposals
5. Self-Critique       — single LLM call on chosen proposal
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic

from models import (
    CEHandoff,
    ExtractedAssumption,
    FeasibilityAnalysis,
    JudgmentEntry,
    PMDecision,
    PMReport,
    Proposal,
    ProjectGoal,
    ProjectUnderstanding,
    RejectionGuidance,
    UnverifiedAssumption,
    VerifiedAssumption,
    FailureRisk,
)
from pm_prompts import (
    PM_EXTRACT_SYSTEM_PROMPT,
    PM_EXTRACT_USER_TEMPLATE,
    PM_VERIFY_SYSTEM_PROMPT,
    PM_VERIFY_USER_TEMPLATE,
)
from tools import ALL_TOOLS, execute_tools

_MODEL = "claude-sonnet-4-6"
MAX_REJECTIONS = 3


# ── Internal bundle (pure local, not a schema class) ──────────────────────────

@dataclass
class _ProposalAnalysis:
    verified_assumptions  : list[VerifiedAssumption]    = field(default_factory=list)
    unverified_assumptions: list[UnverifiedAssumption]  = field(default_factory=list)
    feasibility           : FeasibilityAnalysis | None  = None


# ── Node entry point ──────────────────────────────────────────────────────────

async def pm_node(state: dict) -> dict:
    proposals             : list[Proposal]        = state["proposals"]
    project_goal          : ProjectGoal           = state["project_goal"]
    project_understanding : ProjectUnderstanding  = state["project_understanding"]
    rejection_count       : int                   = state["pm_replan_count"]
    project_path          : str                   = state["project_path"]

    judgment_log: list[JudgmentEntry] = []

    # ── Stage 1: Value Judgment ───────────────────────────────────────────────
    # Single LLM call per proposal
    # proposal.source_opportunity provides the Goal → Opportunity → Proposal chain
    surviving: list[Proposal] = []
    for proposal in proposals:
        entry = _value_judgment_call(proposal, project_goal, project_understanding)
        judgment_log.append(entry)
        if entry.verdict != "fail":
            surviving.append(proposal)

    if not surviving:
        return _build_rejection(judgment_log, rejection_count)

    # ── Stage 2: Assumption Verification ─────────────────────────────────────
    # Per proposal: extract assumptions → verify top 3 → agentic loop
    # LLM judges fatal assumptions via JudgmentEntry verdict="fail"
    analysis: dict[str, _ProposalAnalysis] = {}
    second_surviving: list[Proposal] = []

    for proposal in surviving:
        verified, unverified, entry = await _assumption_verification_loop(
            proposal, project_understanding, project_goal, project_path
        )
        judgment_log.append(entry)
        analysis[proposal.title] = _ProposalAnalysis(
            verified_assumptions=verified,
            unverified_assumptions=unverified,
        )
        if entry.verdict != "fail":
            second_surviving.append(proposal)

    if not second_surviving:
        return _build_rejection(judgment_log, rejection_count)

    # ── Stage 3: Feasibility ──────────────────────────────────────────────────
    # Per proposal: identify 3 key files → verify dependency / underestimation /
    # conflicts. No filtering — reference only.
    for proposal in second_surviving:
        feasibility, entry = _feasibility_loop(
            proposal, project_understanding, project_path
        )
        judgment_log.append(entry)
        analysis[proposal.title].feasibility = feasibility

    # ── Stage 4: Selection ────────────────────────────────────────────────────
    # Single LLM call — PM receives all proposals + full analysis bundle
    chosen, aligned_criteria, confidence, entry = _selection_call(
        second_surviving, analysis, project_goal
    )
    judgment_log.append(entry)

    # ── Stage 5: Self-Critique ────────────────────────────────────────────────
    # Single LLM call — "why would this proposal fail?"
    # Output becomes CE's investigation starting points
    failure_risks, entry = _self_critique_call(
        chosen, analysis[chosen.title], project_goal
    )
    judgment_log.append(entry)

    # ── Build output ──────────────────────────────────────────────────────────
    a = analysis[chosen.title]
    ce_handoff = CEHandoff(
        chosen_proposal=chosen,
        aligned_success_criterion=aligned_criteria,
        verified_assumptions=a.verified_assumptions,
        unverified_assumptions=a.unverified_assumptions,
        failure_risks=failure_risks,
        feasibility_analysis=a.feasibility,
        feasibility_confidence=confidence,
    )
    return {
        "pm_decision":        "approved",
        "pm_decision_detail": PMDecision(judgment_log=judgment_log, ce_handoff=ce_handoff),
    }


# ── Rejection builder ─────────────────────────────────────────────────────────

def _build_rejection(judgment_log: list[JudgmentEntry], rejection_count: int) -> dict:
    new_count = rejection_count + 1

    if new_count >= MAX_REJECTIONS:
        pm_report = PMReport(
            total_iterations=new_count,
            attempted_directions=list({e.target_proposal_title for e in judgment_log}),
            conclusion="needs_user_guidance",
        )
        return {
            "pm_decision":        "rejected",
            "pm_decision_detail": PMDecision(judgment_log=judgment_log, pm_report=pm_report),
            "pm_replan_count":    new_count,
        }

    reason, guidance = _rejection_analysis_call(judgment_log)
    rejection = RejectionGuidance(
        reason=reason,
        target="opportunity",
        guidance=guidance,
        iteration=new_count,
    )
    return {
        "pm_decision":          "rejected",
        "pm_decision_detail":   PMDecision(judgment_log=judgment_log, rejection=rejection),
        "pm_replan_count":      new_count,
        "opportunity_guidance": guidance,
    }


# ── Stage helpers (stubs) ─────────────────────────────────────────────────────

def _value_judgment_call(
    proposal: Proposal,
    project_goal: ProjectGoal,
    project_understanding: ProjectUnderstanding,
) -> JudgmentEntry:
    # proposal.source_opportunity provides the Opportunity context for chain tracing
    raise NotImplementedError


async def _assumption_verification_loop(
    proposal: Proposal,
    project_understanding: ProjectUnderstanding,
    project_goal: ProjectGoal,
    project_path: str,
) -> tuple[list[VerifiedAssumption], list[UnverifiedAssumption], JudgmentEntry]:
    """Stage 2 orchestrator: Phase 1 (extract) → Phase 2 (verify)."""
    all_assumptions = await _extract_assumptions(proposal, project_understanding, project_goal)
    top_3 = all_assumptions[:3]

    if not top_3:
        entry = JudgmentEntry(
            stage="assumption_verification",
            target_proposal_title=proposal.title,
            verdict="pass",
            finding="No significant unconfirmed assumptions found in this proposal.",
            evidence=None,
        )
        return [], [], entry

    return await _verify_assumptions(top_3, proposal.title, project_path)


async def _extract_assumptions(
    proposal: Proposal,
    project_understanding: ProjectUnderstanding,
    project_goal: ProjectGoal,
) -> list[ExtractedAssumption]:
    """
    Phase 1 — single structured call, no tools.
    Returns all assumptions sorted high → medium → low. Caller takes [:3].
    source_opportunity is included so the model has the Opportunity's why + value.
    """
    proposal_dict = proposal.model_dump()

    user_prompt = PM_EXTRACT_USER_TEMPLATE.format(
        proposal_json=json.dumps(proposal_dict, indent=2),
        project_understanding_json=project_understanding.model_dump_json(indent=2),
        project_goal_json=project_goal.model_dump_json(indent=2),
    )

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=PM_EXTRACT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return _parse_extracted_assumptions(text)


async def _verify_assumptions(
    assumptions: list[ExtractedAssumption],
    proposal_title: str,
    project_path: str,
) -> tuple[list[VerifiedAssumption], list[UnverifiedAssumption], JudgmentEntry]:
    """
    Phase 2 — agentic loop.
    Model calls tools until all assumptions are classified, then outputs final JSON
    in a tool-free turn. Loop ends on stop_reason == "end_turn".
    """
    user_prompt = PM_VERIFY_USER_TEMPLATE.format(
        proposal_title=proposal_title,
        assumptions_json=json.dumps([a.model_dump() for a in assumptions], indent=2),
        project_path=project_path,
    )

    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    client = anthropic.AsyncAnthropic()

    while True:
        response = await client.messages.create(
            model=_MODEL,
            max_tokens=8192,
            system=PM_VERIFY_SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_results = execute_tools(response.content, project_path)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # stop_reason == "end_turn": model wrote final JSON, no more tool calls
            text = next((b.text for b in response.content if b.type == "text"), "")
            return _parse_verification_output(text, proposal_title)


# ── Parsers ───────────────────────────────────────────────────────────────────

_RISK_ORDER = {"high": 0, "medium": 1, "low": 2}


def _parse_extracted_assumptions(text: str) -> list[ExtractedAssumption]:
    data = _extract_json_array(text)
    if not data:
        return []
    assumptions = []
    for item in data:
        try:
            assumptions.append(ExtractedAssumption(
                claim=item["claim"],
                risk_level=item["risk_level"],
                blocking_reason=item.get("blocking_reason", ""),
            ))
        except (KeyError, ValueError):
            continue
    return sorted(assumptions, key=lambda a: _RISK_ORDER.get(a.risk_level, 3))


def _parse_verification_output(
    text: str,
    proposal_title: str,
) -> tuple[list[VerifiedAssumption], list[UnverifiedAssumption], JudgmentEntry]:
    data = _extract_json_object(text)

    if not data:
        entry = JudgmentEntry(
            stage="assumption_verification",
            target_proposal_title=proposal_title,
            verdict="uncertain",
            finding="Verification output could not be parsed.",
            evidence=None,
        )
        return [], [], entry

    verified: list[VerifiedAssumption] = []
    for item in data.get("verified_assumptions", []):
        try:
            verified.append(VerifiedAssumption(**item))
        except (TypeError, ValueError):
            continue

    unverified: list[UnverifiedAssumption] = []
    for item in data.get("unverified_assumptions", []):
        try:
            unverified.append(UnverifiedAssumption(**item))
        except (TypeError, ValueError):
            continue

    je_raw = data.get("judgment_entry", {})
    entry = JudgmentEntry(
        stage="assumption_verification",
        target_proposal_title=je_raw.get("target_proposal_title", proposal_title),
        verdict=je_raw.get("verdict", "uncertain"),
        finding=je_raw.get("finding", ""),
        evidence=je_raw.get("evidence"),
    )
    return verified, unverified, entry


# ── JSON extraction helpers ───────────────────────────────────────────────────

def _extract_json_array(text: str) -> list | None:
    fenced = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    last = text.rfind("[")
    if last != -1:
        try:
            return json.loads(text[last:])
        except json.JSONDecodeError:
            pass
    return None


def _extract_json_object(text: str) -> dict | None:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    last = text.rfind("{")
    if last != -1:
        try:
            return json.loads(text[last:])
        except json.JSONDecodeError:
            pass
    return None


def _feasibility_loop(
    proposal: Proposal,
    project_understanding: ProjectUnderstanding,
    project_path: str,
) -> tuple[FeasibilityAnalysis, JudgmentEntry]:
    raise NotImplementedError


def _selection_call(
    proposals: list[Proposal],
    analysis: dict[str, _ProposalAnalysis],
    project_goal: ProjectGoal,
) -> tuple[Proposal, list[str], str, JudgmentEntry]:
    raise NotImplementedError


def _self_critique_call(
    proposal: Proposal,
    analysis: _ProposalAnalysis,
    project_goal: ProjectGoal,
) -> tuple[list[FailureRisk], JudgmentEntry]:
    raise NotImplementedError


def _rejection_analysis_call(
    judgment_log: list[JudgmentEntry],
) -> tuple[str, str]:
    raise NotImplementedError
