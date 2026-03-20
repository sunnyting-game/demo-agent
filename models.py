from __future__ import annotations

from pydantic import BaseModel, Field


class Module(BaseModel):
    name: str
    description: str
    status: str  # "complete" | "partial" | "stub" | "empty"


class Gap(BaseModel):
    type: str  # "empty_directory" | "todo" | "missing_feature" | "placeholder"
    location: str
    description: str


class Opportunity(BaseModel):
    title: str
    what: str
    why: str
    value: list[str]  # e.g. ["Execution", "Quality"]


class Proposal(BaseModel):
    title: str
    what: str      # 1-2 sentence description
    approach: str  # how the agent would execute it using available tools
    effort: str    # "quick win" | "medium" | "larger task"


class ProjectGoal(BaseModel):
    project_goal: str
    success_criteria: list[str]
    value_drivers: list[str]


class ProjectUnderstanding(BaseModel):
    project_name: str
    purpose: str
    tech_stack: list[str] = Field(default_factory=list)
    modules: list[Module] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    recent_focus: str = ""
    notable_observations: list[str] = Field(default_factory=list)
    raw_summary: str = ""
