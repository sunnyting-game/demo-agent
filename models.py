from __future__ import annotations

from typing import Literal, Optional

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


class Contract(BaseModel):
    contract_id: str
    type: Literal["RUNNABLE", "STRUCTURAL", "BEHAVIORAL"]
    description: str                         # 自然語言：驗證什麼
    command: str                             # Verify 直接 shell 執行
    expected_exit_code: int = 0
    expected_output: Optional[str] = None   # None = 只 check exit code
    match_mode: Literal["contains", "regex", "exact"] = "contains"


class Task(BaseModel):
    task_id: str
    description: str                         # 自然語言：做什麼、為什麼
    contract_ids: list[str]                  # 指向對應的 Contract
    is_milestone: bool = False               # TP 決定，Verify pass 後觸發 milestone check
    retry_count: int = 0                     # Error Handling 每次 +1
    max_retries: int = 3
