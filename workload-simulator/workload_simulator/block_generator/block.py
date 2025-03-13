from dataclasses import dataclass, field
from typing import List


@dataclass
class UserUpdateInfo:
    round_id: int
    n_new_users: int

@dataclass
class ExternalBudgetSection:
    total_budget: dict
    dnf: dict

    info: list

@dataclass
class Block:
    id: int

    budget_by_section: List[ExternalBudgetSection]

    privacy_unit: str

    n_users: int
    created: int
    retired: int
    privacy_unit_selection: list = None
    total_budget: dict = None # optional now
    request_ids: list = field(default_factory=list)
