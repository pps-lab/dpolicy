from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

from workload_simulator.block_generator.block import Block, UserUpdateInfo
from workload_simulator.policy.attribute_tagging import AttributeTagging
from workload_simulator.policy.budget_policy import TotalBudgetPolicy
from workload_simulator.request_generator.request import Request
from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import (AttributeName, BudgetRelaxation, CategoryData, Schema,
                                              TimePrivacyUnit)

@dataclass
class BaseModeEncoder(ABC):

    scenario: ScenarioConfig

    @abstractmethod
    def assign_mode(self, schema: Schema, requests: List[Request], user_updates: List[UserUpdateInfo]) -> Tuple[Schema | List[Block] | List[Request]]:
        pass

    def config(self):
        return {
            "name": self.__class__.__name__,
        }

    @abstractmethod
    def short(self) -> str:
        pass

    def n_rounds_active(self):
        # for how many rounds is a block active
        return self.scenario.active_time_window // self.scenario.allocation_interval

    def get_rules_log(self):
        return {}


def n_active_blocks(req_created: int, blocks: List[Block]):
    return sum(req_created >= b.created and req_created < b.retired for b in blocks)

@dataclass
class CohereModeEncoder(BaseModeEncoder):


    name: str
    privacy_unit: TimePrivacyUnit
    budget_policy: TotalBudgetPolicy

    attributes: Dict[AttributeName, AttributeTagging]
    categories: List[CategoryData]

    def assign_mode(self, schema: Schema, requests: List[Request], user_updates: List[UserUpdateInfo]) -> Tuple[Schema | List[Block] | List[Request]]:

        # extend schema
        schema.privacy_units = [self.privacy_unit.name]
        schema.block_sliding_window_size = self.n_rounds_active()
        schema.budget_config = self.budget_policy.export()

        cats = {}
        for category in self.categories:
            level = category.privacy_risk_level
            if level.name not in cats:
                cats[level.name] = []
            cats[level.name].append(category.name)

        attribute_info = {
            "categories": cats,
            "attributes": {name: tag.info() for name, tag in self.attributes.items()}
        }
        schema.attribute_info = attribute_info

        total_budget = self.budget_policy.get_total_budget(relaxation=BudgetRelaxation.NONE, privacy_unit=self.privacy_unit)

        # convert user updates into blocks
        blocks = []
        for user_update in user_updates:

            blk = Block(
                id=user_update.round_id * 100,
                budget_by_section=[],
                privacy_unit=self.privacy_unit.name,
                privacy_unit_selection=None,
                total_budget=total_budget.asdict(),
                n_users=user_update.n_new_users,
                created=user_update.round_id,
                retired=user_update.round_id + self.n_rounds_active(),
            )
            blocks.append(blk)


        # adjust requests
        for req in requests:

            # filter all costs by selected privacy unit
            all_rdp_cost = req.request_info.cost_poisson_amplified.rdp
            costs = {self.privacy_unit.name: all_rdp_cost[self.privacy_unit.name]}
            req.set_request_cost(costs)

        return schema, blocks, requests

    def short(self):
        return f"cohere-{self.name}"
