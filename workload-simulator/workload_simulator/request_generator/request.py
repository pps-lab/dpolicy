from copy import deepcopy
from workload_simulator.request_generator.partitioning import BasePopulationSelection
from workload_simulator.request_generator.metadata import BaseAttributeSelection, BaseBudgetRelaxationSelection
from workload_simulator.request_generator.mechanism import MechanismInfo, MechanismWrapper

from dataclasses import dataclass

import random

from typing import List, Optional, Dict

from workload_simulator.request_generator.time_unit import BaseTimeSelection
from workload_simulator.schema.schema import Schema

from workload_simulator.policy.attribute_tagging import CategoryCorrelationLevel, AttributeTagging
from workload_simulator.schema.schema import BudgetRelaxation, AttributeName


@dataclass
class Request:
    request_id: int
    created: int # round id
    dnf: dict
    request_info: MechanismInfo
    workload_info: dict

    attributes: List[AttributeName]
    categories: Optional[List[str]]
    relaxations: Optional[List[str]]

    budget_relaxation: str
    dnf_pa: dict = None

    privacy_unit_selection: dict = None

    # supplied later
    profit: int = None
    #n_users: int = None
    request_cost: dict = None

    def __post_init__(self):
        if self.dnf_pa is None:
            self.dnf_pa = deepcopy(self.dnf)

    def set_utility(self, utility, info):
        assert isinstance(utility, int), f"utility must be an integer, but is {type(utility)}"
        self.profit = utility
        self.request_info.utility_info = info


    def set_n_users(self, n_users):
        assert isinstance(n_users, int), f"n_users must be an integer, but is {type(n_users)}"
        self.n_users = n_users

    def set_request_cost(self, request_cost):
        assert isinstance(request_cost, dict), f"request_cost must be a dict, but is {type(request_cost)}"
        self.request_cost = request_cost



@dataclass
class Category:

    population_gen: BasePopulationSelection
    attribute_gen: BaseAttributeSelection
    budget_relaxation_gen: BaseBudgetRelaxationSelection
    time_gen: BaseTimeSelection
    mechanism_gen: MechanismWrapper
    budget_relaxations: List[BudgetRelaxation]
    weight: int

    def info(self):
        return {
            "population": self.population_gen.info(),
            "attribute": self.attribute_gen.info(),
            "budget_relaxation": self.budget_relaxation_gen.info(),
            "time": self.time_gen.info(),
            "mechanism": self.mechanism_gen.info(),
            "weight": self.weight
        }

    def generate_request(self, request_id: int, round_id: int, workload_info: dict):

        population_dnf, population_domain_size = self.population_gen.generate()

        attributes, sampled_categories = self.attribute_gen.generate()
        budget_relaxation = self.budget_relaxation_gen.generate()

        alphas = workload_info["cost_config"]["alphas"]

        mechanism_info = self.mechanism_gen.generate(alphas)
        mechanism_info.selection = {
            "n_conjunctions": len(population_dnf["conjunctions"]),
            "n_virtual_blocks": population_domain_size
        }

        # TODO: Find the supported policy values some other way
        category_levels = {CategoryCorrelationLevel.MEMBER_STRONG_WEAK, CategoryCorrelationLevel.MEMBER_STRONG, CategoryCorrelationLevel.MEMBER}
        categories_expanded = set()

        # find the categories from the attributes
        for attr, tag in attributes.items():
            for category, level in tag.category_assignment.items():
                for lvl in category_levels:
                    if int(lvl.value) >= int(level.value):
                        categories_expanded.add(f"{category}-{lvl.name}")

        relaxations = set()
        for relax_level in self.budget_relaxations:
            if relax_level.value >= budget_relaxation.value:
                relaxations.add(relax_level.name)


        privacy_unit_selection = self.time_gen.generate(round_id)
        ### TODO: In mode encoder where we encode the categories and relaxations, verify correctness


        r = Request(
            request_id=request_id,
            created=round_id,
            dnf=population_dnf,
            request_info=mechanism_info,
            workload_info=workload_info,
            attributes=list(attributes.keys()),
            categories=list(categories_expanded),
            budget_relaxation=budget_relaxation.name,
            relaxations=list(relaxations),
            privacy_unit_selection=privacy_unit_selection,
        )

        return r

@dataclass
class Workload:

    name: str
    schema: Schema
    distribution: List[Category]


    _request_id_ctr: int = 0

    def __post_init__(self):

        self.weights = [cat.weight for cat in self.distribution]
        self.population = list(range(len(self.weights)))

        self.workload_info = {
            "name": self.name,
            "size": None,
            "mechanism_mix": [d.info() for d in self.distribution],
            "cost_config": self.schema.cost_config()
        }

    def empty_cost(self):
        rdp = len(self.workload_info["cost_config"]["alphas"]) * [0.0]
        return {
            "Rdp": {
                "eps_values": rdp
            }
        }

    def generate_request(self, round_id: int):

        request_category_idx = random.choices(population=self.population, weights=self.weights, k=1)[0]

        request = self.distribution[request_category_idx].generate_request(request_id=self._request_id_ctr, round_id=round_id, workload_info=self.workload_info)

        self._request_id_ctr += 1

        return request
