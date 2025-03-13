

import math
import typing
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


from workload_simulator.block_generator.block import (Block,
                                                      ExternalBudgetSection,
                                                      UserUpdateInfo)
from workload_simulator.policy.attribute_tagging import (
    AttributeCategory, AttributeTagging, CategoryCorrelationLevel)
from workload_simulator.policy.rules import (
    AttributeRule, ExtensionRule, Rule, RuleId, SubsetRuleId, create_poset,
    deactivate_rules_due_to_covered_rules, deactivate_rules_due_to_time)
from workload_simulator.request_generator.mode import BaseModeEncoder
from workload_simulator.request_generator.request import Request
from workload_simulator.request_generator.time_unit import (
    get_latest, privacy_unit_to_timedelta)
from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import (Attribute, AttributeName,
                                              BaseBudget, BudgetConfig,
                                              BudgetRelaxation, CategoryData,
                                              DomainRange,
                                              Schema, TimePrivacyUnit)


@dataclass
class NewPolicyModeEncoder(BaseModeEncoder):

    name: str

    main_privacy_unit: TimePrivacyUnit
    aux_time_privacy_unit: TimePrivacyUnit

    budget_relaxations: Dict[BudgetRelaxation, Callable[[BaseBudget], BaseBudget]]
    budget: BudgetConfig
    attributes: Dict[AttributeName, AttributeTagging]
    categories: List[CategoryData]
    category_level_extensions: Dict[CategoryCorrelationLevel, Callable[[BaseBudget], BaseBudget]]

    rules_log: typing.Optional[typing.Dict] = field(default_factory=dict)

    def get_rules_log(self):
        return self.rules_log

    def init(self):
        units = [self.main_privacy_unit]
        if self.aux_time_privacy_unit is not None:
            units.append(self.aux_time_privacy_unit)

        optimize_time_rules = False # TODO: where to get config from
        poset, rules = self._build_rule_set(units)

        # print(poset.repr_full())

        info1 = get_rules_info(rules)
        print(f"Initial Rule Set: {info1}")
        self.rules_log["initial"] = info1

        print("  -> deactivating covered rules..")
        deactivate_rules_due_to_covered_rules(poset, rules)

        if optimize_time_rules:
            info2 = get_rules_info(rules)
            print(f"Rule Set: {info2}")
            print("  -> deactivating rules due to time..")
            self.rules_log["after_covered_deactivation"] = info2
            deactivate_rules_due_to_time(rules)
        else:
            warnings.warn("Time optimization is disabled")

        info3 = get_rules_info(rules)
        print(f"Final Rule Set: {info3}")
        self.rules_log["final"] = info3

        self.rules: Dict[RuleId, Rule] = rules

        # building the encoding idx: {unit: {rule_id: idx}}
        self.idx_by_rule_id = {unit: {} for unit in units}
        sorting_ties =  lambda s: tuple([self.rules[s.element].main_rule.name] + [r.name for r in self.rules[s.element].extension_rules])
        for shell in poset.shells_topological(include_special=False, reverse=True, key=sorting_ties):
            rule_id: RuleId = shell.element
            rule: Rule = self.rules[rule_id]

            for unit in units:
                if rule.is_active_rule(unit):
                    idx = len(self.idx_by_rule_id[unit])
                    self.idx_by_rule_id[unit][rule_id] = idx

        encoding_info = {unit: len(self.idx_by_rule_id[unit]) for unit in units}
        print(f"Encoding Size per Privacy Unit: {encoding_info}")

        n = encoding_info[units[0]]
        if not all(encoding_info[unit] == n for unit in units):
            raise ValueError("Encoding size per privacy unit must be the same as long as the planner does not support multiple different PA schemas")






    def assign_mode(self, schema: Schema, requests: List[Request], user_updates: List[UserUpdateInfo]) -> Tuple[Schema | List[Block] | List[Request]]:

        self.init()

        units = [self.main_privacy_unit.name]
        if self.aux_time_privacy_unit is not None:
            units.append(self.aux_time_privacy_unit.name)

        schema.privacy_units = units
        schema.block_sliding_window_size = self.n_rounds_active()

        schema.budget_config = self.budget.export()

        # TODO: In later version, we could have a different PA scheme per privacy unit-> then here we would need to pass both to the schema (and the request would have a dnf per privacy unit)
        attribute: Attribute = self.get_schema_attribute(self.main_privacy_unit)
        schema.attributes.append(attribute)

        # TODO [nku] should this be attribute info or rules_info or something
        schema.attribute_info = self.get_rules_info()

        print(f"Virtual Block Domain Size: {schema.virtual_block_domain_size():_}")

        # convert user updates into blocks
        blocks = []
        for user_update in user_updates:

            budget_by_section = self.get_budget_sections(unit=self.main_privacy_unit)


            user_blk = Block(
                id=user_update.round_id * 100,
                budget_by_section=budget_by_section,
                privacy_unit=self.main_privacy_unit.name,
                privacy_unit_selection=None, # for user block this is always None
                n_users=user_update.n_new_users, # does not matter
                created=user_update.round_id,
                retired=user_update.round_id + self.n_rounds_active(),
            )
            blocks.append(user_blk)

            if self.aux_time_privacy_unit is not None:
                budget_by_section_time = self.get_budget_sections(unit=self.aux_time_privacy_unit)

                # assert len(budget_by_section_time) == len(budget_by_section), "budget sections must be the same for both privacy units for now"
                # TODO for [nku]: Make above assert more robust
                    # Explanation: The problem is that I'm just checking that they have the same length but actually the content is important.
                    # In this example, you see that budget_by_section has only one entry which covers both 0,1
                    # While budget_by_section_time has two entries, 0, and 1 separately.
                    # budget_by_section_time=[ExternalBudgetSection(total_budget={'EpsDeltaDp': {'eps': 3.0, 'delta': 1e-07}}, dnf={'conjunctions': [{'predicates': {'Rules': {'In': [1]}}}]}, info=["RuleId(main=SubsetRuleId(subset=frozenset({'a1'})), extensions=(SubsetRuleId(subset=frozenset({'NONE'})),), duplicate_id=None)"]), ExternalBudgetSection(total_budget={'EpsDeltaDp': {'eps': 5.0, 'delta': 1e-07}}, dnf={'conjunctions': [{'predicates': {'Rules': {'In': [0]}}}]}, info=["RuleId(main=SubsetRuleId(subset=frozenset({'a0', 'a1'})), extensions=(SubsetRuleId(subset=frozenset({'NONE'})),), duplicate_id=None)"])]
                    # budget_by_section=[ExternalBudgetSection(total_budget={'EpsDeltaDp': {'eps': 5.0, 'delta': 1e-07}}, dnf={'conjunctions': [{'predicates': {'Rules': {'In': [0, 1]}}}]}, info=["RuleId(main=SubsetRuleId(subset=frozenset({'a1'})), extensions=(SubsetRuleId(subset=frozenset({'NONE'})),), duplicate_id=None)", "RuleId(main=SubsetRuleId(subset=frozenset({'a0', 'a1'})), extensions=(SubsetRuleId(subset=frozenset({'NONE'})),), duplicate_id=None)"])]

                time_blocks = get_user_time_blocks(user_block=user_blk, privacy_unit=self.aux_time_privacy_unit, budget_by_section=budget_by_section_time, scenario=self.scenario)
                assert len(time_blocks) < 100, "more than 100 privacy units are not supported (-> collisions in block ids)"
                blocks.extend(time_blocks)
                #n_blocks_round = len(time_blocks)
                #print(f"Blocks: round={user_blk.created} n_blocks={n_blocks_round}  (total=12*{n_blocks_round})")


        # adjust requests
        for req in requests:

            # filter all costs by selected privacy unit
            all_rdp_cost = req.request_info.cost_poisson_amplified.rdp
            costs = {self.main_privacy_unit.name: all_rdp_cost[self.main_privacy_unit.name]}
            req.set_request_cost(costs)

            if self.aux_time_privacy_unit is not None:
                costs[self.aux_time_privacy_unit.name] = all_rdp_cost[self.aux_time_privacy_unit.name]

            # adjust request dnf to also include the attribute selections
            # TODO: In later version, we could have a different PA scheme per privacy unit-> then here we would need to pass both to the request
            idxs: list[int] = self.get_rule_selection(self.main_privacy_unit, request=req)
            req.dnf = _to_dnf(rule_idxs=idxs, base_dnf=req.dnf)


            # Compare categories and attribute metadata for correctness
            expected_categories = set()
            expected_budget_relaxations = set()
            for rid, rule in self.rules.items():
                attribute_rule: AttributeRule = rule.main_rule

                # NOTE: a category could also only have one attribute
                if rule.match(req):
                    if attribute_rule.type == "PerCategory":
                        expected_categories.add(attribute_rule.name)

                    assert len(rule.extension_rules) == 1
                    extension_rule: ExtensionRule = rule.extension_rules[0]
                    expected_budget_relaxations.add(extension_rule.name)

            assert set(req.categories) == expected_categories, f"categories do not match: \n  {sorted(req.categories)} != \n  {sorted(expected_categories)}"
            assert set(req.relaxations) == expected_budget_relaxations, f"budget relaxations do not match:\n  {sorted(req.relaxations)} != \n  {sorted(expected_budget_relaxations)}"

        return schema, blocks, requests


    def get_schema_attribute(self, unit: TimePrivacyUnit):
        return Attribute(name="Rules", value_domain=DomainRange(min=0, max=len(self.idx_by_rule_id[unit])-1))


    def get_rule_selection(self, unit: TimePrivacyUnit, request: Request) -> List[int]:
        idxs = []
        for rid, rule in self.rules.items():
            rule: Rule = rule
            if rule.is_active_rule(unit) and rule.match(request):
                idxs.append(self.idx_by_rule_id[unit][rid])

        return idxs


    def get_budget_sections(self, unit: TimePrivacyUnit) -> List[ExternalBudgetSection]:

        budgets = {} #{budget:  List[idx]}

        for rid, rule in self.rules.items():
            rule: Rule = rule

            if rule.is_active_rule(unit):
                budget: BaseBudget = rule.budget()[unit]
                if budget not in budgets:
                    budgets[budget] = []
                budgets[budget].append(rid)

        budget_sections = []
        for budget, rule_ids in budgets.items():

            idxs = [self.idx_by_rule_id[unit][rid] for rid in rule_ids]
            info = [repr(rid) for rid in rule_ids]
            dnf = _to_dnf(rule_idxs=idxs)
            section = ExternalBudgetSection(total_budget=budget.asdict(), dnf=dnf, info=info)
            budget_sections.append(section)

        return budget_sections

    def get_rules_info(self):

        cats = {}
        for category in self.categories:
            level = category.privacy_risk_level
            if level.name not in cats:
                cats[level.name] = []
            cats[level.name].append(category.name)

        idx_lookup = {}
        for unit, rule_idxs in self.idx_by_rule_id.items():
            idx_lookup[unit.name] = {}
            for rule_id, idx in rule_idxs.items():
                rule_id: RuleId = rule_id
                rule: Rule = self.rules[rule_id]
                infos = [rule.main_rule.info()] + [r.info() for r in rule.extension_rules]
                idx_lookup[unit.name][idx] = infos

        rules_info = {
            "idx_lookup": idx_lookup,
            "categories": cats,
            "attributes": {name: tag.info() for name, tag in self.attributes.items()}
        }

        return rules_info

    def short(self):
        return f"dpolicy-{self.name}"


    def _get_per_attribute_budget(self, attribute: AttributeName, unit: TimePrivacyUnit):
        attribute_risk_level = self.attributes[attribute].attribute_risk_level

        budget = self.budget.per_attribute_budget[unit][attribute_risk_level]

        # TODO [later] would apply the overrides here
        #self.budget.per_attribute_budget_overrides
        assert len(self.budget.per_attribute_budget_overrides) == 0, "per-attribute: overrides not yet implemented"

        return budget

    def _get_per_category_budget(self, category: AttributeCategory, category_level: CategoryCorrelationLevel, unit: TimePrivacyUnit):

        categories = [c for c  in self.categories if c.name == category]
        assert len(categories) == 1, f"category {category} is not exactly once available: {categories=}"
        category: CategoryData = categories[0]

        base_budget = self.budget.per_category_budget[unit][category.privacy_risk_level]

        # TODO [later] would apply the overrides here
        #self.budget.per_category_budget_overrides
        assert len(self.budget.per_category_budget_overrides) == 0, "per-category: overrides not yet implemented"

        budget = self.category_level_extensions[category_level](base_budget)


        return budget

    def _get_total_budget(self, unit: TimePrivacyUnit):
        return self.budget.total_budget[unit]


    def _build_rule_set(self, units: List[TimePrivacyUnit]):

        relaxation_rules = []
        for relax, translation_function in self.budget_relaxations.items():
            deps = []
            for other in self.budget_relaxations:
                if other.value <= relax.value:
                    deps.append(other.name)
            relaxation_rules.append(ExtensionRule(SubsetRuleId(frozenset(deps)), f"{relax.name}", "BudgetRelaxation", translation_function))


        attribute_rules = []
        categories_attributes = {c.name: {} for c in self.categories}

        cat_levels: set[CategoryCorrelationLevel] = set()
        for tagging in self.attributes.values():
            for cat, cat_level in tagging.category_assignment.items():
                cat_levels.add(cat_level)

        for name, tagging in self.attributes.items():

            budget = {unit: self._get_per_attribute_budget(name, unit=unit) for unit in units}

            # per attribute rules
            attribute_rules.append(AttributeRule(SubsetRuleId(frozenset([name])), f"{name}", "PerAttribute", budget))

            # prep category rules
            for cat, cat_level in tagging.category_assignment.items():

                for lvl in cat_levels:
                    if int(lvl.value) >= int(cat_level.value):
                        if lvl not in categories_attributes[cat]:
                            categories_attributes[cat][lvl] = []
                        categories_attributes[cat][lvl].append(name)


        # per category rules
        for cat, info in categories_attributes.items():
            for cat_level, attrs in info.items():
                budget = {unit: self._get_per_category_budget(category=cat, category_level=cat_level, unit=unit) for unit in units}
                attribute_rules.append(AttributeRule(SubsetRuleId(frozenset(attrs)), f"{cat}-{cat_level.name}", "PerCategory", budget))


        total_budget = {unit: self._get_total_budget(unit) for unit in units}
        attribute_rules.append(AttributeRule(SubsetRuleId(frozenset(self.attributes.keys())), "All", "All", total_budget))

        poset, rules = create_poset(attribute_rules, relaxation_rules)

        return poset, rules



def _to_dnf(rule_idxs: List[int], base_dnf: dict = None):

    # TODO [later]: Could compact the budget_d with intervals library -> if only a single interval remains, can use {"Between": {"min": x, "max": y}}
    #                                                    -> if only a single value is in list, can use {"Eq": x}
    rule_idxs = sorted(rule_idxs)

    predicates = {"Rules": {"In": rule_idxs}}


    if base_dnf is None:
        dnf = {
            "conjunctions": [{"predicates": predicates}]
        }
        return dnf
    else:
        for conjunction in base_dnf["conjunctions"]:
            conjunction["predicates"].update(predicates)
        return base_dnf

def get_rules_info(rules: Dict[RuleId, Rule]):

    info = None

    for rid, rule in rules.items():
        rule: Rule = rule

        if info is None:
            # we get the units from the first rule
            info = {unit.name: {"n_active": 0, "n_inactive": 0} for unit in rule.budget().keys()}

        for unit, budget in rule.budget().items():
            is_active = rule.is_active.get(unit, True)
            #print(f"{is_active} {rid=}: {unit=}, {budget=}")
            if is_active:
                info[unit.name]["n_active"] += 1
            else:
                info[unit.name]["n_inactive"] += 1
    return info


def get_user_time_blocks(user_block: Block, privacy_unit: TimePrivacyUnit, budget_by_section, scenario: ScenarioConfig):

    U64_MAX = 18_446_744_073_709_551_614

    round_id = user_block.created

    n_units_per_active_window = math.ceil(scenario.active_time_window / privacy_unit_to_timedelta(privacy_unit))

    latest = get_latest(round_id=round_id, privacy_unit=privacy_unit, scenario=scenario)

    blocks = []

    def create_block(selection):
        blk = Block(
            id=user_block.id + len(blocks) + 1,
            budget_by_section=budget_by_section,
            privacy_unit=privacy_unit.name,
            privacy_unit_selection=selection,
            n_users=None,
            created=user_block.created,
            retired=user_block.retired,
            total_budget=None # optional now
        )

        blocks.append(blk)


    n_guaranteed = 1 # TODO [hly] increasing this to e.g, 3 allows for more parallel composition between months (as we can guarantee that multiple months are present as individual time-based blocks across all user groups)

    past = [[0, latest-n_guaranteed]]
    create_block(past)

    n_singletons =  n_units_per_active_window + n_guaranteed

    for i in range(n_singletons):
        cur_i = [[latest + i, latest + i]]
        create_block(cur_i)

    future = [[latest + n_singletons, U64_MAX]]
    create_block(future)

    return blocks