from dataclasses import dataclass, field
from sqlite3 import Time
from typing import List, Literal, Optional, Dict, Callable
from abc import abstractmethod, ABC
import itertools
import warnings

from workload_simulator.request_generator.request import Request
from workload_simulator.schema.schema import BaseBudget, TimePrivacyUnit
from workload_simulator.policy.poset import MutablePoset
from workload_simulator.policy.poset import MutablePosetShell





class BaseRuleId(ABC):

    @abstractmethod
    def __le__(self, other) -> bool:
        pass



@dataclass(frozen=True)
class SubsetRuleId(BaseRuleId):

    subset: frozenset[str]

    def __post_init__(self):
        assert isinstance(self.subset, frozenset)
    def __le__(self, other):
        return self.subset.issubset(other.subset)

    def match(self, other: set[str]) -> bool:
        return not self.subset.isdisjoint(other)




@dataclass(frozen=True)
class RuleId(BaseRuleId):

    main: BaseRuleId
    extensions: tuple[BaseRuleId, ...]

    duplicate_id: Optional[int] = None

    def __post_init__(self):
        assert isinstance(self.extensions, tuple)
        for e in self.extensions:
            assert isinstance(e, BaseRuleId)
        assert isinstance(self.main, BaseRuleId)

    def __le__(self, other: 'RuleId'):
        return self.main <= other.main and all(e <= o for e, o in zip(self.extensions, other.extensions))

    def duplicate(self, new_id: int):
        return RuleId(self.main, self.extensions, new_id)


@dataclass
class AttributeRule:

    id: SubsetRuleId

    name: str

    type: Literal['PerCategory', 'PerAttribute', 'All']

    budget: Dict[TimePrivacyUnit, BaseBudget]

    def __post_init__(self):
        assert isinstance(self.id, SubsetRuleId)
        assert isinstance(self.budget, dict)
        for k, v in self.budget.items():
            assert isinstance(k, TimePrivacyUnit), f"{type(k)}: {k}"
            #assert isinstance(v, float)

    def match(self, request: Request) -> bool:
        return self.id.match(request.attributes)



    def info(self):
        return {
            "name": self.name,
            "attributes": sorted(list(self.id.subset)),
            #"base_budget": self.budget[unit].asdict(),
        }



@dataclass
class ExtensionRule:

    id: SubsetRuleId

    name: str
    type: Literal['BudgetRelaxation']
    budget_extension_function: Callable[[BaseBudget], BaseBudget]

    def budget(self, input: BaseBudget):
        return self.budget_extension_function(input)

    def match(self, request: Request) -> bool:
        # TODO: not generic extension rule but speciufic to budget relaxation
        return self.id.match([request.budget_relaxation])

    def info(self):
        return {
            "name": self.name,
            "labels": sorted(list(self.id.subset)),
        }


@dataclass
class Rule:

    id: RuleId

    main_rule: AttributeRule
    extension_rules: tuple[ExtensionRule, ...]

    is_active: Dict[TimePrivacyUnit, bool]= field(default_factory=dict)
    inactive_reason: Dict[TimePrivacyUnit, List[str]]= field(default_factory=dict)

    inherited_budget: Dict[TimePrivacyUnit, BaseBudget] = field(default_factory=dict)
    inherited_budget_from: Dict[TimePrivacyUnit, RuleId] = field(default_factory=dict)

    def __post_init__(self):
        assert isinstance(self.id, RuleId)
        assert isinstance(self.main_rule, AttributeRule)
        assert isinstance(self.extension_rules, tuple)
        for e in self.extension_rules:
            assert isinstance(e, ExtensionRule)

    def is_active_rule(self, unit: TimePrivacyUnit) -> bool:
        is_active_unit = self.is_active.get(unit, True)
        if is_active_unit:
            return True
        else:

            is_active_for_a_unit = any(self.is_active.get(unit, True) for unit in self.budget().keys())

            if is_active_for_a_unit:
                # as it's active for another unit, and we cannot encode this in the dp planner
                # -> a rule is active if it's active for any unit
                warnings.warn("A rule could be seen as inactive: As long as we the planner does not support multiple different PA schemas, we need to track a rule if it is active for any privacy unit")
                return True
            else:
                # not active for any unit
                return False



    def match(self, request: Request) -> bool:
        return self.main_rule.match(request) and all(e.match(request) for e in self.extension_rules)

    def update_budget(self, parent_ruleid: RuleId, parent_budget: Dict[TimePrivacyUnit, BaseBudget]):

        assert isinstance(parent_budget, dict)


        for unit, my_budget in self.budget().items():
            if parent_budget[unit] <= my_budget:

                # deactivate rule if parent_budget is smaller or equal
                self.is_active[unit] = False

                if unit not in self.inactive_reason:
                    self.inactive_reason[unit] = []
                self.inactive_reason[unit].append(f"deactivated by {parent_ruleid}")


                if parent_budget[unit] < my_budget and (unit not in self.inherited_budget or parent_budget[unit] < self.inherited_budget[unit]):
                    # store inherited budget if parent_budget is the smallest
                    self.inherited_budget[unit] = parent_budget[unit]
                    self.inherited_budget_from[unit] = parent_ruleid


    def budget(self):

        budget_d = {}
        for unit in self.main_rule.budget.keys():

            if unit in self.inherited_budget:
                b = self.inherited_budget[unit]
            else:
                b = self.main_rule.budget[unit]

                for e in self.extension_rules:
                    b = e.budget(b)

            budget_d[unit] = b

        return budget_d



def create_poset(attributes_rules: List[AttributeRule], deployment_rules: List[ExtensionRule]):


    rules_multi = {}

    for attr_rule, depl_rule in itertools.product(attributes_rules, deployment_rules):
        assert isinstance(attr_rule, AttributeRule)
        assert isinstance(depl_rule, ExtensionRule)

        rid = RuleId(attr_rule.id, (depl_rule.id, ))
        rule = Rule(rid, attr_rule, (depl_rule, ))
        if rid not in rules_multi:
            rules_multi[rid] = []
        rules_multi[rid].append(rule)

    rules = {}

    poset = MutablePoset()

    for rid, rule_list in rules_multi.items():
        rid: RuleId = rid
        rule_list: List[Rule] = rule_list
        if len(rule_list) == 1:
            # single rule with this id -> no need to merge
            rules[rid] = rule_list[0]
            poset.add(rid)
        else:
            # merge rules with the same id -> find the rule with the smallest budget (and deactivate the others)

            def min_rule_idx(rule_list: List[Rule]) -> int:

                for i, rule1 in enumerate(rule_list):
                    rule1_budget = rule1.budget()

                    if all(all(rule1_budget[unit] <= budget2 for unit, budget2 in rule2.budget().items()) for rule2 in rule_list):
                        # across all rules, rule1 has the smallest budget for all privacy units,
                        return i
                raise ValueError(f"Could not find a rule with the smallest budget in {rule_list}")

            min_idx = min_rule_idx(rule_list)
            min_rule: Rule = rule_list[min_idx]

            for i, rule in enumerate(rule_list):
                rid_new = rid.duplicate(i)
                rule.id = rid_new

                if i == min_idx:
                    # from the duplicates, only add the rule with the smallest budget to the poset
                    poset.add(rid_new)
                else:
                    # deactivate all except for the smallest
                    rule: Rule = rule
                    for unit in rule.budget().keys():
                        rule.is_active[unit] = False
                        rule.inactive_reason[unit] = f"duplicate rule id with smaller budget"

                rules[rid_new] = rule

    return poset, rules




def deactivate_rules_due_to_covered_rules(poset, rules):
    maximal_element = list(poset.maximal_elements())
    assert len(maximal_element) == 1, f"Expected one maximal element, got {maximal_element}"
    maximal_element_shell = poset.shell(maximal_element[0])
    #print(f"{maximal_element=}  {rules[maximal_element.element].budget()=}")

    def deactivate_children(shell: MutablePosetShell): #, budget: Dict[TimePrivacyUnit, Budget]

        #assert isinstance(budget, dict), f"{budget=}"

        rule = rules[shell.element]
        budget = rule.budget()

        for child_shell in shell.predecessors():

            if child_shell.is_null():
                continue

            child_rule: Rule = rules[child_shell.element]
            child_rule.update_budget(shell.element, budget)

            #assert isinstance(child_budget, dict), f"{child_budget=}  {budget=}  {child_rule=}"

            deactivate_children(child_shell)

    deactivate_children(maximal_element_shell)


def deactivate_rules_due_to_time(rules):

    for _rid, rule in rules.items():

        rule_budget = rule.budget()

        privacy_units = sorted(rule_budget.keys(), key=lambda x: x.value,reverse=True)

        min_budget = None

        for unit in privacy_units:
            if min_budget is None:
                # first unit
                min_budget = rule_budget[unit]
            else:
                # if a privacy unit with a "larger" unit has the same or a smaller budget, then the rule is inactive
                if rule_budget[unit] < min_budget:
                    min_budget = rule_budget[unit]
                else:
                    if unit not in rule.inactive_reason:
                        rule.inactive_reason[unit] = []
                    rule.inactive_reason[unit].append("deactivated by time")
                    rule.is_active[unit] = False