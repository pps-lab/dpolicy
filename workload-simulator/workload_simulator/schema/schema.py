

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List
import enum
import math


@dataclass(frozen=True)
class BaseBudget(ABC):

    @abstractmethod
    def asdict(self):
        pass

    @abstractmethod
    def asstr(self):
        pass


@dataclass(frozen=True)
class AdpBudget(BaseBudget):
    epsilon: float
    delta: float

    def asdict(self):
        return {"EpsDeltaDp": {"eps": self.epsilon, "delta": self.delta}}

    def asstr(self):
        return f"epsilon={self.epsilon}delta={self.delta}"


    def __le__(self, other):
        return self.epsilon <= other.epsilon and self.delta <= other.delta

    def __lt__(self, other):
        return (self.epsilon < other.epsilon and self.delta <= other.delta) or (self.epsilon <= other.epsilon and self.delta < other.delta)


@dataclass
class RdpAccountingType:

    eps_values: List[float] = None

    def __post_init__(self):
        self.eps_values = len(self.alphas()) * [0.0]


    def alphas(self):
        return [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6, 1e10] #, 1e16 1e16  TODO [nku] maybe bring back?

    def empty_cost(self):
        return {"Rdp": {"eps_values": [0.0] * len(self.eps_values)}}


    def cost_config(self):
        return {
            "alphas": self.alphas()
        }


@dataclass
class AccountingType:

    Rdp: RdpAccountingType = RdpAccountingType()

    def empty_cost(self):
        return self.Rdp.empty_cost()

    def cost_config(self):
        return self.Rdp.cost_config()



@dataclass
class DomainRange:
    min: int
    max: int

@dataclass
class Attribute:
    name: str
    value_domain: DomainRange


class PrivacyRiskLevel(str, enum.Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

AttributeName = str
AttributeCategory = str

@dataclass
class CategoryData:
    name: AttributeCategory
    privacy_risk_level: PrivacyRiskLevel
    weight: float


class BudgetRelaxation(str, enum.Enum):
    NONE = 1
    BLACKBOX = 2
#    API_ACCESS = 3



class TimePrivacyUnit(enum.Enum):
    UserDay = 1,
    UserWeek = 7,
    UserMonth = 31,
    UserYear = 366,
    User100Year = 36525, # 100 * 365 + 100/4 (in 100 year there are max 25 leap years)
    User = 1000000000000000,


class CategoryCorrelationLevel(str, enum.Enum):
    MEMBER = 1
    MEMBER_STRONG = 2
    MEMBER_STRONG_WEAK = 3


@dataclass
class PerAttributeBudgetOverride:
    privacy_unit: TimePrivacyUnit
    attribute: AttributeName
    relaxation: BudgetRelaxation
    budget: BaseBudget


@dataclass
class PerCategoryBudgetOverride:
    privacy_unit: TimePrivacyUnit
    category: AttributeCategory
    category_level: CategoryCorrelationLevel
    relaxation: BudgetRelaxation
    budget: BaseBudget

@dataclass
class BudgetConfig:

    per_attribute_budget: Dict[TimePrivacyUnit, Dict[PrivacyRiskLevel, BaseBudget]]
    per_attribute_budget_overrides: List[PerAttributeBudgetOverride]

    per_category_budget: Dict[TimePrivacyUnit, Dict[PrivacyRiskLevel, BaseBudget]]
    per_category_budget_overrides: List[PerCategoryBudgetOverride]

    total_budget: Dict[TimePrivacyUnit, BaseBudget]

    def export(self):
        return {
            "per_attribute_budget": {unit.name: {risk_level.name: budget.asdict() for risk_level, budget in budget_dict.items()} for unit, budget_dict in self.per_attribute_budget.items()},
            "per_attribute_budget_overrides": self.per_attribute_budget_overrides,
            "per_category_budget": {unit.name: {risk_level.name: budget.asdict() for risk_level, budget in budget_dict.items()} for unit, budget_dict in self.per_category_budget.items()},
            "per_category_budget_overrides": self.per_category_budget_overrides,
            "total_budget": {unit.name: budget.asdict() for unit, budget in self.total_budget.items()}
        }



@dataclass
class Schema:
    attributes: List[Attribute]
    accounting_type: AccountingType

    # are assigned later by the mode encoder
    privacy_units: List[str] = None
    block_sliding_window_size: int = None

    attribute_info: dict = None
    budget_config: BudgetConfig = None

    attributes_pa: List[Attribute] = None # only the attributes that are used for partitioning attributes

    def __post_init__(self):
        if self.attributes_pa is None:
            self.attributes_pa = deepcopy(self.attributes)

    def empty_cost(self):
        return self.accounting_type.empty_cost()

    def cost_config(self):
        return self.accounting_type.cost_config()

    def virtual_block_domain_size(self):
        size = 1
        assert len(self.attributes) > 0
        for dim in self.attributes:
            dim: Attribute = dim
            size *= (dim.value_domain.max - dim.value_domain.min + 1)
        return size


    def get_attribute(self, name: str):
        search = [attr for attr in self.attributes if attr.name == name]
        assert len(search) == 1
        return search[0]

def create_single_attribute_schema(domain_size, name="attr0"):
    return Schema(attributes=[Attribute(name=name, value_domain=DomainRange(min=0, max=domain_size-1))], accounting_type=AccountingType())






def compute_group_size(from_unit: TimePrivacyUnit, to_unit: TimePrivacyUnit) -> float:
    assert from_unit.value < to_unit.value, f"from_unit must be smaller than to_unit:  {from_unit} < {to_unit}"
    return math.ceil(to_unit.value[0] / from_unit.value[0])



def rdp_group_privacy(mechanism, from_unit: TimePrivacyUnit, to_unit: TimePrivacyUnit, target_alpha: float):

    group_size = compute_group_size(from_unit, to_unit)

    c = math.log2(group_size)
    # 2^c-stable transformation

    alpha_prime = target_alpha * group_size   # alpha' = target_alpha * 2^c
    eps_prime = mechanism.get_RDP(alpha_prime)

    # epsilon(target_alpha)
    eps = eps_prime * math.pow(3, c)
    return eps


#if __name__ == "__main__":
#
#    print(TimePrivacyUnit.UserDay.value)
#
#    print(f"{compute_group_size(TimePrivacyUnit.UserDay, TimePrivacyUnit.UserWeek)=}")
#    print(f"{compute_group_size(TimePrivacyUnit.UserDay, TimePrivacyUnit.UserMonth)=}")
#    print(f"{compute_group_size(TimePrivacyUnit.UserDay, TimePrivacyUnit.UserYear)=}")
#    print(f"{compute_group_size(TimePrivacyUnit.UserDay, TimePrivacyUnit.User100Year)=}")
#
#    print(f"{compute_group_size(TimePrivacyUnit.UserWeek, TimePrivacyUnit.UserMonth)=}")
#    print(f"{compute_group_size(TimePrivacyUnit.UserWeek, TimePrivacyUnit.UserYear)=}")
#    print(f"{compute_group_size(TimePrivacyUnit.UserWeek, TimePrivacyUnit.User100Year)=}")
#
#    print(f"{compute_group_size(TimePrivacyUnit.UserMonth, TimePrivacyUnit.UserYear)=}")
#    print(f"{compute_group_size(TimePrivacyUnit.UserMonth, TimePrivacyUnit.User100Year)=}")
#
#    print(f"{compute_group_size(TimePrivacyUnit.UserYear, TimePrivacyUnit.User100Year)=}")