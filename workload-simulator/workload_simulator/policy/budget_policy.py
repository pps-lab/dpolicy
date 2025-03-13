from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from workload_simulator.schema.schema import BudgetConfig, AdpBudget, BaseBudget, BudgetRelaxation, PrivacyRiskLevel, TimePrivacyUnit, AttributeName
from workload_simulator.policy.attribute_tagging import AttributeCategory, CategoryCorrelationLevel, AttributeTagging


class Defaults:


    @staticmethod
    def minimal_budget(total_budget: AdpBudget, DELTA: float):

        config = BudgetConfig(
            per_attribute_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(3.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(9.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: total_budget
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(3.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(9.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: total_budget
                }
            },
            per_attribute_budget_overrides=[],
            per_category_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(5.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(10.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: AdpBudget(min(12.0, total_budget.epsilon), DELTA),
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(5.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(10.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: AdpBudget(min(12.0, total_budget.epsilon), DELTA),
                }
            },
            per_category_budget_overrides=[],
            total_budget={
                TimePrivacyUnit.User: total_budget,
                TimePrivacyUnit.UserMonth: total_budget, #AdpBudget(min(3.0, total_budget.epsilon), DELTA)
            }
        )

        return config

    @staticmethod
    def minimal_budget_unlimcat(total_budget: AdpBudget, DELTA: float):

        config = BudgetConfig(
            per_attribute_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(3.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(9.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: total_budget
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(3.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(9.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: total_budget
                }
            },
            per_attribute_budget_overrides=[],
            per_category_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(50.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(100.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: AdpBudget(min(120.0, total_budget.epsilon), DELTA),
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(50.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(100.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: AdpBudget(min(120.0, total_budget.epsilon), DELTA),
                }
            },
            per_category_budget_overrides=[],
            total_budget={
                TimePrivacyUnit.User: total_budget,
                TimePrivacyUnit.UserMonth: total_budget
            }
        )

        return config

    @staticmethod
    def minimal_budget_relaxation(total_budget: AdpBudget, DELTA: float):

        """ Same relative budgets as minimal budget, but smaller """
        config = BudgetConfig(
            per_attribute_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(1.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(1.5, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: total_budget
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(1.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(1.5, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: total_budget
                }
            },
            per_attribute_budget_overrides=[],
            per_category_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(1.2, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(1.6, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: AdpBudget(min(1.7, total_budget.epsilon), DELTA),
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(1.2, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(1.6, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: AdpBudget(min(1.7, total_budget.epsilon), DELTA),
                }
            },
            per_category_budget_overrides=[],
            total_budget={
                TimePrivacyUnit.User: total_budget,
                TimePrivacyUnit.UserMonth: AdpBudget(min(1.0, total_budget.epsilon), DELTA)
            }
        )

        return config

    @staticmethod
    def minimal_budget_time(total_budget: AdpBudget, DELTA: float):

        config = BudgetConfig(
            per_attribute_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.MEDIUM: total_budget,
                    PrivacyRiskLevel.LOW: total_budget
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(3.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: total_budget
                }
            },
            per_attribute_budget_overrides=[],
            per_category_budget={},
            per_category_budget_overrides=[],
            total_budget={
                TimePrivacyUnit.User: total_budget,
                TimePrivacyUnit.UserMonth: total_budget,
            }
        )

        return config

    @staticmethod
    def minimal_budget_all(total_budget: AdpBudget, DELTA: float):

        config = BudgetConfig(
            per_attribute_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.HIGH: total_budget,
                    PrivacyRiskLevel.MEDIUM: total_budget,
                    PrivacyRiskLevel.LOW: total_budget
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(10.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(5.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: total_budget
                }
            },
            per_attribute_budget_overrides=[],
            per_category_budget={
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(5.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(10.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: AdpBudget(min(12.0, total_budget.epsilon), DELTA),
                },
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.HIGH: AdpBudget(min(5.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(min(10.0, total_budget.epsilon), DELTA),
                    PrivacyRiskLevel.LOW: AdpBudget(min(12.0, total_budget.epsilon), DELTA),
                }
            },
            per_category_budget_overrides=[],
            total_budget={
                TimePrivacyUnit.User: total_budget,
                TimePrivacyUnit.UserMonth: total_budget,
            }
        )

        return config

    @staticmethod
    def new_default():
        DELTA = 1e-5

        # TODO [nku] revisit these values
        config = BudgetConfig(
            per_attribute_budget={
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.LOW: AdpBudget(7.0, DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(3.0, DELTA),
                    PrivacyRiskLevel.HIGH: AdpBudget(1.0, DELTA)
                },
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.LOW: AdpBudget(7.0, DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(5.0, DELTA),
                    PrivacyRiskLevel.HIGH: AdpBudget(1.0, DELTA)
                }
            },
            per_attribute_budget_overrides=[],
            per_category_budget={
                TimePrivacyUnit.UserMonth: {
                    PrivacyRiskLevel.LOW: AdpBudget(10.0, DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(7.0, DELTA),
                    PrivacyRiskLevel.HIGH: AdpBudget(3.0, DELTA)
                },
                TimePrivacyUnit.User: {
                    PrivacyRiskLevel.LOW: AdpBudget(30.0, DELTA),
                    PrivacyRiskLevel.MEDIUM: AdpBudget(20.0, DELTA),
                    PrivacyRiskLevel.HIGH: AdpBudget(7.0, DELTA)
                }
            },
            per_category_budget_overrides=[],
            total_budget={
                TimePrivacyUnit.UserMonth: AdpBudget(100.0, DELTA),
                TimePrivacyUnit.User: AdpBudget(300.0, DELTA)
            }
        )

        return config



@dataclass
class TotalBudgetPolicy:

    budget: Dict[TimePrivacyUnit, BaseBudget]

    def get_per_attribute_budget(self, attribute: AttributeName, relaxation: BudgetRelaxation, privacy_unit: TimePrivacyUnit, attributes: Dict[AttributeName, AttributeTagging])  -> BaseBudget:
        raise ValueError("TotalBudgetPolicy does not support per attribute budgets")


    def get_per_category_budget(self, category: AttributeCategory, category_correlation_level: CategoryCorrelationLevel, category_risk_level: PrivacyRiskLevel, relaxation: BudgetRelaxation, privacy_unit: TimePrivacyUnit, attributes: Dict[AttributeName, AttributeTagging])  -> BaseBudget:
        raise ValueError("TotalBudgetPolicy does not support per category budgets")



    def get_total_budget(self, relaxation: BudgetRelaxation, privacy_unit: TimePrivacyUnit)  -> BaseBudget:
        if relaxation != BudgetRelaxation.NONE:
            raise ValueError("TotalBudgetPolicy does not support budget relaxations")
        return self.budget[privacy_unit]

    def export(self) -> dict:
        return {
            "budget": {unit.name: budget.asdict() for unit, budget in self.budget.items()}
        }



def translating_budget(budget: AdpBudget, target_budget_relaxation: BudgetRelaxation):
    # TODO: These could also be encoded in a configuration file
    if target_budget_relaxation == BudgetRelaxation.NONE:
        return budget

    elif target_budget_relaxation == BudgetRelaxation.BLACKBOX:


        assert isinstance(budget, AdpBudget), f"budget must be an AdpBudget, but is {type(budget)}"

        # (empirical epsilon, theoretical epsilon)
        measured = [(0.1, 0.1), (0.5, 0.5), (0.75, 0.75), (1, 1), (1.6, 2), (1.8, 4), (2, 10)]

        # TODO [nku] revisit these values -> the conversion function with the extrapolated values has a strange shape

        # extrapolated
        # [hly]: I added  (15, 60), (20, 70) so it extrapolates sensible ish
        # [hly]: I added more
        extrapolated = [(1.7, 3), (1.83, 5), (1.9, 7), # [hly] I added even more
                        (2.25, 15), (2.5, 20), (2.75, 25), (3, 30)]

        combined = measured + extrapolated

        x = [i[0] for i in combined]
        y = [i[1] for i in combined]

        xs = np.arange((min(x)), (max(x)), 0.01)

        if budget.epsilon <= max(x) and budget.epsilon >= min(x):
            xs = x + [budget.epsilon]
            yinterp = np.interp(xs, x, y)
            return AdpBudget(epsilon=yinterp[-1], delta=budget.delta)
        elif budget.epsilon > max(x):
            epsilon = 2 * budget.epsilon # TODO [nku] another very arbitrary choice
            return AdpBudget(epsilon=epsilon, delta=budget.delta)


    else:
        raise ValueError(f"Unknown target budget relaxation: {target_budget_relaxation}")
