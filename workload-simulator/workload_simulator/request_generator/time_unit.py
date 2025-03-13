



from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from datetime import timedelta
import math
import random

import numpy as np

from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import TimePrivacyUnit

class Defaults:

    @staticmethod
    def only_latest(privacy_units: List[TimePrivacyUnit], scenario: ScenarioConfig):
        distribution = [{"prob": 1.0, "value": ["latest"]}]
        return CategoricalTimeSelection(privacy_units=privacy_units, scenario=scenario, distribution=distribution)

    @staticmethod
    def even_odd(privacy_units: List[TimePrivacyUnit], scenario: ScenarioConfig):
        return EvenOddTimeSelection(privacy_units=privacy_units, scenario=scenario)


class BaseTimeSelection(ABC):

    @abstractmethod
    def generate(self, round_id: int) -> dict:
        pass

    @abstractmethod
    def info(self):
        pass




@dataclass
class CategoricalTimeSelection(BaseTimeSelection):

    privacy_units: List[TimePrivacyUnit]
    scenario: ScenarioConfig

    distribution: list[dict]  # [{"prob": 0.5, "value": ["latest"]}, {"prob": 0.5, "value": ["past", "latest", "future"]}]


    def __post_init__(self):
        self.weights = [b["prob"] for b in self.distribution]
        assert sum(self.weights) == 1.0 or math.isclose(sum(self.weights), 1.0), "probabilities do not sum to 1.0"

        self.population = list(range(len(self.weights)))

        for x in self.distribution:
            for v in x["value"]:
                assert v in ["past", "latest", "future"], f"Invalid"

    def generate(self, round_id: int) -> dict:
        privacy_unit_selection = dict()

        idx = random.choices(population=self.population, weights=self.weights, k=1)[0]
        entry = self.distribution[idx]

        for privacy_unit in self.privacy_units:
            latest = get_latest(round_id=round_id, privacy_unit=privacy_unit, scenario=self.scenario)
            selection = []
            if "past" in entry["value"]:
                select_past = [0, latest-1]
                selection.append(select_past)

            if "latest" in entry["value"]:
                select_latest = [latest, latest]
                selection.append(select_latest)

            if "future" in entry["value"]:
                select_future = [latest+1, latest + 25]
                selection.append(select_future)
            privacy_unit_selection[privacy_unit.name] =  selection

        return privacy_unit_selection

    def info(self) -> dict:
        return {
            "name": __class__.__name__,
            "distribution": self.distribution
        }


@dataclass
class EvenOddTimeSelection(BaseTimeSelection):

    """Selects time-based unit but skews to even or odd rounds"""

    privacy_units: List[TimePrivacyUnit]
    scenario: ScenarioConfig

    # def __post_init__(self):

    def generate(self, round_id: int) -> dict:
        privacy_unit_selection = dict()


        offsets = [-4, -3, -2, -1, 0, 1, 2]
        weights = [1] * len(offsets)
        weights[4] = 3 # "a third"

        for privacy_unit in self.privacy_units:
            latest = get_latest(round_id=round_id, privacy_unit=privacy_unit, scenario=self.scenario)

            offset = random.choices(population=offsets, weights=weights, k=1)[0]
            latest += offset
            # if latest % 2 == 1:
            #     if np.random.choice([True, False], p=[0.2, 0.8]):
            #         latest = latest - 1

            privacy_unit_selection[privacy_unit.name] = [[latest, latest]]

        return privacy_unit_selection

    def info(self) -> dict:
        return {
            "name": __class__.__name__,
        }


def privacy_unit_to_timedelta(privacy_unit: TimePrivacyUnit):
    assert privacy_unit != TimePrivacyUnit.User
    n_days = privacy_unit.value[0]
    return timedelta(days=n_days)


def get_latest(round_id: int, privacy_unit: TimePrivacyUnit, scenario: ScenarioConfig):

    privacy_unit_time = privacy_unit_to_timedelta(privacy_unit)

    n_units_per_active_window = math.ceil(scenario.active_time_window / privacy_unit_time)

    n_allocations_per_active_window = math.ceil(scenario.active_time_window / scenario.allocation_interval)

    n_blocks_with_same = n_allocations_per_active_window // n_units_per_active_window

    #print(f"{n_units_per_active_window=}     {n_allocations_per_active_window=}    {n_blocks_with_same=}")
    latest = scenario.base_time_offset + math.ceil((round_id + 2) / n_blocks_with_same)
    return latest
