from dataclasses import dataclass
from datetime import timedelta
from typing import List

from workload_simulator.schema.schema import BudgetRelaxation, Schema, AttributeName


@dataclass
class ScenarioConfig:

    name: str # scenario identifier

    user_expected_interarrival_time: timedelta # expected time between each user arriving

    request_expected_interarrival_time: timedelta  # expected time between each request arriving

    allocation_interval: timedelta  # how often to run the allocation

    active_time_window: timedelta

    n_allocations: int # how many allocations should be simulated -> determines simulation until

    pa_schema: Schema

    attributes: List[AttributeName]

    budget_relaxations: List[BudgetRelaxation]

    base_time_offset: int = 100
