

import math

from datetime import timedelta
import select
from arrow import get
from matplotlib.pylab import f
from numpy import block, isin
from workload_simulator.block_generator.block import Block
from workload_simulator.request_generator.request import Request
from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import BudgetRelaxation, TimePrivacyUnit, create_single_attribute_schema



if __name__ == "__main__":

    schema = create_single_attribute_schema(domain_size=204800) # what cohere was using

    scenario = ScenarioConfig(
        name="20-1w-12w", # TODO: set back to 40
        allocation_interval=timedelta(weeks=1),
        active_time_window=timedelta(weeks=12), # ~3 months (quartal)
        user_expected_interarrival_time=timedelta(seconds=10), # 786k active users in 3 months
        request_expected_interarrival_time=timedelta(minutes=20), # resulting in 504 requests per week in expectation (batch)
        n_allocations=20,  # TODO: set back to 40
        pa_schema=schema,
        attributes=[f"a{i}" for i in range(10)],
        budget_relaxations=[BudgetRelaxation.NONE, BudgetRelaxation.BLACKBOX],
    )

    n_blocks_active_window = 12

    BASE_OFFSET = 100

    blocks = []
    for bid in range(0, 30):
        user_block = Block(
            id=100 * bid,
            budget_by_section=None,
            privacy_unit=TimePrivacyUnit.User.name,
            privacy_unit_selection=None,
            n_users=None,
            created=bid,
            retired=bid+n_blocks_active_window,
        )
        blocks.append(user_block)
        blocks += get_user_time_blocks(user_block=user_block, privacy_unit=TimePrivacyUnit.UserMonth, scenario=scenario, base_offset=BASE_OFFSET)

    print(f"Generated Blocks:")
    for block in blocks:
        print(block)

    for round in range(11, 30):
        active_user_blocks = [b for b in blocks if b.created <= round < b.retired and b.privacy_unit == TimePrivacyUnit.User.name]
        active_time_blocks = [b for b in blocks if b.created <= round < b.retired and b.privacy_unit == TimePrivacyUnit.UserMonth.name]

        assert len(active_user_blocks) == n_blocks_active_window, f"User: Round={round}  {len(active_user_blocks)=}"

        latest = get_latest(round_id=round, base=BASE_OFFSET, privacy_unit=TimePrivacyUnit.UserMonth, scenario=scenario)

        print(f"Round={round}  past=[0,{latest-1}]  {latest=} future=[{latest+1}, U64_MAX]")
        count = {blk.created for blk in active_time_blocks if blk.privacy_unit_selection == [latest, latest]}
        assert len(count) == n_blocks_active_window, f"Singleton UserTime: Round={round}  {count=}"