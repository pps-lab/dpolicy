import random

import argparse
import os
from copy import deepcopy
from datetime import timedelta
from multiprocessing import Pool
from sqlite3 import Time
from typing import Dict, List, Optional
import numpy as np

import tqdm
import workload_simulator.report
import workload_simulator.request_generator.mechanism as mlib
from workload_simulator.policy import budget_policy
from workload_simulator.policy.attribute_tagging import (
    AttributeTagging, AttributeTaggingGenerator, TimeAttributeTaggingGenerator)
from workload_simulator.policy.policy_mode import NewPolicyModeEncoder
from workload_simulator.request_generator import (calibration, metadata,
                                                  partitioning, request,
                                                  sampling, time_unit, utility)
from workload_simulator.request_generator.mode import CohereModeEncoder
from workload_simulator.request_generator.request import Category as Cat
from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import (AdpBudget, AttributeName,
                                              BudgetRelaxation,
                                              CategoryCorrelationLevel,
                                              CategoryData,
                                              Schema, TimePrivacyUnit,
                                              create_single_attribute_schema)
from workload_simulator.simulation import Simulation, WorkloadVariationConfig

DEFAULT_PRIVACY_UNIT = TimePrivacyUnit.User

def simple_mode_encoders(scenario: ScenarioConfig,
                         attributes: Dict[AttributeName, AttributeTagging],
                         categories: List[CategoryData],
                         has_time_budget: bool = False):

    DELTA = 1e-7

    total_budgets = [AdpBudget(3.0, DELTA), AdpBudget(5.0, DELTA), AdpBudget(7.0, DELTA), AdpBudget(10.0, DELTA), AdpBudget(15.0, DELTA), AdpBudget(20.0, DELTA)]

    category_level_extensions = {
        CategoryCorrelationLevel.MEMBER: lambda budget: budget,
        CategoryCorrelationLevel.MEMBER_STRONG: lambda budget: AdpBudget(budget.epsilon * 1.5, DELTA),
        CategoryCorrelationLevel.MEMBER_STRONG_WEAK: lambda budget: AdpBudget(budget.epsilon * 2, DELTA)
    }

    has_relaxation = BudgetRelaxation.BLACKBOX in scenario.budget_relaxations
    cohere_budgets = total_budgets
    if has_relaxation:
        # we have to assign Cohere the real budget, and compute the inverse of the relaxed budget
        # which we hardcode for now
        total_budgets = [AdpBudget(1.7, DELTA), AdpBudget(1.83, DELTA), AdpBudget(1.9, DELTA), AdpBudget(2, DELTA), AdpBudget(2.25, DELTA), AdpBudget(2.5, DELTA)]

    budget_relaxations = {
        BudgetRelaxation.NONE: lambda budget: budget, # no budget extension (identity function)
        BudgetRelaxation.BLACKBOX: lambda budget: budget_policy.translating_budget(budget, target_budget_relaxation=BudgetRelaxation.BLACKBOX),
    }
    budget_relaxations = {k: v for k, v in budget_relaxations.items() if k in scenario.budget_relaxations }

    privacy_unit = DEFAULT_PRIVACY_UNIT
    encoders = []

    for total_budget, cohere_total_budget in zip(total_budgets, cohere_budgets):
        cohere_bpolicy = budget_policy.TotalBudgetPolicy({privacy_unit: cohere_total_budget})
        name = f"eps{int(cohere_total_budget.epsilon)}-{privacy_unit.name}"
        enc = CohereModeEncoder(name=name, scenario=scenario, privacy_unit=privacy_unit, budget_policy=cohere_bpolicy, attributes=attributes, categories=categories)
        encoders.append(enc)

        if has_relaxation:
            # also output Cohere with the other budget
            cohere_bpolicy_small = budget_policy.TotalBudgetPolicy({privacy_unit: total_budget})
            name = f"eps{total_budget.epsilon:.2f}-{privacy_unit.name}"
            enc = CohereModeEncoder(name=name, scenario=scenario, privacy_unit=privacy_unit, budget_policy=cohere_bpolicy_small, attributes=attributes, categories=categories)
            encoders.append(enc)

        for aux in [None, TimePrivacyUnit.UserMonth]:
            name = f"eps{int(cohere_total_budget.epsilon)}"
            name += f"-{privacy_unit.name}_{aux.name}" if aux is not None else f"-{privacy_unit.name}"

            if has_time_budget and aux is not None:
                budget_config = budget_policy.Defaults.minimal_budget_time(total_budget=total_budget, DELTA=DELTA)
            elif has_relaxation:
                budget_config = budget_policy.Defaults.minimal_budget_relaxation(total_budget=total_budget, DELTA=DELTA)
            else:
                budget_config = budget_policy.Defaults.minimal_budget(total_budget=total_budget, DELTA=DELTA)

            enc = NewPolicyModeEncoder(
                scenario=scenario,
                name=name,
                main_privacy_unit=privacy_unit,
                aux_time_privacy_unit=aux,
                budget_relaxations=budget_relaxations,
                budget=budget_config,
                attributes=attributes,
                categories=categories,
                category_level_extensions=category_level_extensions,
            )

            encoders.append(enc)

    return encoders



def default_utility_assigners():

    ncd = utility.NormalizedCobbDouglasUtilityAssigner(
            privacy_cost_elasticity=2,
            data_elasticity=1,
            scaling_beta_a= 0.25, # with 0.25, the scaling factor is drawn from a "u-shape", so that most values are close to 0 or 1
            scaling_beta_b=0.25,
            use_balance_epsilon=True,
            utility_privacy_unit=DEFAULT_PRIVACY_UNIT
        )

    utility_assigners = [
        # utility.ConstantUtilityShadowAssigner(shadow_assigner=ncd),
        ncd,
    ]
    return utility_assigners


def default_scenario():
    """
    Uses the Cohere schema, and runs a scenario with weekly allocations with an active user window of 12 weeks.
    In expectation, we have 786k active users in 3 months, and 504 requests per allocation in expectation (batch).
    We simulate 50 allocations.
    """

    schema = create_single_attribute_schema(domain_size=204800) # what cohere was using

    n_attributes = 150

    scenario = ScenarioConfig(
        name="20-1w-12w-morecat",
        allocation_interval=timedelta(weeks=1),
        active_time_window=timedelta(weeks=12), # ~3 months (quartal)
        user_expected_interarrival_time=timedelta(seconds=10), # 786k active users in 3 months
        request_expected_interarrival_time=timedelta(minutes=20), # resulting in 504 requests per week in expectation (batch)
        n_allocations=20,
        pa_schema=schema,
        attributes=[f"a{i}" for i in range(n_attributes)],
        budget_relaxations=[BudgetRelaxation.NONE],
    )

    return scenario

def relax_scenario():
    """
    Uses the Cohere schema, and runs a scenario with weekly allocations with an active user window of 12 weeks.
    In expectation, we have 786k active users in 3 months, and 504 requests per allocation in expectation (batch).
    We simulate 50 allocations.
    """

    schema = create_single_attribute_schema(domain_size=204800) # what cohere was using

    n_attributes = 1

    scenario = ScenarioConfig(
        name="20-1w-12w-relaxation",
        allocation_interval=timedelta(weeks=1),
        active_time_window=timedelta(weeks=12), # ~3 months (quartal)
        user_expected_interarrival_time=timedelta(seconds=10), # 786k active users in 3 months
        request_expected_interarrival_time=timedelta(minutes=20), # resulting in 504 requests per week in expectation (batch)
        n_allocations=20,
        pa_schema=schema,
        attributes=[f"a{i}" for i in range(n_attributes)],
        budget_relaxations=[BudgetRelaxation.NONE, BudgetRelaxation.BLACKBOX],
    )

    return scenario

def time_scenario():
    """
    Uses the Cohere schema, and runs a scenario with weekly allocations with an active user window of 12 weeks.
    In expectation, we have 786k active users in 3 months, and 504 requests per allocation in expectation (batch).
    We simulate 50 allocations.
    """

    schema = create_single_attribute_schema(domain_size=204800) # what cohere was using

    n_attributes = 2 # one static one dynamic attribute

    scenario = ScenarioConfig(
        name="20-1w-12w-time",
        allocation_interval=timedelta(weeks=1),
        active_time_window=timedelta(weeks=12), # ~3 months (quartal)
        user_expected_interarrival_time=timedelta(seconds=10), # 786k active users in 3 months
        request_expected_interarrival_time=timedelta(minutes=20), # resulting in 504 requests per week in expectation (batch)
        n_allocations=20,
        pa_schema=schema,
        attributes=[f"a{i}" for i in range(n_attributes)],
        budget_relaxations=[BudgetRelaxation.NONE],
    )

    return scenario

def minimal_scenario():
    schema = create_single_attribute_schema(domain_size=32)
    scenario = time_scenario()
    scenario.name = "13-1w-12w-minimal"
    scenario.n_allocations = 13
    scenario.pa_schema=schema
    return scenario


def all_scenario():
    """
    Uses the Cohere schema, and runs a scenario with weekly allocations with an active user window of 12 weeks.
    In expectation, we have 786k active users in 3 months, and 504 requests per allocation in expectation (batch).
    We simulate 50 allocations.
    """

    schema = create_single_attribute_schema(domain_size=204800) # what cohere was using

    n_attributes = 150

    scenario = ScenarioConfig(
        name="20-1w-12w-all", # TODO: set back to 40
        allocation_interval=timedelta(weeks=1),
        active_time_window=timedelta(weeks=12), # ~3 months (quartal)
        user_expected_interarrival_time=timedelta(seconds=10), # 786k active users in 3 months
        request_expected_interarrival_time=timedelta(minutes=20), # resulting in 504 requests per week in expectation (batch)
        n_allocations=20,  # TODO: set back to 40
        pa_schema=schema,
        attributes=[f"a{i}" for i in range(n_attributes)],
        budget_relaxations=[BudgetRelaxation.NONE, BudgetRelaxation.BLACKBOX],
    )

    return scenario

def mixed_workload(scenario: ScenarioConfig, attributes: Dict[AttributeName, AttributeTagging], categories: Optional[List[CategoryData]], name="mixed:GM-LM-RR-LSVT-SGD-PATE-sub25100-defpa", attribute_assignment_function=None, subsampling=None, inequal_time=False, blackbox_relaxation = False):

    """Corresponds to (W4:All) in the paper.
    """
    schema: Schema = scenario.pa_schema

    highly_partition = partitioning.Defaults.new_highly_partitioned(schema)
    no_partition = partitioning.Defaults.new_low_partitioned(schema)
    norm_partition = partitioning.Defaults.new_normal_partitioned(schema)

    if attribute_assignment_function is None:
        attribute_assignment_function = metadata.Defaults.new_powerlaw_category_based_attributes(attributes,
                                                                          categories)

    no_relax = metadata.Defaults.new_no_budget_relaxation(scenario)
    if blackbox_relaxation:
        blackbox_relax = metadata.Defaults.new_80blackbox_budget_relaxation(scenario)
    else:
        blackbox_relax = no_relax

    sub = sampling.Defaults.equal_p25_p100() if subsampling is None else subsampling

    cost = calibration.Defaults.mice_hare_elephant_v2()

    cheap_cost = calibration.Defaults.mice_hare_elephant_cheap()

    privacy_units = [TimePrivacyUnit.UserMonth]

    if not inequal_time:
        time_selection = time_unit.Defaults.only_latest(privacy_units=privacy_units, scenario=scenario)
    else:
        time_selection = time_unit.Defaults.even_odd(privacy_units=privacy_units, scenario=scenario)

    unit = DEFAULT_PRIVACY_UNIT

    mechanism_distribution = [
        Cat(highly_partition, attribute_assignment_function, no_relax, time_selection, mlib.GaussianMechanism(unit, cost, sub), scenario.budget_relaxations, 1),
        Cat(highly_partition, attribute_assignment_function, no_relax, time_selection, mlib.LaplaceMechanism(unit, cheap_cost, sub), scenario.budget_relaxations, 1),
        Cat(no_partition, attribute_assignment_function, no_relax, time_selection, mlib.SVTLaplaceMechanism(unit, cheap_cost, sub), scenario.budget_relaxations, 1),
        Cat(no_partition, attribute_assignment_function, no_relax, time_selection, mlib.RandResponseMechanism(unit, cheap_cost, sub), scenario.budget_relaxations, 1),
        Cat(norm_partition, attribute_assignment_function, blackbox_relax, time_selection, mlib.MLNoisySGDMechanism(unit, cost, sub), scenario.budget_relaxations, 1), # TODO [nku] BRING BACK!!!
        Cat(norm_partition, attribute_assignment_function, blackbox_relax, time_selection, mlib.MLPateGaussianMechanism(unit, cost, sub), scenario.budget_relaxations, 1),
    ]

    workload_cfg = request.Workload(
        name,
        schema,
        mechanism_distribution
    )
    return workload_cfg


def workload_simulation(output_dir, n_repetitions):
    scenario = default_scenario()

    atg = AttributeTaggingGenerator()
    attributes, categories = atg.generate(attributes=scenario.attributes, n_categories=10, n_categories_per_attribute=2.5)

    workload_variations = WorkloadVariationConfig.product(
        default_utility_assigners(), simple_mode_encoders(scenario, attributes, categories)
    )

    # Find categories for dpolicy, ie where workload_variations[].mode_encoder == PolicyModeEncoder
    workloads = [
        mixed_workload(scenario, attributes, categories), # (W4:All) in paper
    ]

    simulation = Simulation(
            scenario=scenario,
            workloads=workloads,
            workload_variations=workload_variations,
            output_dir=output_dir,
        )

    run_in_parallel(simulation, n_repetitions=n_repetitions)


def workload_simulation_relax(output_dir, n_repetitions):
    scenario = relax_scenario()

    atg = AttributeTaggingGenerator()
    attributes, categories = atg.generate(attributes=scenario.attributes, n_categories=0, n_categories_per_attribute=0)

    workload_variations = WorkloadVariationConfig.product(
        default_utility_assigners(), simple_mode_encoders(scenario, attributes, categories)
    )

    # Find categories for dpolicy, ie where workload_variations[].mode_encoder == PolicyModeEncoder
    workloads = [
        mixed_workload(scenario, attributes, categories, inequal_time=False, blackbox_relaxation=True),
    ]

    simulation = Simulation(
        scenario=scenario,
        workloads=workloads,
        workload_variations=workload_variations,
        output_dir=output_dir,
    )

    run_in_parallel(simulation, n_repetitions=n_repetitions)


def workload_simulation_time(output_dir, n_repetitions):
    scenario = time_scenario()

    atg = TimeAttributeTaggingGenerator()
    attributes, categories = atg.generate(attributes=scenario.attributes, n_categories=0, n_categories_per_attribute=0)

    workload_variations = WorkloadVariationConfig.product(
        default_utility_assigners(), simple_mode_encoders(scenario, attributes, categories, has_time_budget=True)
    )

    # Find categories for dpolicy, ie where workload_variations[].mode_encoder == PolicyModeEncoder
    workloads = [
        mixed_workload(scenario, attributes, categories,
                        attribute_assignment_function=metadata.Defaults.new_static_dynamic_attribute_assignment(attributes),
                       inequal_time=True), # (W4:All) in paper
    ]

    simulation = Simulation(
        scenario=scenario,
        workloads=workloads,
        workload_variations=workload_variations,
        output_dir=output_dir,
    )

    run_in_parallel(simulation, n_repetitions=n_repetitions)


def workload_simulation_minimal(output_dir, n_repetitions):
    scenario = minimal_scenario()

    atg = TimeAttributeTaggingGenerator()
    attributes, categories = atg.generate(attributes=scenario.attributes, n_categories=0, n_categories_per_attribute=0)

    workload_variations = WorkloadVariationConfig.product(
        default_utility_assigners(), simple_mode_encoders(scenario, attributes, categories, has_time_budget=True)
    )

    # Find categories for dpolicy, ie where workload_variations[].mode_encoder == PolicyModeEncoder
    workloads = [
        mixed_workload(scenario, attributes, categories,
                        attribute_assignment_function=metadata.Defaults.new_static_dynamic_attribute_assignment(attributes),
                       inequal_time=True), # (W4:All) in paper
    ]

    simulation = Simulation(
        scenario=scenario,
        workloads=workloads,
        workload_variations=workload_variations,
        output_dir=output_dir,
    )

    run_in_parallel(simulation, n_repetitions=n_repetitions)


def simulation_runner(info):
    i, simulation = info
    # TODO:
    #simulation.run_cached(repetition_start=i, n_repetitions=1)
    simulation.run(repetition_start=i, n_repetitions=1)


def run_in_parallel(simulation, n_repetitions):

    if n_repetitions == 1:
        simulation_runner((0, simulation))
    else:
        lst = []
        for i in range(n_repetitions):
            copy = deepcopy(simulation)
            lst.append((i, copy))


        with Pool(processes=n_repetitions) as p:
            for _ in tqdm.tqdm(p.imap_unordered(simulation_runner, lst), total=len(lst)):
                pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Workload')

    parser.add_argument('-o', '--output-dir', type=str, default='output/applications', help='Output directory path')

    parser.add_argument('-n', '--n-repetition', type=int, default=1, help='Number of repetitions for each workload')

    parser.add_argument('-r', '--report-only', action='store_true', help='Only generate report')

    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    output_dir = args.output_dir

    # create directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # set np seed
    random.seed(args.seed)
    np.random.seed(args.seed)


    if not args.report_only:
        workload_simulation_minimal(output_dir, n_repetitions=args.n_repetition)
        workload_simulation(output_dir, n_repetitions=args.n_repetition)
        workload_simulation_relax(output_dir, n_repetitions=args.n_repetition)
        workload_simulation_time(output_dir, n_repetitions=args.n_repetition)

    # create request workloads to look at advantage of subsampling
    #effect_subsampling_simulation(output_dir, n_repetitions=args.n_repetition)

    # create a workload report for each scenario
    for scn in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, scn)):
            workload_simulator.report.create_workload_report(os.path.join(output_dir, scn), slacks=[0.0], skip_pa_overlap=True)
