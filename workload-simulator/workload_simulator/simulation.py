import datetime
import time
from enum import Enum
import itertools
import os
import random
import json
import copy

from dataclasses import dataclass, asdict
from typing import List, Optional

from datetime import timedelta

import orjson
from tqdm import tqdm

import sys
import simpy
from workload_simulator.block_generator import block
from workload_simulator.request_generator import metadata, request, utility, mode
from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import Schema

import numpy

def orjson_default(obj):
    raise TypeError

def orjson_options():
    return orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY

@dataclass
class WorkloadVariationConfig:
    name: str
    utility_assigner: utility.BaseUtilityAssigner
    mode_encoder: mode.BaseModeEncoder


    def __post_init__(self):
        assert isinstance(self.utility_assigner, utility.BaseUtilityAssigner)
        assert isinstance(self.mode_encoder, mode.BaseModeEncoder)

    @staticmethod
    def product(utility_assigners: List[utility.BaseUtilityAssigner], mode_encoders: List[mode.BaseModeEncoder]):
        workload_variations = []
        for uti, enc in itertools.product(utility_assigners, mode_encoders):
            name = f"{uti.short()}_{enc.short()}"
            workload_variations.append(WorkloadVariationConfig(name=name, utility_assigner=uti, mode_encoder=enc))

        return workload_variations

class Simulation:
    def __init__(
        self,
        workloads: List[request.Workload],
        workload_variations: List[WorkloadVariationConfig],
        scenario: ScenarioConfig,
        output_dir: str,
    ):

        self.scenario = scenario
        self.workloads = workloads
        self.workload = None # will then store the active workload
        self.workload_variations = workload_variations

        # from t=0, we can add users
        # from t=start, we can add requests and at start+ allocation_interval, we run the first allocation
        self.start = scenario.active_time_window.total_seconds() #- scenario.allocation_interval.total_seconds()

        assert (
            scenario.active_time_window.total_seconds() % scenario.allocation_interval.total_seconds() == 0
        ), "active_time_window must be a multiple of allocation_interval"


        self.user_arrival_rate = 1.0 / scenario.user_expected_interarrival_time.total_seconds()
        self.pre_start_user_arrival_rate = self.user_arrival_rate

        self.request_arrival_rate = 1.0 / scenario.request_expected_interarrival_time.total_seconds()


        self.allocation_interval = scenario.allocation_interval.total_seconds()


        self.simulation_until = self.start + scenario.n_allocations * self.allocation_interval + 1

        self.output_dir = output_dir
        self.init_state()



    def init_state(self):

        """
        Initialize and reset the state of the simulation
        """

        # assign a unique id to each user and request
        self.user_id_counter = 1
        self.user_id_counter_history = (
            dict()
        )  # time -> user_id_counter to keep track of the history

        self.requests_new: List[request.Request] = []
        self.requests: List[request.Request] = []
        self.users = []

        self.block_updates: List[block.UserUpdateInfo] = []

        self.round_id = -1


    def dump_scenario_state(self):

        wnames = [w.name for w in self.workloads]
        assert len(wnames) == len(set(wnames)), "workload names must be unique"

        vnames = [v.name for v in self.workload_variations]
        assert len(vnames) == len(set(vnames)), f"variation names must be unique {vnames=}"

        scenario_dir = os.path.join(self.output_dir, self.scenario.name)

        os.makedirs(scenario_dir, exist_ok=True)

        # store scenario config
        with open(os.path.join(scenario_dir, "scenario.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.scenario), f, default=str)
        # with open(os.path.join(scenario_dir, "scenario.json"), "wb") as f:
        #     f.write(orjson.dumps(asdict(self.scenario), default=orjson_default, option=orjson_options()))

    def dump_workload_state(self, repetition: int):

        workload_dir = os.path.join(self.output_dir, self.scenario.name, self.workload.name)

        os.makedirs(os.path.join(workload_dir, ".raw"), exist_ok=True) # note: by creating .raw directory, we also create the workload_dir itself

        if repetition == 0:
            # in first rep, create the folder and store the workload config

            #def factory(kv_lst):
            #    data = {}
            #    for k, v in kv_lst:
            #        if isinstance(v, Enum):
            #            data[k] = v.name
            #        else:
            #            data[k] = v
            #    return data
#

            with open(os.path.join(workload_dir, "workload.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(self.workload), f, default=str)

        # dump (incomplete) requests and blocks (without any variation) to disk
        raw_dir = os.path.join(workload_dir, ".raw")
        # with open(os.path.join(raw_dir, f"requests_{repetition}.json"), "w", encoding="utf-8") as f:
        with open(os.path.join(raw_dir, f"requests_{repetition}.json"), "wb") as f:
            # json.dump([asdict(r) for r in self.requests], f)
            f.write(orjson.dumps(self.requests, default=orjson_default, option=orjson_options()))

        # with open(os.path.join(raw_dir, f"updates_{repetition}.json"), "w", encoding="utf-8") as f:
        with open(os.path.join(raw_dir, f"updates_{repetition}.json"), "wb") as f:
            # json.dump([asdict(b) for b in self.block_updates], f)
            f.write(orjson.dumps(self.block_updates, default=orjson_default, option=orjson_options()))

    def dump_workload_variation(self, schema: Schema, requests: List[request.Request], blocks: List[block.Block], variation: WorkloadVariationConfig, repetition: int, rules_info: Optional[dict] = None):
         # write workload file to disk under simulation folder
        variation_dir = os.path.join(self.output_dir, self.scenario.name, self.workload.name, variation.name)


        os.makedirs(variation_dir, exist_ok=True)

        # with open(os.path.join(variation_dir, "schema.json"), "w+", encoding="utf-8") as f:
        with open(os.path.join(variation_dir, "schema.json"), "wb+") as f:
            # json.dump(asdict(schema), f)
            f.write(orjson.dumps(schema, default=orjson_default, option=orjson_options()))
            #json.dump(asdict(schema, dict_factory=lambda x: {k: v for (k, v) in x if v is not None}), f)

        # with open(os.path.join(variation_dir, f"requests_{repetition}.json"), "w", encoding="utf-8") as f:
        with open(os.path.join(variation_dir, f"requests_{repetition}.json"), "wb") as f:
            # json.dump([asdict(r) for r in requests], f)
            f.write(orjson.dumps(requests, default=orjson_default, option=orjson_options()))
            #json.dump([asdict(r, dict_factory=lambda x: {k: v for (k, v) in x if v is not None}) for r in requests], f)

        # with open(os.path.join(variation_dir, f"blocks_{repetition}.json"), "w", encoding="utf-8") as f:
        with open(os.path.join(variation_dir, f"blocks_{repetition}.json"), "wb") as f:
            # json.dump([asdict(b) for b in blocks], f)
            f.write(orjson.dumps(blocks, default=orjson_default, option=orjson_options()))
            #json.dump([asdict(b, dict_factory=lambda x: {k: v for (k, v) in x if v is not None}) for b in blocks], f)

        if rules_info is not None and len(rules_info.keys()) > 0:
            parent_path = os.path.join(self.output_dir, self.scenario.name, self.workload.name)
            json_data = {
                "scenario": self.scenario.name,
                "workload": self.workload.name,
                "variation": variation.name,
                "rulse": rules_info
            }
            with open(os.path.join(parent_path, f"rules_{repetition}.txt"), "a+", encoding="utf-8") as f:
                json.dump(json_data, f)
                f.write(",\n")



    def run(self, repetition_start: int = 0, n_repetitions: int = 1):

        self.dump_scenario_state()

        for repetition in tqdm(range(repetition_start, repetition_start+n_repetitions), desc="Repetitions"):
            for workload in tqdm(self.workloads, desc="Workloads", leave=False):

                # set the active workload
                self.workload = workload

                # clear state of simulation
                self.init_state()

                env = simpy.Environment()

                # initialize the arrival processes
                user_process = self.user_arrival_process(env=env)
                request_process = self.request_arrival_process(env=env)
                allocation_process = self.periodic_allocation(env=env)

                env.process(user_process)
                env.process(request_process)
                env.process(allocation_process)

                print(f"starting simulation...")

                # generate users and requests
                env.run(until=self.simulation_until)

                dump_time_start = time.time()

                # store the raw requests and block updates (before the final variations are applied)
                self.dump_workload_state(repetition=repetition)

                print(f"starting workload variations...")

                seed = random.randint(0, sys.maxsize) # seed for utility assigner  -> should stay the same for all variations (but different for workload + repetition)

                # simulation results post-processing
                for workload_variation in self.workload_variations:

                    requests = copy.deepcopy(self.requests)
                    updates = copy.deepcopy(self.block_updates)
                    schema = copy.deepcopy(self.scenario.pa_schema)

                    # assign utility to requests
                    workload_variation.utility_assigner.assign_utility(requests, seed=seed)

                    # assign request mode
                    schema, blocks, requests = workload_variation.mode_encoder.assign_mode(schema=schema, requests=requests, user_updates=updates)

                    self.dump_workload_variation(schema=schema, requests=requests, blocks=blocks, variation=workload_variation, repetition=repetition, rules_info=workload_variation.mode_encoder.get_rules_log())

                dump_time_end = time.time()
                print(f"JSON dumping took {dump_time_end - dump_time_start} seconds")




    def periodic_allocation(self, env):

        self.user_id_counter_history[env.now] = self.user_id_counter

        while True:
            yield env.timeout(self.allocation_interval)

            self.round_id += 1

            print(f"  allocation round {self.round_id}")

            self.user_id_counter_history[env.now] = self.user_id_counter

            # create an update that represents all users that joined in the last allocation interval
            self._process_new_users(self.round_id)

            if env.now >= self.start + self.allocation_interval:

                # batch the requests together
                self._process_new_requests()

    def user_arrival_process(self, env):

        # poisson process of newly arriving users
        while env.now <= self.start:
            interarrival = random.expovariate(self.pre_start_user_arrival_rate)
            yield env.timeout(interarrival)
            self._add_new_user(user_time=env.now)

        # NOTE: from here on, we start accepting requests and run the allocation periodically

        while True:
            interarrival = random.expovariate(self.user_arrival_rate)
            yield env.timeout(interarrival)
            self._add_new_user(user_time=env.now)

    def request_arrival_process(self, env):
        yield env.timeout(self.start)  # wait with requests until start time

        while True:
            interarrival = random.expovariate(self.request_arrival_rate)
            yield env.timeout(interarrival)
            req = self.workload.generate_request(round_id=self.round_id)
            self._add_new_request(req_time=env.now, req=req)

    def _add_new_user(self, user_time: int):
        self.users.append({"time": user_time, "user_id": self.user_id_counter})
        self.user_id_counter += 1

    def _add_new_request(self, req_time: int, req: dict):
        assert req_time >= self.start, "request time must be >= start time"

        self.requests_new.append(req)


    def _process_new_users(self, round_id: int):
        blk = block.UserUpdateInfo(round_id=round_id, n_new_users=len(self.users))
        self.block_updates.append(blk)
        self.users = []  # reset users

    def _process_new_requests(self):
        for req in self.requests_new:
            self.requests.append(req)
        self.requests_new = []  # reset new requests





def post_simulation_processing(allocation: Simulation, utility_assigner: utility.BaseUtilityAssigner, request_mode_assigner: mode.BaseModeEncoder):

    requests = allocation.get_requests()
    blocks = allocation.block_updates

    # assign utility to requests
    utility_assigner.assign_utility(requests)

    # assign request cost to requests (based on mode)
    request_mode_assigner.assign_mode(requests, blocks)
