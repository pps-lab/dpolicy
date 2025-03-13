from doespy.etl.steps.extractors import Extractor
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader

import pandas as pd
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import warnings

import os
import json
import itertools

from multiprocessing import Pool, cpu_count
import tqdm
from dataclasses import dataclass, field


def _get_directories(exp_result_dir):

    def _list_dir_only(path):
        lst = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return lst

    directories = []

    runs = _list_dir_only(exp_result_dir)
    for run in runs:
        run_dir = f"{exp_result_dir}/{run}"
        reps = _list_dir_only(run_dir)
        for rep in reps:
            rep_dir = f"{run_dir}/{rep}"
            hosts = _list_dir_only(rep_dir)
        for host in hosts:
            host_dir = f"{rep_dir}/{host}"
            host_idxs = _list_dir_only(host_dir)
            for host_idx in host_idxs:
                data_dir = f"{host_dir}/{host_idx}"
                directories.append(data_dir)

    return sorted(directories)


@dataclass(eq=True, frozen=True)
class Key:
    round_id: int
    allocation_status: str
    mech_name: str
    cost_name: str
    sampling_name: str
    utility_name: str

@dataclass
class Value:
    request_profit: int = 0
    request_count: int = 0
    request_ids: List[int] = field(default_factory=list)
    request_n_virtual_blocks_distribution: List[int] = field(default_factory=list)


def update_stats(row, req, stats):

    round_id = row["round"]
    alloc_status = row["allocation_status"]

    key = Key(
            round_id = round_id,
            allocation_status = alloc_status,
            mech_name = req["request_info"]["mechanism"]["mechanism"]["name"],
            cost_name = req["request_info"]["cost_original"]["name"],
            sampling_name = req["request_info"]["sampling_info"]["name"],
            utility_name = req["request_info"]["utility_info"]["name"]
        )

    val = stats[key]
    val.request_profit += req["profit"]
    val.request_count += 1
    val.request_ids.append(req["request_id"])
    val.request_n_virtual_blocks_distribution.append(req["request_info"]["selection"]["n_virtual_blocks"])



def build_round_request_summary(run_dir):

    if os.path.exists(f"{run_dir}/round_request_summary.csv"):
        print("Round Request Summary already exists => skipping")
        return

    # load round log
    try:
        df_round_log = pd.read_csv(f"{run_dir}/round_log.csv")
    except pd.errors.EmptyDataError:
        warnings.warn(f"round_log.csv is empty: {run_dir}")
        return
    df_round_log = df_round_log.filter(items=['round', 'allocation_status', 'newly_available_requests', 'newly_accepted_requests'])
    df_round_log.loc[:, "newly_available_requests"] = df_round_log["newly_available_requests"].apply(json.loads)
    df_round_log.loc[:, "newly_accepted_requests"] = df_round_log["newly_accepted_requests"].apply(json.loads)

    df_round_log["allocation_status"] = df_round_log["allocation_status"].fillna("Optimal")

    # process requests
    with open(f"{run_dir}/all_requests.json") as f:
        requests = json.load(f)


    # init available stats (empty value for every possible combination)
    available_stats = {}
    accepted_stats = {}
    for _index, row in df_round_log.iterrows():
        round_id = row["round"]
        alloc_status = row["allocation_status"]

        utility = requests[0]["request_info"]["utility_info"]["name"]
        for entry in requests[0]["workload_info"]["mechanism_mix"]:
            mech_name = entry["mechanism"]["name"]
            costs =  [x["name"] for x in entry["mechanism"]["cost_calibration"]["distribution"]]
            samplings =  [x["name"] for x in entry["mechanism"]["sampling"]["distribution"]]
            for cost, sampling in itertools.product(costs, samplings):
                k = Key(round_id=round_id, allocation_status=alloc_status, mech_name=mech_name, cost_name=cost, sampling_name=sampling, utility_name=utility)
                available_stats[k] = Value()
                accepted_stats[k] = Value()


    # transform requests to dict
    requests = {r["request_id"]: r for r in requests}

    # go through rounds and create summary based on requests
    for _index, row in df_round_log.iterrows():
        round_id = row["round"]
        alloc_status = row["allocation_status"]

        for rid in row['newly_available_requests']:
            req = requests[rid]
            update_stats(row, req, available_stats)

        for rid in row['newly_accepted_requests']:
            req = requests[rid]
            update_stats(row, req, accepted_stats)


    def todict(k, v):
        return {
            # key
            "round": k.round_id,
            "allocation_status": k.allocation_status,
            "request_info.mechanism.name": k.mech_name,
            "request_info.cost.name": k.cost_name,
            "request_info.sampling.name": k.sampling_name, # maybe
            "request_info.utility.name": k.utility_name, # maybe

            # value
            "profit": v.request_profit,
            "n_requests": v.request_count,
            "request_ids": v.request_ids,
            "request_n_virtual_blocks_distribution": v.request_n_virtual_blocks_distribution,
        }



    # convert to results
    lst = []
    for k, v in available_stats.items():
        d = todict(k, v)
        d["status"] = "all_requests"
        lst.append(d)

    for k, v in accepted_stats.items():
        d = todict(k, v)
        d["status"] = "accepted_requests"
        lst.append(d)
    df = pd.DataFrame(lst)

    # store the result in the folder
    df.to_csv(f"{run_dir}/round_request_summary.csv", index=False)

    #print(f"Done: {run_dir}")


class PreProcessingDummyExtractor(Extractor):

    file_regex: Union[str, List[str]] = ["stderr.log"]


    def extract(self, path: str, options: Dict) -> List[Dict]:
        base_dir = path
        # go multiple levels back
        for _ in range(3):
            base_dir = os.path.dirname(base_dir)

        if os.path.basename(base_dir) == "rep_0" and os.path.basename(os.path.dirname(base_dir)) == "run_0":
            # we only do work once for the first run first rep
            base_dir = os.path.dirname(os.path.dirname(base_dir))

            directories = _get_directories(base_dir)

            with Pool(processes=cpu_count()) as p:
                #p.map(build_round_request_summary, directories)
                for _ in tqdm.tqdm(p.imap_unordered(build_round_request_summary, directories), total=len(directories)):
                    pass

        return []
