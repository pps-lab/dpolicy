import os, json, math, itertools

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import numpy as np
import math
import warnings

import matplotlib.pyplot as plt

@dataclass
class RdpCost:

    eps_values: List[float] = None

    def __add__(self, other):
        return RdpCost(eps_values=[x + y for x, y in zip(self.eps_values, other.eps_values)])

    @staticmethod
    def empty(n_eps):
        return RdpCost(eps_values=n_eps * [0.0])

    def convert_approx(self, delta, alphas):
        assert len(self.eps_values) == len(alphas)
        min_eps = 100000000
        for eps, alpha in zip(self.eps_values, alphas):
            approx_eps = eps + math.log(1.0 / delta) / (alpha - 1)
            if approx_eps < min_eps:
                min_eps = approx_eps
        return min_eps


    # allows creating a report for all workloads in the scenario directory
def create_workload_report(scenario_dir, epsilon=3.0, delta=1e-7, n_active_rounds=12, slacks=[0.0], skip_pa_overlap=False):

    with open(os.path.join(scenario_dir, "workload_report.txt"), "w") as file:

        file.write("WORKLOAD REPORT\n")

        results = []

        for workload in os.listdir(scenario_dir):


            if not os.path.isdir(os.path.join(scenario_dir, workload)):
                continue

            #workload_mode = "ncd_dpolicy-eps3" if os.path.exists(os.path.join(scenario_dir, workload, "ncd_dpolicy")) else "equal-ncd_dpolicy-eps3"
            # assert os.path.exists(os.path.join(scenario_dir, workload, "equal-ncd_dpolicy"))

            for workload_mode in os.listdir(os.path.join(scenario_dir, workload)):

                if "dpolicy" not in workload_mode:
                    continue

                for rep_file in os.listdir(os.path.join(scenario_dir, workload, workload_mode)):
                    if not rep_file.startswith("requests_"):
                        continue

                    path = os.path.join(scenario_dir, workload, workload_mode, rep_file)

                    with open(path, "r") as f:
                        requests = json.load(f)

                    if not skip_pa_overlap:
                        create_pa_overlap_report(file=file, workload_name=workload, requests=requests, report_dir=scenario_dir)

                    # calculate request probabilities
                    req_probs = {}
                    weight_sum = sum(x["weight"] for x in requests[0]["workload_info"]["mechanism_mix"])
                    for x in requests[0]["workload_info"]["mechanism_mix"]:
                        mprob = x["weight"] / weight_sum

                        costs = {c["name"]: c["prob"] for c in x["mechanism"]["cost_calibration"]["distribution"]}
                        samplings = {c["name"]: c["prob"] for c in x["mechanism"]["sampling"]["distribution"]}
                        #print(f"costs={costs}, samplings={samplings}")
                        for c, s in itertools.product(costs.keys(), samplings.keys()):
                            prob = mprob * costs[c] * samplings[s]
                            req_probs[(x["mechanism"]["name"], c, s)] = prob

                    alphas = requests[0]["workload_info"]["cost_config"]["alphas"]

                    #print(alphas)
                    rdp_budget = convert_to_rdp(epsilon, delta, alphas)

                    total_utility = 0.0
                    # find unique costs and batch sizes
                    costs = {}
                    request_cost_type = 'User'
                    batch_sizes = {}
                    for r in requests:
                        rdp = tuple(r['request_cost'][request_cost_type]['Rdp']['eps_values'])
                        if rdp not in costs:
                            costs[rdp] = Value(count=0, info=r['request_info'], utility_sum=0.0, utilities=[])
                        costs[rdp].count += 1
                        costs[rdp].utility_sum += r['profit']
                        total_utility += r['profit']

                        costs[rdp].utilities.append(r['profit'])

                        if r["created"] not in batch_sizes:
                            batch_sizes[r["created"]] = 0
                        batch_sizes[r["created"]] += 1

                    expected_batch_size = sum(batch_sizes.values()) / len(batch_sizes)

                    for rdp_cost, v in costs.items():

                        info = extract(v.info)
                        prob = req_probs[(info["mechanism_name"], info["cost_name"], info["sampling_name"])]

                        for slack in slacks:

                            d = {
                                "workload_name": workload,
                                "repetition": rep_file,
                                **info,
                                **calc_n_possible(n_active_rounds, slack, rdp_budget, rdp_cost),
                                "exp_total_avl": int(n_active_rounds * expected_batch_size * prob),
                                "exp_round_avl": int(expected_batch_size * prob),
                                "slack": slack,
                                "prob": prob,
                                "utility_share": v.utility_sum / total_utility,
                            }

                            results.append(d)

                    create_utility_report(file=file, workload_name=workload, costs=costs, report_dir=scenario_dir)

                    ## Count utility per attribute
                    # Load schema
                    schema_path = os.path.join(scenario_dir, workload, workload_mode, "schema.json")
                    with open(schema_path, "r") as f:
                        schema = json.load(f)

                    if schema['attribute_info'] is not None:
                        n_eps = len(requests[0]['request_cost'][request_cost_type]['Rdp']['eps_values'])
                        member_types = ['MEMBER', 'MEMBER_STRONG', 'MEMBER_STRONG_WEAK']
                        categories_not_flat = schema['attribute_info']['categories'].values()
                        categories = [item for sublist in categories_not_flat for item in sublist]
                        categories_member = list(itertools.product(categories, member_types))
                        per_category_cost = {key: RdpCost.empty(n_eps) for key in categories_member}
                        attributes_cost = {}
                        for request in requests:
                            attributes = request['attributes']
                            request_cost = RdpCost(eps_values=request['request_cost'][request_cost_type]['Rdp']['eps_values'])
                            for attribute in attributes:
                                if attribute not in attributes_cost:
                                    attributes_cost[attribute] = request_cost
                                else:
                                    attributes_cost[attribute] += request_cost

                                attr_map = schema['attribute_info']['attributes'][attribute]['category_assignment']
                                for member_type, cats in attr_map.items():
                                    for cat in cats:
                                        per_category_cost[(cat, member_type)] += request_cost
                                # print(attr_map)

                        # histogram
                        attributes_cost_approx = {key: value.convert_approx(delta, alphas) for key, value in attributes_cost.items()}
                        attributes_risk_level = {key: schema['attribute_info']['attributes'][key]['attribute_risk_level'] for key in attributes_cost.keys()}
                        color_risk_level_map = {'LOW': 'lightgrey', 'MEDIUM': 'grey', 'HIGH': 'black'}
                        colors = {key: color_risk_level_map[value] for key, value in attributes_risk_level.items()}
                        create_histogram_report(file=file, workload_name=workload, report_dir=scenario_dir, data=attributes_cost_approx, colors=colors, xlabel="Attribute")

                        if len(categories) > 0:
                            categories_cost_approx = {key: value.convert_approx(delta, alphas) for key, value in per_category_cost.items()}
                            create_histogram_report(file=file, workload_name=workload, report_dir=scenario_dir, data=categories_cost_approx, colors=None, xlabel="Category")

                            # output grid of attribute category assignment, with 1 indicating the attribute is in the category
                            plot_attribute_category_grid(schema)

                break


        df = pd.DataFrame(results)

        print(df)

        columns = ["workload_name", "repetition", "slack", "mechanism_name", "cost_name", "sampling_name"]
        df.sort_values(by=columns, inplace=True)
        df.set_index(columns, inplace=True)

        prob_sum = df.groupby(["workload_name", "repetition", "slack"])['prob'].sum().unique()
        # TODO [nku] NOT SURE WHAT THIS IS CHECKING
        #assert all(math.isclose(x, 1.0) for x in prob_sum), f"probabilities do not sum up to 1.0  {prob_sum}"

        file.write(f"BUDGET: epsilon={epsilon}, delta={delta}, n_active_rounds={n_active_rounds}\n\n")

        for idx, df1 in df.groupby(["workload_name"]):
            file.write("=============================================\n")
            file.write(f"WORKLOAD SUMMARY: {idx}\n")


            file.write(df1.to_string(show_dimensions=True))

            file.write("\n-----------------------------------------\n\n")

            file.write("Can any request be accepted?\n")

            df2 = df1[df1["round_n_possible_max"] == 0]
            if df2.empty:
                file.write("  -> yes\n")
            else:
                file.write("  -> no. These requests can never be accepted (see slack):\n\n")

                file.write(df2.to_string(show_dimensions=True))
                file.write("\n\n")

                warnings.warn(f"workload {idx} has requests that can never be accepted (see workload_report.txt)")

            file.write("=============================================\n")


def plot_attribute_category_grid(schema):
    attributes = list(schema['attribute_info']['attributes'].keys())
    categories = [item for sublist in schema['attribute_info']['categories'].values() for item in sublist]
    category_risk_level = {}
    for risk_level, cats in schema['attribute_info']['categories'].items():
        for cat in cats:
            category_risk_level[cat] = risk_level
    grid = np.zeros((len(attributes), len(categories)))
    # Populate the grid with 1s where there is an assignment
    member_type_map = {
        "MEMBER": 3,
        "MEMBER_STRONG": 2,
        "MEMBER_STRONG_WEAK": 1
    }
    for i, attribute in enumerate(attributes):
        attr_map = schema['attribute_info']['attributes'][attribute]['category_assignment']
        for member_type, cats in attr_map.items():
            for cat in cats:
                if cat in categories:
                    j = categories.index(cat)
                    grid[i, j] = member_type_map[member_type]
    # Plotting the grid
    fig, ax = plt.subplots(figsize=(8, 20))
    cax = ax.matshow(grid, cmap="Greys", aspect="auto")
    # Set axis labels
    category_labels_with_risk = [f"{cat} ({category_risk_level[cat]})" for cat in categories]
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(attributes)))
    ax.set_xticklabels(category_labels_with_risk, rotation=90)
    ax.set_yticklabels(attributes)
    # Label the plot
    ax.set_xlabel("Categories")
    ax.set_ylabel("Attributes")
    plt.title("Attribute-Category Assignment Grid")
    plt.show()


@dataclass
class Value:
    count: int
    info: Dict
    utility_sum: float
    utilities: list

def extract(info: Dict):
    return {
        "mechanism_name": info["mechanism"]["mechanism"]["name"],
        "cost_name": info["cost_original"]["name"],
        "sampling_name": info["sampling_info"]["name"],
    }


def convert_to_rdp(epsilon, delta, alphas):
    return [epsilon - (math.log(1. / delta) / (alpha - 1)) for alpha in alphas]

# tighter conversion
#def convert_to_rdp(eps, delta, alphas):
#
#    def rdp_budget(epsilon, delta, alpha: float):
#
#        term = np.log((alpha-1)/alpha) - (np.log(delta) + np.log(alpha))/(alpha-1)
#        eps_rdp = max(0, epsilon - term)
#        return eps_rdp
#
#    res_vec = [rdp_budget(eps, delta, alpha) for alpha in alphas]
#    return res_vec

def calc_n_possible(n_active_rounds, slack, rdp_budget, rdp_cost):

    rdp_budget_round_min = [(1-slack) * x / n_active_rounds for x in rdp_budget]
    rdp_budget_round_max = [(1+slack) * x / n_active_rounds for x in rdp_budget]

    d = {}
    d["total_n_possible"] = max(math.floor(b / c) for c, b in zip(rdp_cost, rdp_budget))
    d["round_n_possible_min"] = max(math.floor(b / c) for c, b in zip(rdp_cost, rdp_budget_round_min))
    d["round_n_possible_max"] = max(math.floor(b / c) for c, b in zip(rdp_cost, rdp_budget_round_max))
    return d



def create_utility_report(file, workload_name, costs, report_dir):

    fig, ax = plt.subplots(1, 1)

    cmap = plt.cm.get_cmap('tab10', len(costs))

    for i, (rdp, v) in enumerate(costs.items()):

        info = extract(v.info)
        ax.hist(v.utilities, bins=int(len(v.utilities) / 4), label="-".join(info.values()), color=cmap(i), alpha=0.5)

    # Adding labels and title
    ax.set_xlabel('Utility')
    ax.set_ylabel('Frequency')
    ax.set_title('Utility Distribution')

    ax.legend()

    # Displaying the plot
    fig.savefig(os.path.join(report_dir, f"utility_hist_{workload_name}.png"))

    plt.close(fig)





def create_pa_overlap_report(file, workload_name, requests, report_dir):

    # without schema, we figure out the number of attributes
    max_val = 0
    for r in requests:
        for x in r['dnf']['conjunctions']:
            cand = x['predicates']['attr0']['Between']['max']
            if cand > max_val:
                max_val = cand


    # init every pa count with 0
    counts = [0] * (max_val + 1)
    counts_per_round = {i: [0] * (max_val + 1) for i in range(requests[0]['created'], requests[-1]['created'] + 1)}

    from tqdm import tqdm

    for r in tqdm(requests):

        for x in r['dnf']['conjunctions']:
            between = x['predicates']['attr0']['Between']
            for i in range(between['min'], between['max'] + 1):
                counts[i] += 1
                counts_per_round[r['created']][i] += 1



    file.write(f"PA OVERLAP REPORT:\n")

    file.write(f"n-requests-total = {len(requests)}\n")


    file.write(f"max_overlap_overall={max(counts)}\n")

    file.write(f"----------------------------------:\n")
    for round_id, x in counts_per_round.items():
        file.write(f"  round={round_id}: max_overlap_round={max(x)}\n")
    file.write(f"----------------------------------:\n")

    import matplotlib.pyplot as plt

    #for round_id, x in counts_per_round.items():
        #plt.plot(x, label=f"Round {round_id}")

    fig, ax = plt.subplots(1, 1)

    # Plotting the histogram
    ax.hist(counts, bins=300, edgecolor='black')

    # Adding labels and title
    ax.set_xlabel('Overlapping Requests per Block')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Overlapping Requests')

    # Displaying the plot
    fig.savefig(os.path.join(report_dir, f"pa_overlap_hist_{workload_name}.png"))

    plt.close(fig)


def create_histogram_report(file, workload_name, report_dir, data: Dict[str, int], colors: Dict[str, str], xlabel: str):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plotting the histogram
    # Sort by values
    sorted_data = dict(sorted(data.items(), key=lambda item: -item[1]))

    # Extract sorted categories and counts
    categories = list(sorted_data.keys())
    if isinstance(categories[0], tuple):
        categories = [f"{cat[0]}_{cat[1]}" for cat in categories]
    counts = list(sorted_data.values())
    if colors:
        ax.bar(categories, counts, edgecolor='black', color=[colors[cat] for cat in categories])
    else:
        ax.bar(categories, counts, edgecolor='black')

    # Adding labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cost (Approximate-DP epsilon')
    ax.set_xticklabels(categories, rotation=45, ha='right')    # ax.set_title('')
    plt.tight_layout()

    # Displaying the plot
    fig.savefig(os.path.join(report_dir, f"histogram_{xlabel}_{workload_name}.png"))

    plt.close(fig)
