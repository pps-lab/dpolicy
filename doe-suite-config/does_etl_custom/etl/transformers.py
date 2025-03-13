from doespy.etl.steps.transformers import Transformer
from typing import List, Dict, Optional, Tuple, Any, Union
import pandas as pd
import itertools
import re

class ConcatColumnsTransformer(Transformer):

    dest: str
    src: List[str]
    separator: str = "-"

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        df[self.dest] = df[self.src].apply(lambda x: self.separator.join(x.astype(str)), axis=1)
        return df

class AcceptUnlimitedLabelTransformer(Transformer):
    groupby_cols: List[str] = ["scenario", "workload.name"]

    skip_suite_name: Optional[str] = None

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        # Initialize final DataFrame to collect results for each group
        final_result = []

        # Grouped processing
        for group_key, group_df in df.groupby(self.groupby_cols):
            # Filter rows for 'round_request_summary.csv' within the group
            group_df_summary = group_df[group_df['source_file'] == 'round_request_summary.csv']
            if group_df_summary.empty:
                continue  # Skip this group if no round_request_summary.csv is found
            if self.skip_suite_name is not None and group_df["suite_name"].iloc[0] == self.skip_suite_name:
                continue

            # Get first row values for the specified columns
            cols = ['suite_name', 'suite_id', 'exp_name', 'run']
            first_row_values = group_df_summary.iloc[0][cols]

            # Filter rows matching the first row values
            filtered_df = group_df_summary[group_df_summary[cols].eq(first_row_values).all(axis=1)].copy()

            # Set rejected columns to 0
            rejected_cols = [col for col in filtered_df.columns if col.endswith("_rejected")]
            filtered_df[rejected_cols] = 0

            # Set accepted columns to all values from corresponding "_all" columns
            accepted_cols = [col for col in filtered_df.columns if col.endswith("_accepted")]
            for col in accepted_cols:
                filtered_df[col] = filtered_df[col.replace("_accepted", "_all")]

            # Mark the run as special
            filtered_df["run"] = -1
            filtered_df["system_name"] = "unlimited"
            filtered_df["budget_name"] = "inf"

            # Append the transformed data to the group result
            final_result.append(pd.concat([group_df, filtered_df], ignore_index=True))

        # Concatenate all group results
        df_result = pd.concat(final_result, ignore_index=True)
        return df_result


class TrapAnalysisTransformer(Transformer):

    trap_type: str = "Attribute" # Attribute, Category
    risk_level: Optional[str] = None

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        source_files = df["source_file"].unique().tolist()

        print(f"source_files={source_files}")

        df_schema = df.loc[df['source_file'] == 'schema.json']
        has_schema = not df_schema.empty
        col_pattern = r"attribute_info\.attributes\.(.+)\.attribute_risk_level"
        # find columns with pattern and put values into dict with attribute as key
        attribute_risk_levels = {re.search(col_pattern, col).group(1): df_schema[col].iloc[0] for col in df_schema.columns if re.search(col_pattern, col)}

        # TODO: Report for all member levels
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        category_risk_levels = {}
        for risk_level in risk_levels:
            elems = df[f'attribute_info.categories.{risk_level}'].loc[~df[f'attribute_info.categories.{risk_level}'].isnull()].iloc[0]
            for elem in elems:
                category_risk_levels[f'{elem}-MEMBER'] = risk_level

        all_risk_levels = {
            "Attribute": attribute_risk_levels,
            "Category": category_risk_levels
        }

        df_workload = df.loc[df['source_file'] == 'trap_analysis_all.json']

        df_workload["run"] = -1
        df_workload["workload_mode"] = "unlimited-inf"
        df_workload["system_name"] = "unlimited"
        df_workload["budget_name"] = "inf"

        print(f"df_workload={df_workload}")
        assert len(df_workload.index) == 1, "trap_analysis_all.json is missing, or appears multiple times"

        df = df.loc[df['source_file'] == 'trap_analysis.json']

        df = pd.concat([df_workload, df], ignore_index=True)
        df.dropna(axis='columns', how='all', inplace=True)

        # Function to get max eps value for entries with "Attribute"
        def get_max_eps(stats):
            if self.risk_level is None or not has_schema:
                eps_values = [
                    entry["cost"]["EpsDeltaDp"]["eps"]
                    for entry in stats
                    if isinstance(entry["attribute_dim"], dict) and self.trap_type in entry["attribute_dim"]
                ]
            else:
                eps_values = [
                    entry["cost"]["EpsDeltaDp"]["eps"]
                    for entry in stats
                    if isinstance(entry["attribute_dim"], dict) and self.trap_type in entry["attribute_dim"] and
                       entry["attribute_dim"][self.trap_type] in all_risk_levels[self.trap_type] and all_risk_levels[self.trap_type][entry["attribute_dim"][self.trap_type]] == self.risk_level
                ]
            return max(eps_values) if eps_values else None

        df["max_epsilon"] = df["stats"].apply(get_max_eps)

        print(f"df_result={df}")

        return df



class TrapCategoryAnalysisTransformer(Transformer):

    trap_type: str = "Category" # Attribute, Category
    risk_level: Optional[str] = None

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        source_files = df["source_file"].unique().tolist()

        print(f"source_files={source_files}")

        df_schema = df.loc[df['source_file'] == 'schema.json']
        has_schema = not df_schema.empty
        col_pattern = r"attribute_info\.attributes\.(.+)\.attribute_risk_level"
        # find columns with pattern and put values into dict with attribute as key
        attribute_risk_levels = {re.search(col_pattern, col).group(1): df_schema[col].iloc[0] for col in df_schema.columns if re.search(col_pattern, col)}

        # TODO: Report for all member levels
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        correlation_levels = ['MEMBER', 'MEMBER_STRONG', 'MEMBER_STRONG_WEAK']
        category_risk_levels = {}
        for risk_level in risk_levels:
            elems = df[f'attribute_info.categories.{risk_level}'].loc[~df[f'attribute_info.categories.{risk_level}'].isnull()].iloc[0]
            for elem in elems:
                for correlation_level in correlation_levels:
                    category_risk_levels[f'{elem}-{correlation_level}'] = risk_level

        df_workload = df.loc[df['source_file'] == 'trap_analysis_all.json']

        df_workload["run"] = -1
        df_workload["workload_mode"] = "unlimited-inf"
        df_workload["system_name"] = "unlimited"
        df_workload["budget_name"] = "inf"

        print(f"df_workload={df_workload}")
        assert len(df_workload.index) == 1, "trap_analysis_all.json is missing, or appears multiple times"

        df = df.loc[df['source_file'] == 'trap_analysis.json']

        df = pd.concat([df_workload, df], ignore_index=True)
        df.dropna(axis='columns', how='all', inplace=True)

        def get_max_eps_wrap(correlation_level):
            def get_max_eps(stats):
                if self.risk_level is None or not has_schema:
                    eps_values = [
                        entry["cost"]["EpsDeltaDp"]["eps"]
                        for entry in stats
                        if isinstance(entry["attribute_dim"], dict) and self.trap_type in entry["attribute_dim"]
                    ]
                else:
                    eps_values = [
                        entry["cost"]["EpsDeltaDp"]["eps"]
                        for entry in stats
                        if isinstance(entry["attribute_dim"], dict) and self.trap_type in entry["attribute_dim"] and
                           entry["attribute_dim"][self.trap_type] in category_risk_levels and category_risk_levels[entry["attribute_dim"][self.trap_type]] == self.risk_level
                           and entry["attribute_dim"][self.trap_type].split("-")[1] == correlation_level
                    ]
                return max(eps_values) if eps_values else None
            return get_max_eps

        for correlation_level in correlation_levels:
            df[f"max_epsilon_{correlation_level}"] = df["stats"].apply(get_max_eps_wrap(correlation_level))

        # Compute as delta of each other
        for i, correlation_level in enumerate(correlation_levels):
            if i == 0:
                continue
            df[f"max_epsilon_delta_{correlation_levels[i-1]}_{correlation_level}"] = df[f"max_epsilon_{correlation_level}"] - df[f"max_epsilon_{correlation_levels[i-1]}"]

        print(f"df_result={df}")

        return df

class DetailedTrapAnalysisTransformer(Transformer):

    trap_type: str = "Attribute" # Attribute, Category

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        source_files = df["source_file"].unique().tolist()
        print(f"source_files={source_files}")

        # df_schema contains column attribute and attribute_risk_level
        df_schema = df.loc[df['source_file'] == 'schema.json']
        has_schema = not df_schema.empty

        df_workload = df.loc[df['source_file'] == 'trap_analysis_all.json']
        df_workload["run"] = -1
        df_workload["workload_mode"] = "unlimited-inf"
        df_workload["system_name"] = "unlimited"
        df_workload["budget_name"] = "inf"

        print(f"df_workload={df_workload}")
        assert len(df_workload.index) == 1, "trap_analysis_all.json is missing, or appears multiple times"

        df = df.loc[df['source_file'] == 'trap_analysis.json']

        df = pd.concat([df_workload, df], ignore_index=True)

        df = df.explode('stats')
        df["attribute"] = df["stats"].apply(lambda x: x["attribute_dim"]["Attribute"] if "Attribute" in x["attribute_dim"] else None)
        df["category"] = df["stats"].apply(lambda x: x["attribute_dim"]["Category"] if "Category" in x["attribute_dim"] else None)

        # group by suite_id, run, rep
        groupby_cols = ["suite_name", "suite_id", "run", "rep"]
        def compute_risk_level(x):
            return x.apply(
                lambda y: df_schema.loc[df_schema["attribute_name"] == y]["attribute_risk_level"].iloc[0] if y is not None else None
            )
        def compute_category_risk_level(x):
            return x.apply(
                lambda y: df_schema.loc[df_schema["category_name"] == y.split("-")[0]]["category_risk_level"].iloc[0] if y is not None else None
            )

        # Use `groupby().transform()` which returns a Series with the same index as `df`
        if has_schema:
            df["attribute_risk_level"] = df.groupby(groupby_cols)["attribute"].transform(compute_risk_level)
            df["category_risk_level"] = df.groupby(groupby_cols)["category"].transform(compute_category_risk_level)
        # df["attribute_risk_level"] = res.reset_index(level=groupby_cols, drop=True)
        df["eps"] = df["stats"].apply(lambda x: x["cost"]["EpsDeltaDp"]["eps"])

        # remove where both attribute and category are null
        df = df[(df["attribute"].notnull()) | (df["category"].notnull())]

        df = df.sort_values(by=["attribute", "category"])
        df = df.sort_values(by=["attribute"], key=lambda x: x.fillna("-1").str[1:].astype(int))

        return df


class TrapRelaxationAnalysisTransformer(Transformer):

    relaxation_type: str = "NONE" # Attribute, Category

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        source_files = df["source_file"].unique().tolist()

        print(f"source_files={source_files}")

        df_workload = df.loc[df['source_file'] == 'trap_analysis_all.json']

        df_workload["run"] = -1
        df_workload["workload_mode"] = "unlimited-inf"
        df_workload["system_name"] = "unlimited"
        df_workload["budget_name"] = "inf"

        print(f"df_workload={df_workload}")
        assert len(df_workload.index) == 1, "trap_analysis_all.json is missing, or appears multiple times"

        df = df.loc[df['source_file'] == 'trap_analysis.json']

        df = pd.concat([df_workload, df], ignore_index=True)
        df.dropna(axis='columns', how='all', inplace=True)

        # Function to get max eps value for entries with "Attribute"
        def get_max_eps(stats):
            eps_values = [
                entry["cost"]["EpsDeltaDp"]["eps"]
                for entry in stats if entry["relaxation"] == self.relaxation_type
            ]
            return max(eps_values) if eps_values else None

        df["max_epsilon"] = df["stats"].apply(get_max_eps)

        print(f"df_result={df}")

        return df


class RelativeCorrectionTransformer(Transformer):

    """Finds for each groupby cols the metric of selector and subtracts the relatives from each other"""

    groupby_cols: List[str] = ["scenario", "workload.name", "run", "budget_name"]

    metric: str
    selector: Dict[str, Union[str, float]] # add the rows that match one of these selection criteria
    relative_col: str
    relatives: List[str] # list of relative cols to subtract

    decomposed_col: Optional[str] = None

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        final_result = []
        for group_key, group_df in df.groupby(self.groupby_cols):
            # Find metric values based on entries in selector in the group
            output_df = group_df.copy()
            idx = pd.Series(True, index=output_df.index)
            idx = (~output_df[self.metric].isnull()) & idx
            for key, value in self.selector.items():
                idx = (output_df[key] == value) & idx
            if len(output_df[idx]) > 0:
                if self.decomposed_col is not None:
                    # the metric is split over two rows, for decomposed True and False.
                    # We need to add them up first and then subtract, and then decompose again
                    assert len(output_df[idx]) == 2*len(self.relatives), f"Expected {2*len(self.relatives)} rows, got {len(output_df[idx])}"
                    for i in range(1, len(self.relatives)):
                        idx_relative = (group_df[self.relative_col] == self.relatives[i]) & idx
                        idx_relative_prev = (group_df[self.relative_col] == self.relatives[i-1]) & idx
                        # add up values
                        original_bounded_value = group_df.loc[idx_relative & group_df[self.decomposed_col], self.metric].iloc[0]
                        if original_bounded_value > 0:
                            total_value = group_df.loc[idx_relative, self.metric].sum()
                            total_value_prev = group_df.loc[idx_relative_prev, self.metric].sum()
                            difference = total_value - total_value_prev
                            assert difference < original_bounded_value, "Expected the total value to be less than the original bounded value, otherwise we will get weird display issues."
                            output_df.loc[idx_relative & group_df[self.decomposed_col], self.metric] = difference
                            output_df.loc[idx_relative & ~group_df[self.decomposed_col], self.metric] = 0.0
                        else:
                            # we arent in violation, compute diff but we only update the non violation one
                            total_value = group_df.loc[idx_relative, self.metric].sum()
                            total_value_prev = group_df.loc[idx_relative_prev, self.metric].sum()
                            difference = total_value - total_value_prev
                            output_df.loc[idx_relative & ~group_df[self.decomposed_col], self.metric] = difference

                else:
                    assert len(output_df[idx]) == len(self.relatives), f"Expected {len(self.relatives)} rows, got {len(output_df[idx])}"
                    # find the rows where relative col and subtract from each other
                    for i in range(1, len(self.relatives)):
                        idx_relative = (output_df[self.relative_col] == self.relatives[i]) & idx
                        idx_relative_prev = (output_df[self.relative_col] == self.relatives[i-1]) & idx
                        assert len(output_df[idx_relative]) == 1, f"Expected 1 row, got {len(output_df[idx_relative])}"
                        output_df.loc[idx_relative, self.metric] = output_df.loc[idx_relative, self.metric].values[0] - output_df.loc[idx_relative_prev, self.metric].values[0]

            group_result = pd.DataFrame(output_df)
            group_result[self.groupby_cols] = group_key
            final_result.append(group_result)

        df_result = pd.concat(final_result, ignore_index=True)
        return df_result

class TrapCombinedAnalysisTransformer(Transformer):
    risk_level: str = "HIGH"
    privacy_time_unit: str = "UserMonth"
    time_attribute: str = "a1"

    groupby_cols: List[str] = ["suite_name", "suite_id", "scenario", "workload.name"]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        source_files = df["source_file"].unique().tolist()
        print(f"source_files={source_files}")

        # Schema-based processing
        df_schema = df.loc[df['source_file'] == 'schema.json']
        has_schema = not df_schema.empty
        attribute_risk_levels = self.extract_attribute_risk_levels(df_schema)

        # Category-based processing
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        correlation_levels = ['MEMBER', 'MEMBER_STRONG', 'MEMBER_STRONG_WEAK']
        category_risk_levels = self.extract_category_risk_levels(df, risk_levels, correlation_levels)

        # Initialize final DataFrame to collect results for each group
        final_result = []
        non_analysis_df = df[~df["source_file"].isin(["schema.json", "trap_analysis.json", "trap_analysis_all.json"])]

        # Grouped processing
        for group_key, group_df in df.groupby(self.groupby_cols):
            # Process trap_analysis_all.json for each group
            df_workload = group_df.loc[group_df['source_file'] == 'trap_analysis_all.json']
            assert len(df_workload.index) == 1, "trap_analysis_all.json is missing, or appears multiple times per group"
            df_workload["run"] = -1
            df_workload["workload_mode"] = "unlimited-inf"
            df_workload["system_name"] = "unlimited"
            df_workload["budget_name"] = "inf"

            # Process trap_analysis.json files for the group and combine with workload
            df_analysis = group_df.loc[group_df['source_file'] == 'trap_analysis.json']
            if group_key[0] != "trap-relax-cohere":
                df_combined = pd.concat([df_workload, df_analysis], ignore_index=True)
            else:
                df_combined = df_analysis
            df_combined.dropna(axis='columns', how='all', inplace=True)

            # Initialize new rows for this group
            new_rows = []

            # Process attributes
            for _, row in df_combined.iterrows():
                attribute_eps = self.get_max_eps(row["stats"], "Attribute", attribute_risk_levels, has_schema)
                if attribute_eps is not None:
                    new_row = row.copy()
                    new_row["max_epsilon"] = attribute_eps
                    new_row["correlation_level"] = None
                    new_row["trap_type"] = "attribute"
                    new_rows.append(new_row)

            # Process categories with different correlation levels
            for _, row in df_combined.iterrows():
                # Since category correlation is relative to the next, subtract
                # last_eps = 0
                for correlation_level in correlation_levels:
                    category_eps = self.get_max_eps(row["stats"], "Category", category_risk_levels, has_schema, correlation_level)
                    if category_eps is not None:
                        new_row = row.copy()
                        new_row["max_epsilon"] = category_eps
                        new_row["correlation_level"] = correlation_level
                        new_row["relaxation_type"] = None
                        new_row["trap_type"] = "category"
                        new_rows.append(new_row)
                        # last_eps = category_eps

            # Process relaxation
            for _, row in df_combined.iterrows():
                # Since NONE / BLACKBOX are also cumulative, subtract
                last_relaxation_eps = 0
                for relaxation_type in ["NONE", "BLACKBOX"]:
                    relaxation_eps = self.get_max_eps_relaxation(row["stats"], relaxation_type)
                    if relaxation_eps is not None:
                        new_row = row.copy()
                        new_row["max_epsilon"] = relaxation_eps - last_relaxation_eps
                        new_row["correlation_level"] = None
                        new_row["relaxation_type"] = relaxation_type
                        new_row["trap_type"] = "relaxation"
                        new_rows.append(new_row)
                        last_relaxation_eps = relaxation_eps

            # Process time
            for _, row in df_combined.iterrows():
                time_eps = self.get_max_eps_time(row["stats"])
                if time_eps is not None:
                    new_row = row.copy()
                    new_row["max_epsilon"] = time_eps
                    new_row["correlation_level"] = None
                    new_row["relaxation_type"] = None
                    new_row["trap_type"] = "time"
                    new_rows.append(new_row)

            # Add group result to the final result
            group_result = pd.DataFrame(new_rows)
            group_result[self.groupby_cols] = group_key  # Add group columns to each row
            final_result.append(group_result)

        # Concatenate all group results
        df_result = pd.concat(final_result + [non_analysis_df], ignore_index=True)
        df_result.dropna(axis='columns', how='all', inplace=True)
        print(f"df_result={df_result}")
        return df_result

    def extract_attribute_risk_levels(self, df_schema: pd.DataFrame) -> Dict[str, str]:
        col_pattern = r"attribute_info\.attributes\.(.+)\.attribute_risk_level"
        return {
            re.search(col_pattern, col).group(1): df_schema[col].iloc[0]
            for col in df_schema.columns if re.search(col_pattern, col)
        }

    def extract_category_risk_levels(self, df: pd.DataFrame, risk_levels: List[str], correlation_levels: List[str]) -> Dict[str, str]:
        category_risk_levels = {}
        for risk_level in risk_levels:
            if f'attribute_info.categories.{risk_level}' not in df.columns:
                continue
            elems = df[f'attribute_info.categories.{risk_level}'].loc[~df[f'attribute_info.categories.{risk_level}'].isnull()].iloc[0]
            for elem in elems:
                for correlation_level in correlation_levels:
                    category_risk_levels[f'{elem}-{correlation_level}'] = risk_level
        return category_risk_levels

    def get_max_eps(self, stats: List[Dict], trap_type: str, risk_levels: Dict[str, str], has_schema: bool, correlation_level: Optional[str] = None) -> Optional[float]:
        eps_values = [
            entry["cost"]["EpsDeltaDp"]["eps"]
            for entry in stats
            if isinstance(entry["attribute_dim"], dict) and trap_type in entry["attribute_dim"]
               and (not has_schema or (
                entry["attribute_dim"][trap_type] in risk_levels and
                risk_levels[entry["attribute_dim"][trap_type]] == self.risk_level and
                (correlation_level is None or entry["attribute_dim"][trap_type].endswith(correlation_level))
            ))
        ]
        return max(eps_values) if eps_values else None

    def get_max_eps_relaxation(self, stats: List[Dict], relaxation_type) -> Optional[float]:
        eps_values = [
            entry["cost"]["EpsDeltaDp"]["eps"]
            for entry in stats if entry.get("relaxation") == relaxation_type
        ]
        return max(eps_values) if eps_values else None

    def get_max_eps_time(self, stats: List[Dict]) -> Optional[float]:
        # if there are entries for the privacy time unit, get the max eps value
        # otherwise take the overall budget
        eps_values = [
            entry["cost"]["EpsDeltaDp"]["eps"]
            for entry in stats if entry.get("privacy_unit") == self.privacy_time_unit and
                                  isinstance(entry.get("attribute_dim"), dict) and "Attribute" in entry["attribute_dim"] and
                                entry["attribute_dim"]["Attribute"] == self.time_attribute

        ]
        private_time_max = max(eps_values) if eps_values else None
        if private_time_max is not None:
            return private_time_max
        return max([
            entry["cost"]["EpsDeltaDp"]["eps"]
            for entry in stats])


class FilterRowTransformer(Transformer):

    conditions: List[Dict[str, str]]
    """
    List of conditions that should be matched. Each condition is a dictionary with column name as key and value as value.
    """
    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        full_match = pd.Series(False, index=df.index)
        for condition in self.conditions:
            idx = pd.Series(True, index=df.index)
            for key, value in condition.items():
                idx = (df[key] == value) & idx
            full_match = full_match | idx

        print(f"Filtering out {len(df) - full_match.sum()} rows, resulting in {full_match.sum()} rows")
        return df[full_match]

class AssignSuiteTrapTypeTransformer(Transformer):

    """Assigns a relaxation type to a set of runs"""
    groupby_cols: List[str] = ["scenario", "workload.name"]

    assignment: List[Dict[str, Any]]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        # assign -> scenario, workload.name to a trap_type
        final_result = []
        # Grouped processing
        for group_key, group_df in df.groupby(self.groupby_cols):
            if list(group_key) not in [asi["key"] for asi in self.assignment]:
                print(f"Warning: No assignment for {group_key}")
                continue
            trap_type = next(asi["trap_type"] for asi in self.assignment if asi["key"] == list(group_key))
            # filter any rows not with the trap type
            group_df = group_df[(group_df["trap_type"] == trap_type) | group_df["trap_type"].isnull()]
            # add the trap type to the group
            group_df["trap_type"] = trap_type
            final_result.append(group_df)

        # Concatenate all group results
        df_result = pd.concat(final_result, ignore_index=True)
        return df_result


class GreaterThanTransformer(Transformer):

    """
    Adds a new column with the result of the comparison.
    Compares one or a sum of multiple values in each group for the comparison, but applies it to the whole column in the group
    """
    groupby_cols: List[str] = ["scenario", "workload.name", "run"]

    output_column: str
    metric: str
    comparison_value: float
    # expected_n_sum: Optional[int] = None # for debugging purposes to check we dont select too many columns
    selector: List[Dict[str, str]] = [] # add the rows that match one of these selection criteria

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        # Initialize final DataFrame to collect results for each group
        final_result = []
        # Grouped processing
        for group_key, group_df in df.groupby(self.groupby_cols):
            # Find metric values based on entries in selector in the group
            if self.selector:
                metric_values = []
                for sel in self.selector:
                    idx = pd.Series(True, index=group_df.index)
                    for key, value in sel.items():
                        idx = (group_df[key] == value) & idx
                    selected_df = group_df[idx]
                    assert len(selected_df) == 1 or len(selected_df) == 0, f"Expected 0 or 1 row, got {len(selected_df)}"
                    if len(selected_df) == 1:
                        metric_values.append(selected_df[self.metric].iloc[0])
                metric_value = sum(metric_values)
            else:
                print("Warning: No selector found, summing all rows")
                metric_value = group_df[self.metric].sum()

            cmp = metric_value > self.comparison_value
            group_df[self.output_column] = cmp

            group_result = pd.DataFrame(group_df)
            group_result[self.groupby_cols] = group_key  # Add group columns to each row
            final_result.append(group_result)

        # Concatenate all group results
        df_result = pd.concat(final_result, ignore_index=True)
        return df_result


class GreaterThanColumnTransformer(Transformer):

    """
    Adds a new column with the result of the comparison.
    Compares one or a sum of multiple values in each group for the comparison, but applies it to the whole column in the group
    """
    groupby_cols: List[str] = ["scenario", "workload.name", "run", "budget_name"]

    output_column: str
    metric: str
    # expected_n_sum: Optional[int] = None # for debugging purposes to check we dont select too many columns
    selector: List[Dict[str, Union[str, float]]] = [] # add the rows that match one of these selection criteria

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        assert self.selector is not None, "Selector must be provided"
        # Initialize final DataFrame to collect results for each group
        final_result = []
        # Grouped processing
        for group_key, group_df in df.groupby(self.groupby_cols):
            # Find metric values based on entries in selector in the group
            metric_values = []
            output_df = group_df.copy()
            output_df[self.output_column] = False
            for sel in self.selector:
                idx = pd.Series(True, index=output_df.index)
                for key, value in sel.items():
                    if key == "$comparison_value$":
                        continue
                    idx = (output_df[key] == value) & idx
                selected_df = output_df[idx]
                assert len(selected_df) == 1 or len(selected_df) == 0, f"Expected 0 or 1 row, got {len(selected_df)}"
                if len(selected_df) == 1:
                    metric_value = selected_df[self.metric].iloc[0]
                    cmp = metric_value > sel["$comparison_value$"]
                    output_df[self.output_column] = cmp

            group_result = pd.DataFrame(output_df)
            group_result[self.groupby_cols] = group_key  # Add group columns to each row
            final_result.append(group_result)

        # Concatenate all group results
        df_result = pd.concat(final_result, ignore_index=True)
        return df_result



class GreaterThanColumnSplitTransformer(Transformer):

    """
    Adds a new row with the result of the comparison.
    Compares one or a sum of multiple values in each group for the comparison, but applies it to the whole column in the group
    """
    groupby_cols: List[str] = ["scenario", "workload.name", "run", "budget_name"]

    output_column: str
    mark_column: str
    metric: str
    # expected_n_sum: Optional[int] = None # for debugging purposes to check we dont select too many columns
    selector: List[Dict[str, Union[str, float]]] = [] # add the rows that match one of these selection criteria

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        assert self.selector is not None, "Selector must be provided"
        # Initialize final DataFrame to collect results for each group
        final_result = []
        df[self.output_column] = False
        df[self.mark_column] = False
        # Grouped processing
        for group_key, group_df in df.groupby(self.groupby_cols):
            # Find metric values based on entries in selector in the group
            metric_values = []
            output_df = group_df.copy()
            for sel in self.selector:
                idx = pd.Series(True, index=output_df.index)
                for key, value in sel.items():
                    if key == "$comparison_value$":
                        continue
                    idx = (output_df[key] == value) & idx
                selected_df = output_df[idx]
                assert len(selected_df) == 1 or len(selected_df) == 0, f"Expected 0 or 1 row, got {len(selected_df)}"
                if len(selected_df) == 1:
                    selected_row = selected_df.iloc[0]
                    new_row = selected_row.copy()
                    new_row[self.output_column] = True
                    new_row[self.metric] = max(selected_row[self.metric] - sel["$comparison_value$"], 0)
                    output_df.loc[idx, self.metric] = min(sel["$comparison_value$"], selected_row[self.metric])
                    cmp = selected_row[self.metric] > sel["$comparison_value$"]
                    output_df[self.mark_column] = cmp
                    new_row[self.mark_column] = cmp
                    output_df = output_df.append(new_row, ignore_index=True)

            group_result = pd.DataFrame(output_df)
            group_result[self.groupby_cols] = group_key  # Add group columns to each row
            final_result.append(group_result)

        # Concatenate all group results
        df_result = pd.concat(final_result, ignore_index=True)
        return df_result


class MergeColumnSplitTransformer(Transformer):

    """
    Adds a new row with the result of the comparison.
    Compares one or a sum of multiple values in each group for the comparison, but applies it to the whole column in the group
    """
    groupby_cols: List[str] = ["scenario", "workload.name"]

    output_column: str
    mark_column: str
    metric: str
    # expected_n_sum: Optional[int] = None # for debugging purposes to check we dont select too many columns
    selector: List[Dict[str, Union[str, float]]] = [] # add the rows that match one of these selection criteria
    match_column: str
    match_rows: Dict[str, str] = {
        "eps3": "eps1.70",
        "eps5": "eps1.83",
        "eps7": "eps1.90",
        "eps10": "eps2.00",
        "eps15": "eps2.25",
        "eps20": "eps2.50",
    }

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        assert self.selector is not None, "Selector must be provided"
        # Initialize final DataFrame to collect results for each group
        final_result = []
        df[self.output_column] = False
        df[self.mark_column] = False
        # Grouped processing
        for group_key, group_df in df.groupby(self.groupby_cols):
            # Find metric values based on entries in selector in the group
            metric_values = []
            output_df = group_df.copy()
            for sel in self.selector:
                idx = pd.Series(True, index=output_df.index)
                for key, value in sel.items():
                    if key == "$comparison_value$":
                        continue
                    idx = (output_df[key] == value) & idx
                selected_df = output_df[idx]
                if len(selected_df) == 0:
                    continue
                all_keys_rows = list(self.match_rows.keys()) + list(self.match_rows.values())
                assert selected_df[selected_df[self.match_column] != "inf"][self.match_column].isin(all_keys_rows).all(), f"Expected all rows to be in {all_keys_rows}"

                # then we repurpose the rows
                for r2, r1 in self.match_rows.items():
                    idx_r1 = (selected_df[self.match_column] == r1)
                    idx_r2 = (selected_df[self.match_column] == r2)
                    assert len(selected_df[idx_r1]) == 1, f"Expected 1 row, got {len(selected_df[idx_r1])}"
                    assert len(selected_df[idx_r2]) == 1, f"Expected 1 row, got {len(selected_df[idx_r2])}"
                    split_value = selected_df.loc[idx_r2, self.metric].values[0] -  selected_df.loc[idx_r1, self.metric].values[0]
                    output_df_idx = idx & (output_df[self.match_column] == r2)
                    output_df.loc[output_df_idx, self.metric] = split_value
                    output_df.loc[output_df_idx, self.output_column] = True
                    output_df[self.mark_column] = True

            group_result = pd.DataFrame(output_df)
            group_result[self.groupby_cols] = group_key  # Add group columns to each row
            final_result.append(group_result)

        # Concatenate all group results
        df_result = pd.concat(final_result, ignore_index=True)
        return df_result



class RoundRequestSummaryTransformer(Transformer):

    config_cols: List[str] = [
        'source_file', 'allocation', 'composition',
        'scenario', 'workload_mode', 'workload_profit',
        'workload.mechanism_mix', 'workload.sub_mix',
        'workload.pa_mix', 'budget.mode',
        'budget.ModeConfig.Unlocking.slack', 'budget.type',
        'system_name', 'budget_name'
    ]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        # Filter to only process rows where source_file is 'round_request_summary.csv'
        df_round_request = df[df['source_file'] == 'round_request_summary.csv'].copy()
        df_other = df[df['source_file'] != 'round_request_summary.csv']

        # Perform transformations only on df_round_request
        df_round_request = self.preprocess(df_round_request)

        job_id_cols = ['suite_name', 'suite_id', 'exp_name', 'run', 'workload.name', 'workload.rep'] + self.config_cols
        # job_id_cols = df.columns

        category_columns = [
            "request_info.mechanism.name", "request_info.cost.name",
            "request_info.sampling.name", "request_info.utility.name"
        ]

        data_columns = ["profit", "n_requests"]

        df_round_request[data_columns] = df_round_request[data_columns].apply(pd.to_numeric)

        # Aggregation steps and calculations
        agg_d = {col: "sum" for col in data_columns}
        agg_d["round"] = "count"
        agg_d["allocation_status"] = lambda x: str(dict(x.value_counts()))

        df_accepted = df_round_request[df_round_request["status"] == "accepted_requests"].groupby(by=job_id_cols + category_columns).agg(agg_d)
        df_accepted.rename({"round": "n_rounds"}, axis=1, inplace=True)

        df_all = df_round_request[df_round_request["status"] == "all_requests"].groupby(by=job_id_cols + category_columns).agg(agg_d)
        df_all.rename({"round": "n_rounds"}, axis=1, inplace=True)

        # n_rounds = df_all["n_rounds"].max()
        # df_incomplete = df_all[df_all["n_rounds"] != n_rounds]
        # df_incomplete.reset_index(inplace=True)
        # df_incomplete = df_incomplete[job_id_cols].drop_duplicates()
        # df_incomplete.reset_index(inplace=True, drop=True)
        #
        # print("WARNING: We have incomplete runs => are filtered out:")
        # print(df_incomplete)
        #
        # df_all = df_all[df_all["n_rounds"] == n_rounds]
        # df_accepted = df_accepted[df_accepted["n_rounds"] == n_rounds]

        df1 = df_all.join(df_accepted, how='inner', lsuffix='_all', rsuffix='_accepted')

        df1["profit_rejected"] = df1["profit_all"] - df1["profit_accepted"]
        df1["n_requests_rejected"] = df1["n_requests_all"] - df1["n_requests_accepted"]

        main_category_columns = ["request_info.mechanism.name", "request_info.cost.name"]
        value_columns = [f"{col}_{x}" for col, x in itertools.product(data_columns, ["all", "accepted", "rejected"])]
        agg_d = {col: "sum" for col in value_columns}
        df1 = df1.groupby(by=job_id_cols + main_category_columns + ["allocation_status_all"]).agg(agg_d)

        df1.reset_index("request_info.cost.name", inplace=True)
        df_pivot = df1.pivot(columns="request_info.cost.name", values=value_columns)

        df_pivot = df_pivot.reset_index()
        df_pivot.columns = map(lambda x: x.strip("_"), df_pivot.columns.to_series().str.join('_'))
        df_pivot = df_pivot.reset_index(drop=True)

        def swap_last_two(s, sep='_'):
            parts = s.split(sep)
            if len(parts) >= 2:
                parts[-1], parts[-2] = parts[-2], parts[-1]
            return sep.join(parts)

        df_pivot.columns = [swap_last_two(col) if col.startswith("profit_") or col.startswith("n_requests_")  else col for col in df_pivot.columns]

        # Combine back with other rows that were not transformed
        result = pd.concat([df_pivot, df_other], ignore_index=True)
        return result

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # df.rename(columns={'workload.name': 'workload_info.name'}, inplace=True)

        if "budget.ModeConfig.Unlocking.slack" in df.columns:
            df['budget.ModeConfig.Unlocking.slack'].fillna('-', inplace=True)

        df.loc[df["composition"] == "block-composition", 'workload.pa_mix'] = 'nopa'

        x = df[df["budget.type"] != "RdpFromEpsDeltaDp"]
        if not x.empty:
            raise ValueError(f"budget type is not RdpFromEpsDeltaDp: {x['budget.type'].unique()}")

        return df

# class MergeRequestSummaryTransformer(Transformer):
#
#     groupby_cols: List[str] = ["scenario", "workload.name", "run", "budget_name"]
#     sum_cols: List[str] = ["profit_elephant_accepted", "profit_hare_accepted", "profit_mice_accepted"]
#     result_col: str = "profit_all_accepted"
#     source_file: str = "round_request_summary.csv"
#
#     def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
#
#         df_request_summary = df[df['source_file'] == self.source_file].copy()
#         df_rest = df[df['source_file'] != self.source_file].copy()
#
#         df_request_summary[self.result_col] = df_request_summary[self.sum_cols].sum(axis=1)
#
#         # Perform the groupby aggregation
#         # Save non-groupby columns to add back later
#         non_groupby_cols = [col for col in df_request_summary.columns if col not in self.groupby_cols + [self.result_col]]
#         df_non_groupby = df_request_summary[self.groupby_cols + non_groupby_cols].drop_duplicates()
#
#         df_request_summary = df_request_summary.groupby(self.groupby_cols).agg({self.result_col: "sum"}).reset_index()
#
#         # Merge back non-groupby columns
#         df_request_summary = df_request_summary.merge(df_non_groupby, on=self.groupby_cols, how="left")
#
#         # Concatenate the final result with the rest of the original DataFrame
#         df_result = pd.concat([df_request_summary, df_rest], ignore_index=True)
#
#         return df_result

class MergeRequestSummaryTransformer(Transformer):

    groupby_cols: List[str] = [
        'source_file', 'allocation', 'composition',
        'scenario', 'workload_mode', 'workload_profit',
        'workload.mechanism_mix', 'workload.sub_mix',
        'workload.pa_mix', 'budget.mode',
        'budget.ModeConfig.Unlocking.slack', 'budget.type',
        'system_name', 'budget_name',
        'run', 'workload.name'
    ]
    sum_cols: List[str] = ["profit_elephant_accepted", "profit_hare_accepted", "profit_mice_accepted",
                           "n_requests_elephant_accepted", "n_requests_hare_accepted", "n_requests_mice_accepted",
                           "profit_elephant_all", "profit_hare_all", "profit_mice_all",
                            "n_requests_elephant_all", "n_requests_hare_all", "n_requests_mice_all",
                            "profit_elephant_rejected", "profit_hare_rejected", "profit_mice_rejected",
                            "n_requests_elephant_rejected", "n_requests_hare_rejected", "n_requests_mice_rejected"
                           ]
    result_cols: List[str] = [
        "profit_all_accepted", "n_requests_all_accepted", "profit_all_all", "n_requests_all_all", "profit_all_rejected", "n_requests_all_rejected"
    ]
    source_file: str = "round_request_summary.csv"

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        # Identify columns to preserve that are neither in groupby_cols nor sum_cols
        df_request_summary = df[df['source_file'] == self.source_file].copy()
        df_rest = df[df['source_file'] != self.source_file].copy()

        # Raise an error if any column has more than one unique value per group

        # Create the aggregation dictionary
        agg_dict = {col: 'sum' for col in self.sum_cols}

        # Perform the groupby and aggregation
        df_grouped = df_request_summary.groupby(self.groupby_cols, as_index=False).agg(agg_dict)

        df_result = pd.concat([df_grouped, df_rest], ignore_index=True)
        # df_result[self.result_col] = df_request_summary[self.sum_cols].sum(axis=1)
        for i, result_col in enumerate(self.result_cols):
            df_result[result_col] = df_result[self.sum_cols[i*3:(i+1)*3]].sum(axis=1)

        return df_result
