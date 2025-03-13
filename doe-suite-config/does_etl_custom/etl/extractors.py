import json

from doespy.etl.steps.extractors import Extractor
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader

import pandas as pd
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import csv


class FilterCsvExtractor(Extractor):


    delimiter: str = ","

    has_header: bool = True

    fieldnames: List[str] = None


    config_filter: Dict[str, List[str]] = None


    file_regex: Union[str, List[str]] = [r".*\.csv$"]



    def extract(self, path: str, options: Dict) -> List[Dict]:
        # load file as csv: by default treats the first line as a header
        #   for each later row, we create a dict and add it to the result list to return

        # skip if config does not match
        config = options["$config_flat$"]
        for key, allowed_values in self.config_filter.items():
            if config[key] not in allowed_values:
                return []

        data = []

        with open(path, "r") as f:

            if self.has_header or self.fieldnames is not None:
                reader = csv.DictReader(f, delimiter=self.delimiter, fieldnames=self.fieldnames)
            else:
                reader = csv.reader(f, delimiter=self.delimiter)
            for row in reader:
                data.append(row)

        return data

class AttributeInfoExtractor(Extractor):

    def extract(self, path: str, options: Dict) -> List[Dict]:
        # read in schema.json
        with open(path, "r") as f:
            data = json.load(f)

        attrs = data["attribute_info"]["attributes"]

        info = [{
           "attribute_name": attr_name,
            "attribute_risk_level": attr["attribute_risk_level"]
        } for attr_name, attr in attrs.items()]

        categories = data["attribute_info"]["categories"]
        for risk_level, cats in categories.items():
            for cat in cats:
                info.append({
                    "category_name": cat,
                    "category_risk_level": risk_level
                })

        return info
