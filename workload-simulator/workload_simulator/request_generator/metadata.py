
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import numpy as np

from scipy.stats import beta

from policy.utility import zipf
from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import AttributeName, BudgetRelaxation

from policy.attribute_tagging import AttributeTagging
from schema.schema import AttributeCategory, PrivacyRiskLevel, CategoryData


class Defaults:

    @staticmethod
    def new_single_uniform_attributes(attributes: Dict[AttributeName, AttributeTagging]):

        #Bell-Shaped Distribution:
        #When 𝛼>1 and 𝛽>1, the Beta Distribution takes on a bell-shaped, unimodal form. The peak of the distribution occurs at 𝛼−1 / 𝛼+𝛽−2, making it suitable for modeling distributions with a central tendency.
        # => for alpha=2 beta=20: peak at 0.05  (probability of up to 10% of attributes: 0.63)
        return UniformAttributeSelection(attributes=attributes, beta_a=2, beta_b=20)

    @staticmethod
    def new_powerlaw_attributes(attributes: Dict[AttributeName, AttributeTagging]):

        #Bell-Shaped Distribution:
        #When 𝛼>1 and 𝛽>1, the Beta Distribution takes on a bell-shaped, unimodal form. The peak of the distribution occurs at 𝛼−1 / 𝛼+𝛽−2, making it suitable for modeling distributions with a central tendency.
        # => for alpha=2 beta=20: peak at 0.05  (probability of up to 10% of attributes: 0.63)
        return PowerLawAttributeSelection(attributes=attributes, beta_a=1, beta_b=2.5) # 2, 20
        # return PowerLawZipfAttributeSelection(attributes=scenario.attributes, alpha=2)

    @staticmethod
    def new_powerlaw_category_based_attributes(attributes: Dict[AttributeName, AttributeTagging],
                                               categories: List[CategoryData]):
        # return PowerLawCategoryBasedAttributeSelection(attributes=attributes, categories=categories, alpha=2)
        return PowerLawZipfEqualAttributeSelection(attributes=attributes, categories=categories, alpha=1)


    @staticmethod
    def new_static_dynamic_attribute_assignment(attributes: Dict[AttributeName, AttributeTagging]):
        return StaticDynamicAttributeAssignment(attributes=attributes, p_static_dynamic=[0.5, 0.5])


    @staticmethod
    def new_no_budget_relaxation(scenario: ScenarioConfig):
        weights = {x: 0 for x in scenario.budget_relaxations}

        assert BudgetRelaxation.NONE in weights, "BudgetRelaxation.NONE must be included in the weights"

        weights[BudgetRelaxation.NONE] = 1
        return SingleBudgetRelaxationSelection(distribution=weights)


    @staticmethod
    def new_80blackbox_budget_relaxation(scenario: ScenarioConfig):

        weights = {
            BudgetRelaxation.NONE: 0.2,
            BudgetRelaxation.BLACKBOX: 0.8
        }

        assert set(weights.keys()) == set(scenario.budget_relaxations), "Weights must be defined for all relaxations"

        return SingleBudgetRelaxationSelection(distribution=weights)


class BaseAttributeSelection(ABC):

    @abstractmethod
    def generate(self) -> Tuple[Dict[AttributeName, AttributeTagging], List[CategoryData]]:
        pass

    @abstractmethod
    def info(self):
        pass



@dataclass
class UniformAttributeSelection(BaseAttributeSelection):


    attributes: Dict[AttributeName, AttributeTagging]

    # beta distribution controls the number of attributes selected
    beta_a: float
    beta_b: float

    # TODO: probably also a beta distribution should be used to asssign weights to attributes


    def generate(self):

        selection_percentage = beta.rvs(self.beta_a, self.beta_b, size=1)[0]

        n = math.ceil(len(self.attributes.keys()) * selection_percentage)

        # TODO: Each attribute is selected with equal probability -> which is not realistic. What should it be?
        weights = [1] * len(self.attributes.keys())

        selection_idx = random.choices(population=list(range(len(self.attributes))), weights=weights, k=n)

        selection_idx = list(set(selection_idx))  # remove duplicates
        selection = [list(self.attributes.keys())[i] for i in selection_idx]

        # return dict with only selected indices
        return { attr: self.attributes[attr] for attr in selection }, None


    def info(self):
        return {
            "name": __class__.__name__,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "distribution": "equal"
        }



@dataclass
class PowerLawAttributeSelection(BaseAttributeSelection):


    attributes: Dict[AttributeName, AttributeTagging]

    # beta distribution controls the number of attributes selected
    beta_a: float
    beta_b: float


    def __post_init__(self):
        x = np.linspace(0, 1, num=len(self.attributes))
        # assign weights according to power law distribution (beta with alpha=1)
        self.weights = beta.pdf(x, a=self.beta_a, b=self.beta_b) # NOTE: was 5 before
        weight_sum = np.sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def generate(self):

        selection_percentage = beta.rvs(self.beta_a, self.beta_b, size=1)[0]

        n = round(len(self.attributes.keys()) * selection_percentage)
        n = np.random.randint(0, 3)
        # print("Selection: ", n)

        if n > 0:
            selection_keys = np.random.choice(self.attributes.keys(), p=self.weights, size=n, replace=False)
            selection_keys = list(set(selection_keys.tolist()))
        else:
            selection_keys = []

        # filter dict
        selection_dict = {k: self.attributes[k] for k in selection_keys}
        return selection_dict, None  # No categories


    def info(self):
        return {
            "name": __class__.__name__,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "distribution": "powerlaw (a=1, b=5)"
        }

@dataclass
class StaticDynamicAttributeAssignment(BaseAttributeSelection):

    attributes: Dict[AttributeName, AttributeTagging]
    p_static_dynamic: List[float]

    def __post_init__(self):
        assert len(self.attributes) == 2, "attributes must have length 2"
        assert len(self.p_static_dynamic) == 2, "p_static_dynamic must have length 2"
        assert sum(self.p_static_dynamic) == 1, "p_static_dynamic must sum to 1"

    def generate(self):
        select_both = np.random.choice([False, True], p=self.p_static_dynamic)
        # dynaimc selects both
        if select_both:
            return self.attributes, None
        else:
            first_key = list(self.attributes.keys())[0]
            assert first_key == "a0", "First key must be a0"
            return { first_key: self.attributes[first_key] }, None

    def info(self):
        return {
            "name": __class__.__name__,
            "p_static_dynamic": self.p_static_dynamic
        }

@dataclass
class PowerLawZipfAttributeSelection(BaseAttributeSelection):

    attributes: List[AttributeName]

    # beta distribution controls the number of attributes selected
    alpha: float

    def __post_init__(self):
        x = np.linspace(0, 1, num=len(self.attributes))
        # assign weights according to zipf
        self.weights = 1 / (x + 1)**self.alpha
        weight_sum = np.sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def generate(self):

        selection_percentage = beta.rvs(2, 20, size=1)[0] # TODO: This is still beta

        n = round(len(self.attributes) * selection_percentage)

        if n > 0:
            selection_keys = np.random.choice(self.attributes.keys(), p=self.weights, size=n, replace=False)
            selection_keys = list(set(selection_keys.tolist()))
        else:
            selection_keys = []

        # filter dict
        selection_dict = {k: self.attributes[k] for k in selection_keys}
        return selection_dict, None  # No categories

    def info(self):
        return {
            "name": __class__.__name__,
            "alpha": self.alpha,
            "distribution": "zipf"
        }

@dataclass
class PowerLawCategoryBasedAttributeSelection(BaseAttributeSelection):

    attributes: Dict[AttributeName, AttributeTagging]
    categories: List[CategoryData]

    # beta distribution controls the number of attributes selected
    alpha: float

    def __post_init__(self):
        # # assign weights according to power law distribution (beta with alpha=1)
        # alpha = 2  # Shape parameter; lower values make the top items larger
        # self.weights = np.random.pareto(alpha, len(self.attributes.keys()))
        # # self.weights = zipf(len(self.attributes.keys()), alpha=self.alpha)
        # weight_sum = np.sum(self.weights)
        # self.weights = [w / weight_sum for w in self.weights]
        # self.attribute_weights = {attr: self.weights[i] for i, attr in enumerate(self.attributes.keys())}

        # # do weights by correlation level, first get attributes by corr level
        attributes_by_corr_level = { PrivacyRiskLevel.LOW: [], PrivacyRiskLevel.MEDIUM: [], PrivacyRiskLevel.HIGH: [] }
        for attr_name, tagging in self.attributes.items():
            attributes_by_corr_level[tagging.attribute_risk_level].append(attr_name)

        self.attribute_weights = {}
        for level, attributes in attributes_by_corr_level.items():
            if len(attributes) == 0:
                continue
            weights = zipf(len(attributes), alpha=self.alpha, scale_max_elements=len(self.attributes.keys()))
            weight_sum = np.sum(weights)
            # weights = [w / weight_sum for w in weights]
            for i, attr in enumerate(attributes):
                self.attribute_weights[attr] = weights[i]

        # normalize
        weight_sum = np.sum(list(self.attribute_weights.values()))
        self.attribute_weights = {k: v / weight_sum for k, v in self.attribute_weights.items()}

        # import matplotlib.pyplot as plt
        # sorted_data = dict(sorted(self.attribute_weights.items(), key=lambda item: -item[1]))
        # risk = [self.attributes[k].attribute_risk_level for k in sorted_data.keys()]
        # risk_color = {PrivacyRiskLevel.LOW: 'lightgrey', PrivacyRiskLevel.MEDIUM: 'grey', PrivacyRiskLevel.HIGH: 'black'}
        # color = [risk_color[r] for r in risk]
        #
        # plt.bar(sorted_data.keys(), sorted_data.values(), color=color)
        # plt.show()


    def generate(self):
        n_categories = 2
        # Sample n random categories
        category_names = list(map(lambda x: x.name, self.categories))
        category_weights = list(map(lambda x: x.weight, self.categories))
        assigned_categories_idx = self.sample_random_n_elements(population=list(range(len(category_names))), weights=category_weights, target_n=n_categories)
        assigned_categories = [self.categories[i] for i in assigned_categories_idx]

        selection = {}
        n_attributes = 4
        selected_category: CategoryData
        for selected_category in assigned_categories:
            # Sample n random attributes from category
            # attributes_in_category = [attr for attr, tag in self.attributes.items() if selected_category.name in tag.category_assignment]
            attributes_in_category = {attr: tag for attr, tag in self.attributes.items() if selected_category.name in tag.category_assignment}
            if len(attributes_in_category.keys()) == 0:
                continue
            attributes_in_category_weights = [self.attribute_weights[attr] for attr in attributes_in_category.keys()]
            # normalize
            attributes_in_category_weights = [w / sum(attributes_in_category_weights) for w in attributes_in_category_weights]
            selected_attributes_keys = self.sample_random_n_elements(population=list(attributes_in_category.keys()), weights=attributes_in_category_weights, target_n=n_attributes)
            selected_attributes = { attr: tag for attr, tag in attributes_in_category.items() if attr in selected_attributes_keys }
            selection.update(selected_attributes)

        return selection, assigned_categories

    def sample_random_n_elements(self, population: List, weights, target_n: int) -> List:
        # first sample n
        # n_elements = np.random.randint(1, target_n + 1)
        # return random.choices(population=population, k=n_elements)
        # "Bernoulli trial"
        sampling = True
        current_population = population
        current_weights = weights
        sampled = []
        # probability depends on expected target_n
        p_continue = 1 - (1 / target_n)
        while sampling:
            n_left = len(current_population)
            if n_left == 0:
                break
            if n_left == 1:
                sampled.append(current_population[0])
                break
            result = random.choices(population=list(range(n_left)), k=1, weights=current_weights)
            sampled.append(current_population[result[0]])
            current_population = [x for i, x in enumerate(current_population) if i != result[0]]
            current_weights = [x for i, x in enumerate(current_weights) if i != result[0]]
            # normalize
            current_weights = [w / sum(current_weights) for w in current_weights]

            sampling = np.random.choice([True, False], p=[p_continue, 1 - p_continue])

        # print(f"{sampled=}")

        return sampled

    def info(self):
        return {
            "name": __class__.__name__,
            "alpha": self.alpha,
            "distribution": f"powerlaw"
        }

@dataclass
class PowerLawZipfEqualAttributeSelection(BaseAttributeSelection):

    attributes: Dict[AttributeName, AttributeTagging]
    categories: List[CategoryData]

    # beta distribution controls the number of attributes selected
    alpha: float

    def __post_init__(self):
        attributes_by_corr_level = { PrivacyRiskLevel.LOW: [], PrivacyRiskLevel.MEDIUM: [], PrivacyRiskLevel.HIGH: [] }
        for attr_name, tagging in self.attributes.items():
            attributes_by_corr_level[tagging.attribute_risk_level].append(attr_name)

        self.attribute_weights = {}
        for level, attributes in attributes_by_corr_level.items():
            if len(attributes) == 0:
                continue
            weights = zipf(len(attributes), alpha=self.alpha, scale_max_elements=len(self.attributes.keys()))
            # weight_sum = np.sum(weights)
            # weights = [w / weight_sum for w in weights]
            for i, attr in enumerate(attributes):
                self.attribute_weights[attr] = weights[i]

        # normalize
        weight_sum = np.sum(list(self.attribute_weights.values()))
        self.attribute_weights = {k: v / weight_sum for k, v in self.attribute_weights.items()}
        # import matplotlib.pyplot as plt
        # sorted_data = dict(sorted(self.attribute_weights.items(), key=lambda item: -item[1]))
        # risk = [self.attributes[k].attribute_risk_level for k in sorted_data.keys()]
        # risk_color = {PrivacyRiskLevel.LOW: 'lightgrey', PrivacyRiskLevel.MEDIUM: 'grey', PrivacyRiskLevel.HIGH: 'black'}
        # color = [risk_color[r] for r in risk]
        #
        # plt.bar(sorted_data.keys(), sorted_data.values(), color=color)
        # plt.show()

    def generate(self):
        # Sample n random categories
        n_attributes = 4
        selected_attributes_keys = self.sample_random_n_elements(population=list(self.attributes.keys()), weights=list([self.attribute_weights[k] for k in self.attributes.keys()]), target_n=n_attributes)
        selected_attributes = { attr: self.attributes[attr] for attr in selected_attributes_keys }

        # select categories based on attributes
        assigned_categories = []
        for attr, tag in selected_attributes.items():
            for cat in tag.category_assignment:
                assigned_categories.append(cat)

        assigned_categories = list(set(assigned_categories))
        assigned_categories_data = [cat for cat in self.categories if cat.name in assigned_categories]
        return selected_attributes, assigned_categories_data

    def sample_random_n_elements(self, population: List, weights, target_n: int) -> List:
        # first sample n
        # n_elements = np.random.randint(1, target_n + 1)
        # return random.choices(population=population, k=n_elements)
        # "Bernoulli trial"
        sampling = True
        current_population = population
        current_weights = weights
        sampled = []
        # probability depends on expected target_n
        p_continue = 1 - (1 / target_n)
        while sampling:
            n_left = len(current_population)
            if n_left == 0:
                break
            if n_left == 1:
                sampled.append(current_population[0])
                break
            result = random.choices(population=list(range(n_left)), k=1, weights=current_weights)
            sampled.append(current_population[result[0]])
            current_population = [x for i, x in enumerate(current_population) if i != result[0]]
            current_weights = [x for i, x in enumerate(current_weights) if i != result[0]]
            # normalize
            current_weights = [w / sum(current_weights) for w in current_weights]

            sampling = np.random.choice([True, False], p=[p_continue, 1 - p_continue])

        # print(f"{sampled=}")

        return sampled

    def info(self):
        return {
            "name": __class__.__name__,
            "alpha": self.alpha,
            "distribution": "zipf"
        }

class BaseBudgetRelaxationSelection(ABC):

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def info(self):
        pass




@dataclass
class SingleBudgetRelaxationSelection(BaseBudgetRelaxationSelection):

    distribution: Dict[BudgetRelaxation, float]

    def generate(self):
        relaxations = list(self.distribution.keys())
        weights = [self.distribution[relaxation] for relaxation in relaxations]
        selection = random.choices(population=relaxations, weights=weights, k=1)[0]
        return selection


    def info(self):
        return {
            "name": __class__.__name__,
            "distribution": {k.name: v for k,v in self.distribution.items()}
        }