from dataclasses import dataclass, field
import enum
import numpy as np
import random
from typing import Dict, List, Tuple
from scipy.stats import beta

from workload_simulator.schema.schema import CategoryData, PrivacyRiskLevel, AttributeName, AttributeCategory, CategoryCorrelationLevel

from .utility import zipf, sample_random_n_elements



@dataclass
class AttributeTagging:

    attribute_risk_level: PrivacyRiskLevel
    category_assignment: Dict[AttributeCategory, CategoryCorrelationLevel]


    def info(self):

        cat_info = {}
        for k, v in self.category_assignment.items():
            if v.name not in cat_info:
                cat_info[v.name] = []
            cat_info[v.name].append(k)

        return {
            "attribute_risk_level": self.attribute_risk_level.name,
            "category_assignment": cat_info
        }


@dataclass
class AttributeTaggingGenerator:

    attribute_risk_level_dist: Dict[PrivacyRiskLevel, int] = field(default_factory=lambda: {PrivacyRiskLevel.LOW: 80, PrivacyRiskLevel.MEDIUM: 10, PrivacyRiskLevel.HIGH: 10})

    category_risk_level_dist: Dict[PrivacyRiskLevel, int] = field(default_factory=lambda: {PrivacyRiskLevel.LOW: 50, PrivacyRiskLevel.MEDIUM: 40, PrivacyRiskLevel.HIGH: 10})
    category_correlation_level_dist: Dict[CategoryCorrelationLevel, int] = field(default_factory=lambda: {CategoryCorrelationLevel.MEMBER: 20, CategoryCorrelationLevel.MEMBER_STRONG: 40, CategoryCorrelationLevel.MEMBER_STRONG_WEAK: 40})

    # attribute_risk_level_dist: Dict[PrivacyRiskLevel, int] = field(default_factory=lambda: { PrivacyRiskLevel.HIGH: 100})
    #
    # category_risk_level_dist: Dict[PrivacyRiskLevel, int] = field(default_factory=lambda: {PrivacyRiskLevel.LOW: 50, PrivacyRiskLevel.MEDIUM: 40, PrivacyRiskLevel.HIGH: 10})
    # category_correlation_level_dist: Dict[CategoryCorrelationLevel, int] = field(default_factory=lambda: {CategoryCorrelationLevel.MEMBER: 20, CategoryCorrelationLevel.MEMBER_STRONG: 30, CategoryCorrelationLevel.MEMBER_STRONG_WEAK: 50})

    def generate(self, attributes: List[AttributeName], n_categories: int, n_categories_per_attribute: float) -> Tuple[Dict[AttributeName, AttributeTagging], List[CategoryData]]:

        category_correlation_levels = sorted(self.category_correlation_level_dist.keys(), key=lambda x: x.value)

        attribute_risk_levels_list: List[PrivacyRiskLevel] = sorted(self.attribute_risk_level_dist.keys(), key=lambda x: x.value)
        attribute_risk_level_weights = [self.attribute_risk_level_dist[rl] for rl in attribute_risk_levels_list]
        attribute_risk_levels = random.choices(population=attribute_risk_levels_list, weights=attribute_risk_level_weights, k=len(attributes))
        if len(attributes) == 1:
            print("Manually setting the attribute risk level to low for the only attribute")
            attribute_risk_levels = [PrivacyRiskLevel.LOW]
        print(f"{attribute_risk_levels=}")

        categories = [f"cat_{i}" for i in range(n_categories)]

        category_risk_levels: List[PrivacyRiskLevel] = sorted(self.category_risk_level_dist.keys(), key=lambda x: x.value)
        category_risk_level_weights = [self.category_risk_level_dist[rl] for rl in category_risk_levels]
        category_risk_levels = random.choices(population=category_risk_levels, weights=category_risk_level_weights, k=n_categories)

        if n_categories > 0 and len([c for c in category_risk_levels if c == PrivacyRiskLevel.HIGH]) == 0:
            raise ValueError("There should be at least one high-risk category for the simulation to be meaningful,"
                             "retry with a different seed or increase the probability of a high-risk category.")
        print(f"{category_risk_levels=}")

        # alpha = np.ones(n_categories) # could be other distribution
        # category_popularity = np.random.dirichlet(alpha) # TODO: Update this distribution

        # x = np.linspace(0.05, 0.8, num=n_categories)
        # category_popularity = beta.pdf(x, a=2, b=5)
        # weight_sum = np.sum(category_popularity)
        category_popularity = zipf(n_categories, alpha=0.1)
        # category_popularity = [1. / n_categories] * n_categories
        print(f"Category popularity {category_popularity}")

        category_risk_levels = list(map(lambda x: CategoryData(x[0], x[1], x[2]), zip(categories, category_risk_levels, category_popularity)))
        #print(f"{category_popularity=}")


        attribute_taggings = {}
        category_risk_levels_without_member = [c for c in category_correlation_levels if c != CategoryCorrelationLevel.MEMBER]
        category_risk_levels_without_member_weights = [self.category_correlation_level_dist[cl] for cl in category_risk_levels_without_member]
        for i, attr in enumerate(attributes):
            category_assignment = {}
            if n_categories > 0:
                assigned_categories = sample_random_n_elements(population=categories, weights=category_popularity, target_n=n_categories_per_attribute)
                # TODO: fix, sample exactly n_categories_per_attribute categories evenly
                # assigned_categories = random.sample(population=categories, k=int(n_categories_per_attribute))

                levels = random.choices(population=category_risk_levels_without_member, weights=category_risk_levels_without_member_weights,
                                        k=len(assigned_categories) - 1)
                category_assignment = { assigned_categories[0]: CategoryCorrelationLevel.MEMBER }
                category_assignment.update(dict(zip(assigned_categories[1:], levels)))

            tagging = AttributeTagging(attribute_risk_level=attribute_risk_levels[i], category_assignment=category_assignment)
            attribute_taggings[attr] = tagging
            print(f"{attr=}, {tagging.info()}")

        # low_risk_category = [c for c in category_risk_levels if c.privacy_risk_level == PrivacyRiskLevel.LOW][0]
        #
        # # Find the most frequent high-risk attribute
        # high_risk_attributes = [attr for attr, tagging in attribute_taggings.items() if tagging.attribute_risk_level == PrivacyRiskLevel.HIGH]
        # high_risk_attribute = high_risk_attributes[0]
        # attribute_taggings[high_risk_attribute].category_assignment = {low_risk_category.name: CategoryCorrelationLevel.MEMBER }




        # for i, attr in enumerate(attributes):
        #     attribute_risk_level = attribute_risk_levels[i]
        #     # assigned_categories = random.choices(population=categories, weights=category_popularity, k=n_categories_per_attribute)
        #     assigned_categories = sample_random_n_elements(population=categories, weights=category_popularity, target_n=n_categories_per_attribute)
        #     levels = random.choices(population=category_correlation_levels, weights=category_correlation_level_weights, k=len(assigned_categories))
        #     category_assignment = dict(zip(assigned_categories, levels))
        #     tagging = AttributeTagging(attribute_risk_level=attribute_risk_level, category_assignment=category_assignment)
        #     attribute_taggings[attr] = tagging
        #     print(f"{attr=}, {tagging.info()}")

        return attribute_taggings, category_risk_levels



@dataclass
class TimeAttributeTaggingGenerator:

    def generate(self, attributes: List[AttributeName], n_categories: int, n_categories_per_attribute: float) -> Tuple[Dict[AttributeName, AttributeTagging], List[CategoryData]]:

        assert len(attributes) == 2, "Expecting only two attributes: One static and one dynamic."
        assert n_categories == 0, "Categories not supported"

        # attr 0 is the static attribute, risk level LOW
        # attr 1 is the dynamic attribute, risk level MEDIUM (abusing risk level here slightly)
        attribute_risk_levels = [PrivacyRiskLevel.LOW, PrivacyRiskLevel.MEDIUM]
        category_risk_levels = []
        attribute_taggings = { attr: AttributeTagging(attribute_risk_level=attribute_risk_levels[i], category_assignment={}) for i, attr in enumerate(attributes) }
        print(f"{attribute_taggings=}")

        return attribute_taggings, category_risk_levels
