from typing import List

import numpy as np
import random

def zipf(n_elements, alpha=1, scale_max_elements=None):
    if scale_max_elements is not None:
        scale = float(scale_max_elements) / float(n_elements)
        values = np.array([1 / ((scale * i) + 1) ** alpha for i in range(n_elements)])
        return values
    else:
        values = np.array([1 / (i + 1) ** alpha for i in range(n_elements)])
        normalized = values / sum(values)
        return normalized

def sample_random_n_elements(population: List, weights, target_n: int) -> List:
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

    return sampled