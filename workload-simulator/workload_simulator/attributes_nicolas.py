import numpy as np
import matplotlib.pyplot as plt
from typing import List

n_attributes = 100
w_attributes = np.ones(n_attributes)

# w_attributes Zipf
w_attributes = np.array([1 / (i + 1)**(1/2) for i in range(n_attributes)])
p_attributes = w_attributes / w_attributes.sum()
# print(p_attributes)

n_categories = 10
w_categories = np.ones(n_categories)

# w_categories = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0.5])
# w_categories = np.array([1 / (i + 1) for i in range(n_categories)])
w_categories = np.array([1 / (i + 1)**(1/4) for i in range(n_categories)])

p_categories = w_categories / w_categories.sum()

# attr to cat
assignment: List[List[int]] = []
for i in range(n_attributes):
    # n_assigned = np.random.randint(1, 2)
    # assigned = np.random.choice(n_categories, n_assigned, replace=False).tolist()
    # assignment.append(assigned)
    # n_assigned = n_categories / n_attributes
    # assigned = [int(n_assigned * i)]
    # assignment.append(assigned)

    # construct assignment based on the idea that frequent categories are more likely to be assigned
    n_assigned = np.random.randint(1, 3)
    assigned = np.random.choice(n_categories, n_assigned, p=p_categories, replace=False).tolist()
    assignment.append(assigned)
    # base_probabilities = p_categories

    # Adjust probabilities based on the attribute frequency: higher freq -> bias towards earlier categories
    # exponent = np.log(1 + p_attributes[i] * 1000)
    # adjusted_probabilities = base_probabilities ** exponent  # Squaring freq to increase bias effect
    # adjusted_probabilities = adjusted_probabilities / adjusted_probabilities.sum()  # Normalize
    # print(p_attributes[i], exponent,  adjusted_probabilities)
    # assigned = np.random.choice(n_categories, n_assigned, p=adjusted_probabilities)
    # assignment.append(assigned)


# compute p(attribute | category)
p_attribute_given_category = np.zeros((n_attributes, n_categories))
for attr_id, cat_list in enumerate(assignment):
    for cat in cat_list:
        p_attribute_given_category[attr_id, cat] = p_attributes[attr_id] / p_categories[cat]
# normalize
p_attribute_given_category /= p_attribute_given_category.sum(axis=0)
# print(p_attribute_given_category)
# print(p_attribute_given_category[:, 0])

# scale parameter gamma
gamma = 0.5  # Change this value between 0 and 1 to control influence

# Sample 1000 requests
n_requests = 5000
all_sampled_attributes = []
all_sampled_categories = []

for _ in range(n_requests):
    sample_n_categories = np.random.randint(1, 4)
    sample_categories = np.random.choice(n_categories, sample_n_categories, p=p_categories,\
        replace=False)

    # sample_n_attributes = np.random.randint(1, 10)
    # sampled_attributes = np.random.choice(n_attributes, sample_n_attributes, p=p_attributes, replace=False)

    sampled_attributes = []
    for category in sample_categories:
        # sample an attribute and sample the next attribute
        sample_n_attributes = np.random.randint(1, 3)
        sampled_attribute = np.random.choice(n_attributes, sample_n_attributes, p=p_attribute_given_category[:, category])
        sampled_attributes.extend(sampled_attribute)

    # sampled_attributes = np.random.choice(n_attributes, sample_n_attributes, p=sample_weight / sample_weight.sum())
    all_sampled_attributes.extend(sampled_attributes)
    # count categories that we actually sampled based on sampled_attributes
    sample_categories = []
    for attr_id in sampled_attributes:
        sample_categories.extend(assignment[attr_id])
    sample_categories = list(set(sample_categories))
    all_sampled_categories.extend(sample_categories)

# Plotting the results
plt.figure(figsize=(20, 15))

# Plot w_attributes
plt.subplot(3, 3, 1)
plt.bar(range(n_attributes), w_attributes)
plt.title('Initial Attribute Weights')
plt.xlabel('Attribute ID')
plt.ylabel('Weight')

# Plot w_categories
plt.subplot(3, 3, 2)
plt.bar(range(n_categories), w_categories)
plt.title('Category Weights')
plt.xlabel('Category ID')
plt.ylabel('Weight')

# Plot sample_weight (example from last request)
# plt.subplot(3, 3, 3)
# plt.bar(range(n_attributes), sample_weight)
# plt.title('Sample Attribute Weights After Category Influence (One Sample)')
# plt.xlabel('Attribute ID')
# plt.ylabel('Weight')

# Visualize attribute to category assignment
plt.subplot(3, 3, 7)
for attr_id, cat_list in enumerate(assignment):
    for cat in cat_list:
        plt.plot([attr_id, attr_id], [0, cat], 'k-', alpha=0.1)
plt.scatter(
    [attr_id for attr_id, cat_list in enumerate(assignment) for _ in cat_list],
    [cat for sublist in assignment for cat in sublist],
    alpha=0.6
)
plt.title('Attribute to Category Assignment')
plt.xlabel('Attribute ID')
plt.ylabel('Category ID')

# Plot distribution of sampled attributes
plt.subplot(3, 3, 4)
plt.hist(all_sampled_attributes, bins=n_attributes, edgecolor='black')
plt.title('Distribution of Sampled Attributes')
plt.xlabel('Attribute ID')
plt.ylabel('Frequency')

# Plot distribution of sampled categories
plt.subplot(3, 3, 5)
plt.hist(all_sampled_categories, bins=n_categories, edgecolor='black')
plt.title('Distribution of Sampled Categories')
plt.xlabel('Category ID')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()