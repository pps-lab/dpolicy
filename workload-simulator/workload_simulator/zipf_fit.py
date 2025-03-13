
import numpy as np

n_elements = 10

# compute zipf dist n_elements with alpha
alpha = 1
w_attributes = np.array([1 / (i + 1) ** alpha for i in range(n_elements)])

def zipf(n_elements, alpha=1):
    return np.array([1 / (i + 1) ** alpha for i in range(n_elements)])

print(f"Zipf dist: {w_attributes}")