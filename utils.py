import numpy as np

def randomize_parameter_uniform(initial_value, rand_prop):
    lower_bound = initial_value*(1-rand_prop)
    upper_bound = initial_value*(1+rand_prop)
    return np.random.uniform(lower_bound, upper_bound)