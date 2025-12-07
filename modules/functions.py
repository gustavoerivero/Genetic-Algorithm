import math
import config.config as config

def decode_chromosome(individual, r_min, r_max, n_vars):
  variables = []
  bits_per_chunk = len(individual) // n_vars
  max_value = 2**bits_per_chunk - 1

  for i in range(n_vars):
    chunk = individual[i*bits_per_chunk : (i+1)*bits_per_chunk]
    int_value = int("".join(map(str, chunk)), 2)
    real_value = r_min + (int_value / max_value) * (r_max - r_min)
    variables.append(real_value)
  return variables

def evaluate_fitness(individual):
  PARAMS = config.get_problem_config()

  x = decode_chromosome(individual=individual, r_min=PARAMS["var_range"][0], r_max=PARAMS["var_range"][1], n_vars=PARAMS["n_vars"])
  val_function = 0.0

  if config.PROBLEM_ID == 1: # Schewefel
    val_function = sum([-xi * math.sin(math.sqrt(abs(xi))) for xi in x])

  elif config.PROBLEM_ID == 2: # Six-Hump Camel Back
    x1, x2 = x[0], x[1]
    term1 = 4 * x1**2 - 2.1 * x1**4 + (1/3) * x1**6
    term2 = x1 * x2
    term3 = -4 * x2**2 + 4 * x2**4
    val_function = term1 + term2 + term3

  fitness_roulette = PARAMS["offset_roulette"] - val_function
  return fitness_roulette,

