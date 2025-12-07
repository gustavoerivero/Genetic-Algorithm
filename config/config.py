PROBLEM_ID: int = 1

MUTATION_PROB: float = 0.01
CROSSOVER_PROB: float = 0.8

PROBLEMS = {
  1: {
    "name": "Schwefel",
    "n_vars": 30,
    "var_range": [-500, 500],
    "population_size": 1000,
    "generations": 200,
    "offset_roulette": 30000,
    "bits_per_var": 16
  },
  2: {
    "name": "Six-Hump Camel Back",
    "n_vars": 2,
    "var_range": [-5, 5],
    "population_size": 200,
    "generations": 100,
    "offset_roulette": 10000,
    "bits_per_var": 20
  }
}

def get_problem_config(problem_id: int = None):
  if problem_id is None:
    problem_id = PROBLEM_ID

  return PROBLEMS[problem_id]