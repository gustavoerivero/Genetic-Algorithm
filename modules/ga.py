import random
from deap import base, creator, tools
import config.config as config
import modules.functions as functions

def setup_ga():
  params = config.get_problem_config()

  if hasattr(creator, "FitnessMax"): del creator.FitnessMax
  if hasattr(creator, "Individual"): del creator.Individual

  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax)

  toolbox = base.Toolbox()

  toolbox.register("attr_bool", random.randint, 0, 1)
  total_bits = params["n_vars"] * params["bits_per_var"]

  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, total_bits)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)

  toolbox.register("evaluate", functions.evaluate_fitness)
  toolbox.register("select", tools.selRoulette)
  toolbox.register("mate", tools.cxOnePoint)
  toolbox.register("mutate", tools.mutFlipBit, indpb=config.MUTATION_PROB)

  return toolbox