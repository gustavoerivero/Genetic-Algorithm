import random
import numpy as np
from deap import tools, algorithms

import config.config as config
import modules.ga as ga
import modules.graphics as graphics
import modules.functions as functions

def main():
  random.seed(42)

  params = config.get_problem_config()

  print(f"[INFO] Iniciando optimización para el problema: {params['name']}")
  print(f"[INFO] Población: {params['population_size']} | Generaciones: {params['generations']}")

  toolbox = ga.setup_ga()
  poblation = toolbox.population(n=params["population_size"])
  hof = tools.HallOfFame(1)

  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("max", np.max)
  stats.register("avg", np.mean)

  poblation, logbook = algorithms.eaSimple(
    poblation, toolbox,
    cxpb=config.CROSSOVER_PROB,
    mutpb=config.MUTATION_PROB,
    ngen=params["generations"],
    stats=stats,
    halloffame=hof,
    verbose=True
  )

  best_individual = hof[0]
  fitness_final = best_individual.fitness.values[0]
  real_value = params["offset_roulette"] - fitness_final
  decoded_vars = functions.decode_chromosome(
    individual=best_individual,
    r_min=params["var_range"][0],
    r_max=params["var_range"][1],
    n_vars=params["n_vars"]
  )

  print("\n" + "="*40)
  print(f"[INFO] RESULTADO FINAL ({params['name']})")
  print("="*40)
  print(f"[INFO] Mínimo Encontrado: {real_value:.5f}")
  print(f"[INFO] Variables: {decoded_vars[:5]}")

  graphics.plot_convergence(logbook)

if __name__ == "__main__":
  main()