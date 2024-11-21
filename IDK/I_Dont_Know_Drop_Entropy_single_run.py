import random
from M_E_GA_Base_V2 import M_E_GA_Base
import os
from I_Dont_Know_Drop_entropy import IDontKnow

GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

VOLUME = 20
NUM_PARTICLES = 400

best_organism = {
    "genome": None,
    "fitness": float('-inf')
}

def update_best_organism(current_genome, current_fitness, verbose=True):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

# Initialize the fitness function with update function passed in
fitness_function = IDontKnow(
    volume=VOLUME,
    num_particles=NUM_PARTICLES,
    update_best_func=update_best_organism,
    repulsion_strength=1.0,
    min_distance=1.0
)
genes = fitness_function.genes

config = {
    'mutation_prob': 0.15,
    'delimited_mutation_prob': 0.10,
    'open_mutation_prob': 0.009,
    'capture_mutation_prob': 0.003,
    'delimiter_insert_prob': 0.002,
    'crossover_prob': 0.00,
    'elitism_ratio': 0.00,
    'base_gene_prob': 0.40,
    'capture_gene_prob': 0.03,
    'max_individual_length': 100,
    'population_size': 500,
    'num_parents': 200,
    'max_generations': 1000,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': True,
    'generation_logging': True,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': False,
    'seed': GLOBAL_SEED
}

# Initialize the GA with the selected genes and the fitness function's compute method
ga = M_E_GA_Base(genes, lambda ind, ga_instance: fitness_function.compute(ind, ga_instance), **config)

# Run the GA
ga.run_algorithm()

# Find the best solution
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)

print('Length of best solution:', len(best_solution_decoded))
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
print('Length of best genome:', len(best_organism["genome"]))
print(f"Best Genome (Encoded): {best_genome}")