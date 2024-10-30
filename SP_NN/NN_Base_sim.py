# main.py

import random
import numpy as np
from src.encoding import EncodingManager
from src.genetic import M_E_GA_Base
from src.simulation import BaseSimulation
from src.fitness import NetworkEvolutionFitness

class ExperimentRunner:
    def __init__(self, config_file=None):
        # Network configuration
        self.network_params = {
            'volume_size': 10.0,
            'total_neurons': 800,
            'activation_budget': 1000,
            'time_window_size': 100
        }

        # GA configuration
        self.ga_config = {
            'mutation_prob': 0.10,
            'delimited_mutation_prob': 0.08,
            'open_mutation_prob': 0.002,
            'capture_mutation_prob': 0.009,
            'delimiter_insert_prob': 0.01,
            'crossover_prob': 0.90,
            'elitism_ratio': 0.7,
            'base_gene_prob': 0.44,
            'capture_gene_prob': 0.04,
            'max_individual_length': 90,
            'population_size': 700,
            'num_parents': 100,
            'max_generations': 900,
            'delimiters': True,
            'delimiter_space': 2,
            'logging': True,
            'experiment_name': 'neural_evolution_exp',
            'seed': None
        }

        # Best solution tracking
        self.best_organism = {
            "genome": None,
            "fitness": float('-inf')
        }

    def update_best_organism(self, genome, fitness, verbose=True):
        if fitness > self.best_organism["fitness"]:
            self.best_organism["genome"] = genome
            self.best_organism["fitness"] = fitness
            if verbose:
                print(f"New best fitness: {fitness}")

    def setup_experiment(self):
        # Create simulation and fitness function
        self.simulation = BaseSimulation()
        self.fitness_function = NetworkEvolutionFitness(
            simulation=self.simulation,
            network_params=self.network_params,
            update_best_func=self.update_best_organism
        )

        # Initialize GA
        self.ga = M_E_GA_Base(
            genes=self.fitness_function.genes,
            fitness_function=lambda ind, ga_instance: self.fitness_function.compute(ind, ga_instance),
            **self.ga_config
        )

    def run_experiment(self):
        print("Starting experiment...")
        print(f"Network parameters: {self.network_params}")
        print(f"Population size: {self.ga_config['population_size']}")
        print(f"Max generations: {self.ga_config['max_generations']}")

        # Run the GA
        self.ga.run_algorithm()

        # Get results
        best_genome = self.best_organism["genome"]
        best_fitness = self.best_organism["fitness"]
        best_solution = self.ga.decode_organism(best_genome, format=True)

        print("\nExperiment Results:")
        print(f"Best Solution: {best_solution}")
        print(f"Best Fitness: {best_fitness}")
        print(f"Solution Length: {len(best_solution)}")

        return {
            'best_genome': best_genome,
            'best_fitness': best_fitness,
            'best_solution': best_solution
        }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create and run experiment
    experiment = ExperimentRunner()
    experiment.setup_experiment()
    results = experiment.run_experiment()