import random
import numpy as np
from M_E_GA import M_E_GA_Base, M_E_Engine
from SP_NN_Fitness import NetworkEvolutionFitness

class ExperimentRunner:
    def __init__(self, config_file=None):
        # Network configuration
        self.network_params = {
            'volume_size': 8.0,
            'total_neurons': 500,
            'activation_budget': 1000,
            'time_window_size': 50
        }

        # GA configuration
        self.ga_config = {
            'mutation_prob': 0.15,
            'delimited_mutation_prob': 0.08,
            'open_mutation_prob': 0.05,
            'capture_mutation_prob': 0.04,
            'delimiter_insert_prob': 0.07,
            'crossover_prob': 0.0,
            'elitism_ratio': 0.0,
            'base_gene_prob': 0.15,
            'capture_gene_prob': 0.3,
            'max_individual_length': 100,
            'population_size': 500,
            'num_parents': 100,
            'max_generations': 1000,
            'delimiters': False,
            'delimiter_space': 2,
            'logging': False,
            'experiment_name': 'neural_evolution',
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
        # Create fitness function with update callback
        self.fitness_function = NetworkEvolutionFitness(
            network_params=self.network_params,
            update_best_func=self.update_best_organism,
            max_path_length=90,
            path_step_reward=0.01
        )

        # Initialize GA
        self.ga = M_E_GA_Base(
            genes=self.fitness_function.genes,
            fitness_function=lambda ind, ga_instance: self.fitness_function.compute(ind, ga_instance),
            **self.ga_config
        )

    def run_experiment(self):
        print("Starting Neural Evolution Experiment...")
        print(f"Network parameters: {self.network_params}")
        print(f"Population size: {self.ga_config['population_size']}")
        print(f"Max generations: {self.ga_config['max_generations']}\n")

        # Run the GA
        self.ga.run_algorithm()

        # Get results
        best_genome = self.best_organism["genome"]
        best_fitness = self.best_organism["fitness"]
        best_solution = self.ga.decode_organism(best_genome, format=True) if best_genome is not None else []

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


    # Create and run experiment
    experiment = ExperimentRunner()
    experiment.setup_experiment()
    results = experiment.run_experiment()