import random
import numpy as np
from M_E_GA import M_E_GA_Base, M_E_Engine
from SP_NN_Fitness import NetworkEvolutionFitness


class ExperimentRunner:
    def __init__(self, debug: bool = False, config_file=None):
        self.debug = debug

        # Network configuration
        self.config = {
            'network_params': {
                'volume_size': 70.0,
                'num_input': 10,
                'num_output': 20,
                'total_neurons': 300,
                'max_radius': 3.5,
                'min_radius': 0.90,
                'hidden_radius_range': (.5, 1.0),
                'input_radius_factor': 1.0,
                'interface_radius_factor': 2.0,
                'activation_budget': 1000,
                'time_window_size': 100,
                'activation_threshold': 0.5
            },
            'path_rewards': {
                'max_path_length': 100,
                'pickup_reward': 1.0,
                'successful_drop_reward': 6.0,
                'distance_penalty': 20.0,
                'proximity_factor': 2.0
            }
        }

        # GA configuration
        self.ga_config = {
            'mutation_prob': 0.15,
            'delimited_mutation_prob': 0.11,
            'open_mutation_prob': 0.10,
            'capture_mutation_prob': 0.05,
            'delimiter_insert_prob': 0.04,
            'delimit_delete_prob': 0.06,
            'crossover_prob': 0.00,
            'elitism_ratio': 0.00,
            'base_gene_prob': 0.35,
            'capture_gene_prob': 0.03,
            'max_individual_length': 50,
            'population_size': 700,
            'num_parents': 300,
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
                self.print_network_stats()

    def print_network_stats(self):
        if not self.debug or not hasattr(self, 'fitness_function'):
            return

        stats = self.fitness_function.get_stats()
        if 'health_metrics' in stats:
            health = stats['health_metrics']
            print(f"\nNetwork Health:")
            print(f"Combined Health: {health['combined_health']:.1f}%")
            print(f"Structural Connectivity: {health['structural_connectivity']:.1f}%")
            print(f"Connection Density: {health['connection_density']:.1f}%")

    def setup_experiment(self):
        # Define movement genes
        self.movement_genes = ['U', 'D', 'F', 'B', 'L', 'R']
        self.control_genes = ['SU', 'SD', 'TP', 'DR']
        all_genes = self.movement_genes + self.control_genes

        # Create fitness function with update callback
        self.fitness_function = NetworkEvolutionFitness(
            config=self.config,
            update_best_func=self.update_best_organism,
            debug=self.debug
        )

        # Initialize GA
        self.ga = M_E_GA_Base(
            genes=all_genes,
            fitness_function=lambda ind, ga_instance: self.fitness_function.compute(ind, ga_instance),
            **self.ga_config
        )

    def run_experiment(self):
        if self.debug:
            print("\nStarting Neural Evolution Experiment")
            print(f"Population size: {self.ga_config['population_size']}")
            print(f"Max generations: {self.ga_config['max_generations']}\n")

        # Run the GA
        self.ga.run_algorithm()

        # Get results
        best_genome = self.best_organism["genome"]
        best_fitness = self.best_organism["fitness"]
        best_solution = self.ga.decode_organism(best_genome) if best_genome else []

        if self.debug:
            print("\nExperiment Results:")
            print(f"Best Solution: {best_solution}")
            print(f"Best Fitness: {best_fitness}")
            print(f"Solution Length: {len(best_solution)}")
            self.print_network_stats()

        return {
            'best_genome': best_genome,
            'best_fitness': best_fitness,
            'best_solution': best_solution
        }


def run_single_experiment(debug: bool = False, seed: int = None):
    """Utility function to run a single experiment with optional debug and seed"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    experiment = ExperimentRunner(debug=debug)
    experiment.setup_experiment()
    return experiment.run_experiment()


if __name__ == "__main__":
    debug_mode = False  # Set to True only when debugging is needed
    random_seed = 42  # Set to None for random initialization
    results = run_single_experiment(debug=debug_mode, seed=random_seed)