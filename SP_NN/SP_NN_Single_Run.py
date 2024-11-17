# SP_NN_Single_Run.py

import random
import numpy as np
from M_E_GA import M_E_GA_Base, M_E_Engine
from SP_NN_Fitness import NetworkEvolutionFitness


class ExperimentRunner:
    def __init__(self, debug: bool = True, config_file=None):
        self.debug = debug

        # Comprehensive configuration dictionary
        self.config = {
            'network_params': {
                # Core network structure
                'volume_size': 35.0,
                'num_input': 10,
                'num_output': 20,
                'total_neurons': 400,
                
                # Neuron radius parameters
                'max_radius': 3.0,
                'min_radius': 1.0,
                'hidden_radius_range': (.25, 1.0),
                'base_radius_shrink_rate': 0.95,
                'input_radius_factor': 1.0,
                'interface_radius_factor': 1.0,
                'interface_offset': 1.0,
                
                # Activation parameters
                'activation_budget': 1000,
                'time_window_size': 100,
                'activation_threshold': 0.5,
                'activation_radius_factor': 0.2
            },
            'path_rewards': {
                'max_path_length': 60,
                'path_step_reward': 1.00,
                'pickup_reward': 0.5,
                'successful_drop_reward': 5.0,
                'failed_drop_penalty': -0.0,
                'empty_bag_reward': 10.00,
                'step_penalty': -10.0
            }
        }

        # GA configuration - modified for navigation genome
        self.ga_config = {
            'mutation_prob': 0.15,
            'delimited_mutation_prob': 0.11,
            'open_mutation_prob': 0.10,
            'capture_mutation_prob': 0.04,
            'delimiter_insert_prob': 0.04,
            'delimit_delete_prob': 0.06,
            'crossover_prob': 0.00,
            'elitism_ratio': 0.00,
            'base_gene_prob': 0.35,
            'capture_gene_prob': 0.02,
            'max_individual_length': 50,
            'population_size': 400,
            'num_parents': 200,
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

    def update_best_organism(self, genome, fitness, verbose=False):
        if fitness > self.best_organism["fitness"]:
            self.best_organism["genome"] = genome
            self.best_organism["fitness"] = fitness
            print(f"New best fitness: {fitness}")
            if verbose:
                print(f"New best fitness: {fitness}")
                print("\nBest Genome:")
                # Print genome in chunks of 10 for readability
                genome_str = ''.join(map(str, genome))
                chunks = [genome_str[i:i+10] for i in range(0, len(genome_str), 10)]
                print(' '.join(chunks))
                self.print_network_stats()

    def print_network_stats(self):
        """Print detailed network statistics for debugging"""
        if hasattr(self, 'fitness_function'):
            stats = self.fitness_function.get_stats()
            if self.debug:
                print("\nNetwork Statistics:")
                print(f"Connectivity: {stats['connectivity']['current']:.2f}%")
                print(f"Unreachable neurons: {stats['connectivity']['unreachable_neurons']}")
                print("\nPath Metrics:")
                print(f"Max Length: {stats['path']['max_length']}")
                print(f"Step Reward: {stats['path']['path_step_reward']}")
                print(f"Pickup Reward: {stats['path']['pickup_reward']}")
                print(f"Drop Reward: {stats['path']['successful_drop_reward']}")
                print(f"Step Penalty: {stats['path']['step_penalty']}")
                print("\nNeuron Distribution:")
                print(f"Total neurons: {stats['network']['total_neurons']}")
                print(f"Input neurons: {stats['network']['input_neurons']}")
                print(f"Hidden neurons: {stats['network']['hidden_neurons']}")
                print(f"Output neurons: {stats['network']['output_neurons']}")
                print("\nCurrent State:")
                print(f"Pickup bag size: {stats['current_state']['pickup_bag_size']}")
                print(f"Current position: {stats['current_state']['current_position']}")

    def setup_experiment(self):
        # Create fitness function with update callback
        self.fitness_function = NetworkEvolutionFitness(
            config=self.config,
            update_best_func=self.update_best_organism,
            debug=self.debug
        )

        # Initialize GA with numeric genes [0-9]
        self.ga = M_E_GA_Base(
            genes=list(range(10)),  # Numeric genes for binary navigation encoding
            fitness_function=lambda ind, ga_instance: self.fitness_function.compute(ind, ga_instance),
            **self.ga_config
        )

    def run_experiment(self):
        if self.debug:
            print("Starting Neural Evolution Navigation Experiment...")
            print("\nComprehensive Configuration:")

            # Network Parameters
            print("\nNetwork Parameters:")
            for key, value in self.config['network_params'].items():
                print(f"  {key}: {value}")

            # Path Rewards
            print("\nPath Rewards:")
            for key, value in self.config['path_rewards'].items():
                print(f"  {key}: {value}")

            # GA Configuration
            print("\nGA Configuration:")
            print(f"  Population size: {self.ga_config['population_size']}")
            print(f"  Max generations: {self.ga_config['max_generations']}")
            print(f"  Mutation rate: {self.ga_config['mutation_prob']}")
            print(f"  Crossover rate: {self.ga_config['crossover_prob']}")
            print(f"  Genome length: {self.ga_config['max_individual_length']}\n")

        # Run the GA
        self.ga.run_algorithm()

        # Get results - genome is now list of integers
        best_genome = self.best_organism["genome"]
        best_fitness = self.best_organism["fitness"]

        # Print final results
        print("\nExperiment Results:")
        if best_genome is not None:
            print("Best Genome:")
            genome_str = ''.join(map(str, best_genome))
            chunks = [genome_str[i:i + 10] for i in range(0, len(genome_str), 10)]
            print(' '.join(chunks))
        print(f"Best Fitness: {best_fitness}")
        print(f"Genome Length: {len(best_genome) if best_genome else 0}")

        if self.debug:
            self.print_network_stats()

        return {
            'best_genome': best_genome,
            'best_fitness': best_fitness,
            'metrics': self.fitness_function.get_stats() if hasattr(self, 'fitness_function') else None
        }


if __name__ == "__main__":
    # Create and run experiment with debug flag
    debug_mode = True  # Set to True to enable debug output
    experiment = ExperimentRunner(debug=debug_mode)
    experiment.setup_experiment()
    results = experiment.run_experiment()
