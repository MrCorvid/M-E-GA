from typing import Dict, Optional
import random
import numpy as np
from M_E_GA_Base import M_E_GA_Base
from fitness_evaluator import FitnessEvaluator

class ExperimentRunner:
    def __init__(self, debug: bool = False, config_file=None):
        self.debug = debug

        # Comprehensive configuration dictionary matching original
        self.config = {
            'network_params': {
                # Match original SP_NN_Fitness.py parameters
                'volume_size': 8.0,
                'num_input': 100,
                'num_output': 4,
                'total_neurons': 500,
                'max_radius': 3.0,
                'min_radius': 0.1,
                'hidden_radius_range': (.50, 1.0),
                'base_radius_shrink_rate': 0.95,
                'input_radius_factor': 0.25,
                'interface_radius_factor': 0.3,
                'interface_offset': 1.0,
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

        # GA configuration remains the same
        self.ga_config = {
            'mutation_prob': 0.15,
            'delimited_mutation_prob': 0.08,
            'open_mutation_prob': 0.04,
            'capture_mutation_prob': 0.05,
            'delimiter_insert_prob': 0.04,
            'delimit_delete_prob': 0.05,
            'crossover_prob': 0.00,
            'elitism_ratio': 0.00,
            'base_gene_prob': 0.30,
            'capture_gene_prob': 0.04,
            'max_individual_length': 100,
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

        # Initialize components
        self.fitness_evaluator = None
        self.ga = None

    def update_best_organism(self, genome, fitness, verbose=True):
        """Update the best organism if a better solution is found."""
        if fitness > self.best_organism["fitness"]:
            self.best_organism["genome"] = genome
            self.best_organism["fitness"] = fitness
            if verbose and self.debug:
                print(f"\nNew best fitness: {fitness}")
                self.print_network_stats()

    def print_network_stats(self):
        """Print detailed network statistics for debugging."""
        if self.fitness_evaluator:
            stats = self.fitness_evaluator.get_stats()
            if self.debug:
                print("\nNetwork Statistics:")
                print(f"Connectivity: {stats['connectivity']['current']:.2f}%")
                print(f"Unreachable neurons: {stats['connectivity']['unreachable_neurons']}")
                print("\nNeuron Distribution:")
                print(f"Total neurons: {stats['network']['total_neurons']}")
                print(f"Input neurons: {stats['network']['input_neurons']}")
                print(f"Hidden neurons: {stats['network']['hidden_neurons']}")
                print(f"Output neurons: {stats['network']['output_neurons']}")
                print("\nActivation Parameters:")
                print(f"Activation budget: {self.config['network_params']['activation_budget']}")
                print(f"Time window: {self.config['network_params']['time_window_size']}")

    def setup_experiment(self):
        """Initialize all components needed for the experiment."""
        # Create fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(
            config=self.config,
            update_best_func=self.update_best_organism,
            debug=self.debug
        )

        # Initialize GA
        self.ga = M_E_GA_Base(
            genes=self.fitness_evaluator.genes,
            fitness_function=lambda ind, ga_instance: self.fitness_evaluator.compute(ind, ga_instance),
            **self.ga_config
        )

    def run_experiment(self):
        """Run the experiment and return results."""
        if not self.fitness_evaluator or not self.ga:
            raise RuntimeError("Experiment not set up. Call setup_experiment() first.")

        if self.debug:
            print("Starting Neural Evolution Experiment...")
            print("\nComprehensive Configuration:")
            for section, params in self.config.items():
                print(f"\n{section.replace('_', ' ').title()}:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
            print(f"\nPopulation size: {self.ga_config['population_size']}")
            print(f"Max generations: {self.ga_config['max_generations']}\n")

        # Run the GA
        self.ga.run_algorithm()

        # Get results
        best_genome = self.best_organism["genome"]
        best_fitness = self.best_organism["fitness"]
        best_solution = self.ga.decode_organism(best_genome, format=True) if best_genome is not None else []

        # Print final results regardless of debug setting
        print("\nExperiment Results:")
        print(f"Best Solution: {best_solution}")
        print(f"Best Fitness: {best_fitness}")
        print(f"Solution Length: {len(best_solution)}")

        if self.debug:
            self.print_network_stats()

        return {
            'best_genome': best_genome,
            'best_fitness': best_fitness,
            'best_solution': best_solution,
            'stats': self.fitness_evaluator.get_stats() if self.fitness_evaluator else None
        }

def run_experiment(config: Optional[Dict] = None, debug: bool = False):
    """
    Convenience function to run an experiment with optional configuration.
    
    Args:
        config: Optional configuration dictionary
        debug: Enable debug output
        
    Returns:
        Dict containing experiment results
    """
    # Create and run experiment
    experiment = ExperimentRunner(debug=debug)
    
    # If custom config provided, update the default config
    if config:
        if 'network_params' in config:
            experiment.config['network_params'].update(config['network_params'])
        if 'path_rewards' in config:
            experiment.config['path_rewards'].update(config['path_rewards'])
        if 'ga_config' in config:
            experiment.ga_config.update(config['ga_config'])
    
    experiment.setup_experiment()
    return experiment.run_experiment()

if __name__ == "__main__":
    # Example usage
    results = run_experiment(debug=True)
    print(f"\nExperiment completed with best fitness: {results['best_fitness']}")
