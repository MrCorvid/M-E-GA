from typing import Dict, Optional, Callable, List
from network_evolution import NetworkEvolution
from spatial_neural_network import NetworkParameters
import sys

class FitnessEvaluator:
    def __init__(self, 
                 config: Dict,
                 update_best_func: Optional[Callable] = None,
                 debug: bool = False):
        """
        Initialize the fitness evaluator.
        
        Args:
            config: Configuration dictionary containing network and path parameters
            update_best_func: Optional callback for updating best solution
            debug: Enable debug output
        """
        # Extract configuration parameters
        network_params = config.get('network_params', {})
        path_rewards = config.get('path_rewards', {})
        
        # Initialize the network evolution system
        self.evolution_system = NetworkEvolution(
            params=NetworkParameters(**network_params),
            debug=debug
        )
        
        # Path reward parameters directly from config
        self.max_path_length = path_rewards.get('max_path_length', 60)
        self.path_step_reward = path_rewards.get('path_step_reward', 1.00)
        self.pickup_reward = path_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = path_rewards.get('successful_drop_reward', 5.0)
        self.failed_drop_penalty = path_rewards.get('failed_drop_penalty', -0.0)
        self.empty_bag_reward = path_rewards.get('empty_bag_reward', 10.00)
        self.step_penalty = path_rewards.get('step_penalty', -10.0)
        
        self.debug = debug
        self.update_best = update_best_func
        
        # Define valid genes for path evolution
        self.genes = ['U', 'D', 'F', 'B', 'L', 'R', 'DR']

    def compute(self, encoded_individual: List[int], ga_instance) -> float:
        """
        Compute fitness score for an individual.
        
        Args:
            encoded_individual: Encoded genome representing movement sequence
            ga_instance: Genetic algorithm instance for decoding
        """
        # Decode and execute path
        path = ga_instance.decode_organism(encoded_individual) if ga_instance else encoded_individual
        movement_results = self.evolution_system.execute_movement_sequence(path)
        
        # Get raw path score from execution
        path_score = self._calculate_path_score(movement_results)
        
        # Calculate connectivity percentage (0-100)
        connectivity_score = int(self.evolution_system.calculate_connectivity())
        
        # Check if we've achieved 100% connectivity
        if connectivity_score == 100:
            if self.debug:
                print("\nExiting Phase 1: Achieved 100% network connectivity")
                print(f"Final Stats:")
                print(f"Neurons Moved: {movement_results['neurons_moved']}")
                print(f"Path Length: {movement_results['total_steps']}")
                print(f"Failed Drops: {movement_results['failed_drops']}")
            sys.exit(0)
        
        # Final fitness calculation matching original
        fitness = (path_score * self.path_step_reward) + (connectivity_score * 2)
        
        if self.debug:
            self._print_debug_info(movement_results, path_score, connectivity_score, fitness)
        
        if self.update_best:
            self.update_best(encoded_individual, fitness)
        
        return fitness

    def _calculate_path_score(self, results: Dict) -> float:
        """Calculate path score following original reward structure."""
        score = 0.0
        steps_within_limit = min(results['total_steps'], self.max_path_length)
        excess_steps = max(0, results['total_steps'] - self.max_path_length)
        
        # Rewards for steps within limit
        score += steps_within_limit * self.path_step_reward
        score += results['pickups_made'] * self.pickup_reward
        score += results['successful_drops'] * self.successful_drop_reward
        
        # Penalties
        score += results['failed_drops'] * self.failed_drop_penalty
        score += excess_steps * self.step_penalty
        
        # Empty bag bonus if applicable
        if len(self.evolution_system.pickup_bag) == 0:
            score += self.empty_bag_reward
            
        return score

    def get_stats(self) -> Dict:
        """Get current statistics from evolution system."""
        return {
            'connectivity': {
                'current': self.evolution_system.calculate_connectivity(),
                'unreachable_neurons': self.evolution_system.network.compute_unreachable_neurons()
            },
            'path': {
                'max_length': self.max_path_length,
                'path_step_reward': self.path_step_reward,
                'pickup_reward': self.pickup_reward,
                'successful_drop_reward': self.successful_drop_reward,
                'failed_drop_penalty': self.failed_drop_penalty,
                'empty_bag_reward': self.empty_bag_reward,
                'step_penalty': self.step_penalty
            },
            'network': self.evolution_system.get_network_stats()
        }

    def _print_debug_info(self, results: Dict, path_score: float, 
                         connectivity_score: float, fitness: float):
        """Print debug information matching original format."""
        print(f"\nFitness Calculation Details:")
        print(f"Path Length: {results['total_steps']}")
        print(f"Successful Drops: {results['neurons_moved']}")
        print(f"Failed Drops: {results['failed_drops']}")
        print(f"Raw Path Score: {path_score:.2f}")
        print(f"Connectivity Score: {connectivity_score}%")
        print(f"Combined Fitness: {fitness:.2f}")
