from typing import List, Dict
from network_evolution import NetworkEvolution


class FitnessEvaluator:
    def __init__(
            self,
            evolution_system: NetworkEvolution,
            movement_rewards: Dict,
            debug: bool = False
    ):
        """
        Initialize the fitness evaluator with core mechanics only.

        Args:
            evolution_system: NetworkEvolution instance
            movement_rewards: Dictionary of reward parameters
            debug: Enable debug output
        """
        self.evolution_system = evolution_system
        self.debug = debug

        # Core reward parameters
        self.max_path_length = movement_rewards.get('max_path_length', 60)
        self.rotation_reward = movement_rewards.get('rotation_reward', 0.5)
        self.pickup_reward = movement_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 5.0)
        self.distance_penalty = movement_rewards.get('distance_penalty', 0.1)

        # For tracking
        self.current_generation = 0
        self._last_connectivity = 0

    def compute_fitness(self, path: List[int]) -> float:
        """
        Compute fitness for a given path based on core mechanics.

        Args:
            path: List of integers that will be converted to binary commands

        Returns:
            float: Computed fitness score
        """
        try:
            # Execute path and get results
            results = self.evolution_system.execute_movement_sequence(path)
            
            # Calculate raw path score
            path_score = self.evaluate_path_execution(results)

            # Get network health metrics - Using correct method name
            network_stats = self.evolution_system.get_network_stats()
            health_metrics = network_stats['health_metrics']
            
            # Get combined health score (already includes both structural and density)
            combined_health = health_metrics['combined_health']

            # Store connectivity for tracking
            self._last_connectivity = health_metrics['structural_connectivity']
            
            # Update monitor if exists
            if self.evolution_system.monitor:
                self.evolution_system.monitor.update_health_metrics(health_metrics)

            # Final score = path_score * (combined_health/100)
            final_score = path_score * (combined_health / 100.0)

            if self.debug:
                self._print_debug_info(results, path_score, health_metrics, final_score)

            return max(0.0, final_score)

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return 0.0

    def evaluate_path_execution(self, results: Dict) -> float:
        """
        Evaluate path execution using core mechanics only.
        """
        score = 0.0

        # Base rewards
        score += results['rotations_made'] * self.rotation_reward
        score += results['pickups_made'] * self.pickup_reward
        score += results['successful_drops'] * self.successful_drop_reward

        # Path length handling
        total_distance = results['path_length']
        if total_distance <= self.max_path_length:
            score += total_distance
        else:
            score += self.max_path_length
            excess_distance = total_distance - self.max_path_length
            score -= excess_distance * self.distance_penalty

        return score

    def evaluate_network_structure(self) -> float:
        """Get network connectivity score (0-100)."""
        # Get complete health metrics using correct method
        network_stats = self.evolution_system.get_network_stats()
        health_metrics = network_stats['health_metrics']
        return health_metrics['combined_health']

    def get_stats(self) -> Dict:
        """Get current evaluation statistics"""
        return {
            'current_generation': self.current_generation,
            'rewards': {
                'rotation_reward': self.rotation_reward,
                'pickup_reward': self.pickup_reward,
                'drop_reward': self.successful_drop_reward,
                'distance_penalty': self.distance_penalty,
                'max_path_length': self.max_path_length
            },
            'last_connectivity': self._last_connectivity
        }

    def _print_debug_info(self, results: Dict, path_score: float,
                         health_metrics: Dict, final_fitness: float):
        """Print debug information about fitness calculation"""
        if not self.debug:
            return

        print("\nFitness Calculation Details:")
        print(f"Total Distance: {results['path_length']:.2f}")
        print(f"Max Path Length: {self.max_path_length}")
        print(f"Rotations: {results['rotations_made']}")
        print(f"Pickups: {results['pickups_made']}")
        print(f"Successful Drops: {results['successful_drops']}")
        
        if results['path_length'] > self.max_path_length:
            excess = results['path_length'] - self.max_path_length
            penalty = excess * self.distance_penalty
            print(f"Excess Distance: {excess:.2f}")
            print(f"Distance Penalty: -{penalty:.2f}")
        
        print(f"\nNetwork Health Metrics:")
        print(f"Structural Connectivity: {health_metrics['structural_connectivity']:.1f}%")
        print(f"Connection Density: {health_metrics['connection_density']:.1f}%")
        print(f"Combined Health: {health_metrics['combined_health']:.1f}%")
        
        print(f"\nScores:")
        print(f"Raw Path Score: {path_score:.2f}")
        print(f"Final Fitness: {final_fitness:.2f}")
        
        # Print metadata if available
        if 'metadata' in health_metrics:
            print("\nHealth Calculation Details:")
            meta = health_metrics['metadata']
            print(f"Density Margin: {meta['density_margin']}")
            print(f"Structural Weight: {meta['structural_weight']}")
            print(f"Density Weight: {meta['density_weight']}")
            print(f"Raw Density: {meta['raw_density']:.1f}%")
            print(f"Adjusted Density: {meta['adjusted_density']:.1f}%")
