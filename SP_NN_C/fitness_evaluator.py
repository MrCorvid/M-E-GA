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
        Initialize the fitness evaluator with core mechanics and health metrics.

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

        # Weights for fitness components
        self.health_weight = 0.6  # Weight for network health
        self.path_weight = 0.4  # Weight for path execution
        self.structural_weight = 0.20  # Weight for structural connectivity within health
        self.density_weight = 0.80  # Weight for connection density within health

        # For tracking
        self.current_generation = 0
        self._last_health_metrics = None

    def compute_fitness(self, path: List[int]) -> float:
        """
        Compute fitness based on path execution and network health metrics.

        Args:
            path: List of integers representing movement commands

        Returns:
            float: Combined fitness score (minimum 0.1)
        """
        try:
            # Execute path and get results
            results = self.evolution_system.execute_movement_sequence(path)

            # Calculate raw path score
            path_score = self.evaluate_path_execution(results)

            # Get comprehensive network health metrics
            network_stats = self.evolution_system.get_network_stats()
            health_metrics = network_stats['health_metrics']

            # Store metrics for tracking
            self._last_health_metrics = health_metrics

            # Extract and normalize health components
            structural_score = health_metrics['structural_connectivity']
            density_score = health_metrics['connection_density']

            # Calculate combined network health score
            network_score = (self.structural_weight * structural_score +
                             self.density_weight * density_score)

            # Calculate overall fitness
            # Normalize path_score to 0-1 range (assuming max possible is around 100)
            normalized_path_score = path_score

            # Combine network health and path execution scores
            final_score = (path_score * network_score) * 100.0

            if self.debug:
                self._print_debug_info(results, path_score, health_metrics, final_score)

            # Ensure minimum positive fitness
            return max(0.1, final_score)

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return 0.1  # Return small positive value instead of 0

    def evaluate_path_execution(self, results: Dict) -> float:
        """
        Evaluate path execution using core mechanics.

        Returns:
            float: Raw path execution score
        """
        score = 0.0

        # Base rewards
        score += results['rotations_made'] * self.rotation_reward
        score += results['pickups_made'] * self.pickup_reward
        score += results['successful_drops'] * self.successful_drop_reward

        # Path length handling
        total_distance = results.get('path_length', 0)
        if total_distance <= self.max_path_length:
            score += total_distance
        else:
            score += self.max_path_length
            excess_distance = total_distance - self.max_path_length
            score -= excess_distance * self.distance_penalty

        return max(0.0, score)  # Ensure non-negative score

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
            'weights': {
                'health_weight': self.health_weight,
                'path_weight': self.path_weight,
                'structural_weight': self.structural_weight,
                'density_weight': self.density_weight
            },
            'last_health_metrics': self._last_health_metrics
        }

    def _print_debug_info(self, results: Dict, path_score: float,
                          health_metrics: Dict, final_fitness: float):
        """Print comprehensive debug information"""
        if not self.debug:
            return

        print("\nFitness Calculation Details:")
        print(f"Path Execution:")
        print(f"  Total Distance: {results.get('path_length', 0):.2f}")
        print(f"  Max Path Length: {self.max_path_length}")
        print(f"  Rotations: {results.get('rotations_made', 0)}")
        print(f"  Pickups: {results.get('pickups_made', 0)}")
        print(f"  Successful Drops: {results.get('successful_drops', 0)}")

        if results.get('path_length', 0) > self.max_path_length:
            excess = results['path_length'] - self.max_path_length
            penalty = excess * self.distance_penalty
            print(f"  Excess Distance: {excess:.2f}")
            print(f"  Distance Penalty: -{penalty:.2f}")

        print(f"\nNetwork Health Metrics:")
        print(f"  Structural Connectivity: {health_metrics['structural_connectivity']:.1f}%")
        print(f"  Connection Density: {health_metrics['connection_density']:.1f}%")
        print(f"  Combined Health: {health_metrics['combined_health']:.1f}%")

        print(f"\nScores:")
        print(f"  Raw Path Score: {path_score:.2f}")
        print(f"  Final Fitness: {final_fitness:.2f}")

        # Print metadata if available
        if 'metadata' in health_metrics:
            print("\nHealth Calculation Details:")
            meta = health_metrics['metadata']
            print(f"  Density Margin: {meta['density_margin']}")
            print(f"  Structural Weight: {meta['structural_weight']}")
            print(f"  Density Weight: {meta['density_weight']}")
            print(f"  Raw Density: {meta['raw_density']:.1f}%")
            print(f"  Adjusted Density: {meta['adjusted_density']:.1f}%")