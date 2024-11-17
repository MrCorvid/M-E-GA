# fitness_evaluator.py

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
        Initialize the fitness evaluator.
        Args:
            evolution_system: NetworkEvolution instance
            movement_rewards: Dictionary of reward parameters
            debug: Enable debug output
        """
        self.evolution_system = evolution_system
        self.debug = debug

        # Initialize reward parameters
        self.max_path_length = movement_rewards.get('max_path_length', 60)
        self.path_step_reward = movement_rewards.get('path_step_reward', 1.00)
        self.pickup_reward = movement_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 5.0)
        self.failed_drop_penalty = movement_rewards.get('failed_drop_penalty', -0.0)
        self.empty_bag_reward = movement_rewards.get('empty_bag_reward', 10.00)
        self.step_penalty = movement_rewards.get('step_penalty', -10.0)

        # Generation tracking
        self.current_generation = 0
        self._last_connectivity = 0

    def compute_fitness(self, path: List[str]) -> float:
        """
        Compute fitness for a given path.
        Args:
            path: List of movement commands
        Returns:
            float: Computed fitness score
        """
        # Execute path and get results
        movement_results = self.evolution_system.execute_movement_sequence(path)

        # Evaluate path execution
        path_score = self.evaluate_path_execution(movement_results)

        # Evaluate network structure
        structure_score = self.evaluate_network_structure()

        # Combine scores for final fitness
        fitness = self._calculate_final_fitness(path_score, structure_score, movement_results)

        if self.debug:
            self._print_debug_info(movement_results, path_score, structure_score, fitness)

        return fitness

    def evaluate_path_execution(self, results: Dict) -> float:
        """
        Evaluate the execution of a path based on movement results.
        """
        path_score = 0.0

        # Base movement rewards
        path_score += results['moves_made'] * self.path_step_reward
        path_score += results['pickups_made'] * self.pickup_reward
        path_score += results['successful_drops'] * self.successful_drop_reward
        path_score += results['failed_drops'] * self.failed_drop_penalty

        # Step penalties
        if results['total_steps'] > self.max_path_length:
            excess_steps = results['total_steps'] - self.max_path_length
            path_score += excess_steps * self.step_penalty

        # Empty bag bonus
        if results['pickup_bag_size'] == 0:
            path_score += self.empty_bag_reward

        return path_score

    def evaluate_network_structure(self) -> float:
        """
        Evaluate the network structure based on connectivity.
        """
        connectivity = self.evolution_system.calculate_connectivity()

        # Update monitor if significant change
        if abs(connectivity - self._last_connectivity) > 1:
            self._last_connectivity = connectivity
            self.evolution_system.update_monitor_connectivity(connectivity)

        return connectivity

    def _calculate_final_fitness(self, path_score: float, structure_score: float, results: Dict) -> float:
        """
        Calculate final fitness combining path and structure scores.
        """
        # Base fitness from path execution
        fitness = path_score

        # Scale based on connectivity achievement
        connectivity_factor = (structure_score * 0.01)  # Convert to 0-1 range
        fitness *= (1.0 + connectivity_factor)

        # Penalties for inefficient solutions
        if results['total_steps'] > self.max_path_length:
            efficiency_penalty = 1.0 - (results['total_steps'] - self.max_path_length) / self.max_path_length
            efficiency_penalty = max(0.1, efficiency_penalty)  # Prevent negative fitness
            fitness *= efficiency_penalty

        return max(0.0, fitness)  # Ensure non-negative fitness

    def get_stats(self) -> Dict:
        """Get current evaluation statistics"""
        return {
            'current_generation': self.current_generation,
            'path_rewards': {
                'step_reward': self.path_step_reward,
                'pickup_reward': self.pickup_reward,
                'drop_reward': self.successful_drop_reward,
                'failed_penalty': self.failed_drop_penalty,
                'empty_bonus': self.empty_bag_reward,
                'step_penalty': self.step_penalty
            },
            'last_connectivity': self._last_connectivity
        }

    def _print_debug_info(self, results: Dict, path_score: float, structure_score: float, final_fitness: float):
        """Print debug information about fitness calculation"""
        if not self.debug:
            return

        print("\nFitness Calculation Details:")
        print(f"Path Length: {results['total_steps']}")
        print(f"Moves Made: {results['moves_made']}")
        print(f"Pickups: {results['pickups_made']}")
        print(f"Successful Drops: {results['successful_drops']}")
        print(f"Failed Drops: {results['failed_drops']}")
        print(f"Path Score: {path_score:.2f}")
        print(f"Network Connectivity: {structure_score:.1f}%")
        print(f"Final Fitness: {final_fitness:.2f}")