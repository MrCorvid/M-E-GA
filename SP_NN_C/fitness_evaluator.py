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
        Initialize the fitness evaluator with simple scoring rules.

        Args:
            evolution_system: NetworkEvolution instance
            movement_rewards: Dictionary of reward parameters
            debug: Enable debug output
        """
        self.evolution_system = evolution_system
        self.debug = debug

        # Core reward parameters
        self.max_path_length = movement_rewards.get('max_path_length', 60)
        self.path_step_reward = movement_rewards.get('path_step_reward', 1.00)
        self.pickup_reward = movement_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 5.0)
        self.step_penalty = movement_rewards.get('step_penalty', -3.0)

        # For tracking
        self.current_generation = 0
        self._last_connectivity = 0

    def compute_fitness(self, path: List[int]) -> float:
        """
        Compute fitness for a given path using simple scoring rules.

        Args:
            path: List of integers that will be converted to binary commands

        Returns:
            float: Computed fitness score
        """
        try:
            # Execute path and get results
            movement_results = self.evolution_system.execute_movement_sequence(path)

            # Calculate path score
            path_score = self.evaluate_path_execution(movement_results)

            # Get network connectivity (already 0-100)
            connectivity = self.evaluate_network_structure()

            # Final score is path_score * (connectivity/100)
            # This makes connectivity act as a multiplier (0.0 - 1.0)
            fitness = path_score ** (connectivity *.02)

            if self.debug:
                self._print_debug_info(movement_results, path_score, connectivity, fitness)

            return max(0.0, fitness)  # Ensure non-negative fitness

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return 0.0

    def evaluate_path_execution(self, results: Dict) -> float:
        """
        Evaluate path execution with simple scoring rules.
        """
        path_score = 0.0

        # Calculate base path length reward
        total_steps = len(results['positions'])
        if total_steps <= self.max_path_length:
            # Full reward for steps up to max length
            path_score += total_steps * self.path_step_reward
        else:
            # Reward max length steps, then subtract penalty for excess
            path_score += self.max_path_length * self.path_step_reward
            excess_steps = total_steps - self.max_path_length
            path_score += excess_steps * self.step_penalty

        # Add pickup and drop rewards
        path_score += results['pickups_made'] * self.pickup_reward
        path_score += results['successful_drops'] * self.successful_drop_reward

        return path_score

    def evaluate_network_structure(self) -> float:
        """
        Get network connectivity score (0-100).
        """
        connectivity = self.evolution_system.calculate_connectivity()

        # Update monitor if significant change
        if abs(connectivity - self._last_connectivity) > 1:
            self._last_connectivity = connectivity
            self.evolution_system.update_monitor_connectivity(connectivity)

        return connectivity

    def get_stats(self) -> Dict:
        """Get current evaluation statistics"""
        return {
            'current_generation': self.current_generation,
            'path_rewards': {
                'step_reward': self.path_step_reward,
                'pickup_reward': self.pickup_reward,
                'drop_reward': self.successful_drop_reward,
                'step_penalty': self.step_penalty
            },
            'last_connectivity': self._last_connectivity
        }

    def _print_debug_info(self, results: Dict, path_score: float,
                          connectivity: float, final_fitness: float):
        """Print debug information about fitness calculation"""
        if not self.debug:
            return

        print("\nFitness Calculation Details:")
        print(f"Path Length: {len(results['positions'])} steps")
        print(f"Max Path Length: {self.max_path_length}")
        print(f"Moves Made: {results['moves_made']}")
        print(f"Pickups: {results['pickups_made']}")
        print(f"Successful Drops: {results['successful_drops']}")
        print(f"Path Score: {path_score:.2f}")
        print(f"Network Connectivity: {connectivity:.1f}%")
        print(f"Final Fitness: {final_fitness:.2f}")