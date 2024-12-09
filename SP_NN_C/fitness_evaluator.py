from typing import List, Dict
from network_evolution import NetworkEvolution


class FitnessEvaluator:
    """
    Evaluates fitness for spatial neural network paths with dynamic rewards and stagnation detection.

    Parameters:
        evolution_system: NetworkEvolution instance for executing paths
        movement_rewards: Dictionary containing reward parameters
        debug: Enable detailed debug output
    """

    def __init__(
            self,
            evolution_system: NetworkEvolution,
            movement_rewards: Dict,
            debug: bool = False
    ):
        self.evolution_system = evolution_system
        self.debug = debug

        # Core movement parameters
        self.max_path_length = movement_rewards.get('max_path_length', 200)
        self.pickup_reward = movement_rewards.get('pickup_reward', 1.0)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 2.0)
        self.distance_penalty = movement_rewards.get('distance_penalty', 2.0)

        # Dynamic reward parameters
        self.base_step_reward = movement_rewards.get('base_step_reward', 1.0)
        self.current_step_reward = self.base_step_reward
        self.health_change_reward_factor = movement_rewards.get('health_change_reward_factor', 2000.0)
        self.stagnation_threshold = movement_rewards.get('stagnation_threshold', 5)
        self.stagnation_penalty_rate = movement_rewards.get('stagnation_penalty_rate', 0.5)
        self.health_delta_threshold = movement_rewards.get('health_delta_threshold', 0.1)

        # State tracking
        self.last_health = None
        self.stagnation_counter = 0
        self.evaluation_count = 0

    def compute_fitness(self, path: List[str]) -> float:
        """
        Compute fitness score for a given path, incorporating health changes and stagnation effects.

        Args:
            path: List of movement commands to execute

        Returns:
            float: Combined fitness score
        """
        try:
            # Execute movement sequence and get results
            results = self.evolution_system.execute_movement_sequence(path)

            # Get current network health
            health_metrics = results['network_health']
            current_health = health_metrics['combined_health']

            # Calculate health delta and reward
            health_delta_reward = 0
            if self.last_health is not None:
                health_delta = current_health - self.last_health
                health_delta_reward = health_delta * self.health_change_reward_factor

                # Reset or increment stagnation counter based on health change
                if abs(health_delta) > self.health_delta_threshold:
                    self.stagnation_counter = 0
                    self.current_step_reward = self.base_step_reward
                else:
                    self.stagnation_counter += 1

                    # Apply increasing penalty if stagnant
                    if self.stagnation_counter > self.stagnation_threshold:
                        excess_stagnation = self.stagnation_counter - self.stagnation_threshold
                        penalty_factor = 1.0 - (excess_stagnation * self.stagnation_penalty_rate)
                        self.current_step_reward *=penalty_factor

            self.last_health = current_health

            # Calculate path execution score with dynamic step reward
            path_score = self.evaluate_path_execution(results)

            # Final score combines path score and health delta reward
            final_fitness = path_score + health_delta_reward

            if self.debug:
                self._print_debug_info(path_score, health_metrics, health_delta_reward, final_fitness)

            self.evaluation_count += 1
            return final_fitness

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return -100.0

    def evaluate_path_execution(self, results: Dict) -> float:
        """
        Evaluate the execution results of a movement sequence.

        Args:
            results: Dictionary containing execution metrics

        Returns:
            float: Path execution score
        """
        score = 0.0

        # Reward each step with the current (possibly degraded) step reward
        moves_made = results.get('moves_made', 0)
        score += moves_made * self.current_step_reward

        # Core rewards for successful actions
        score += results['pickups_made'] * self.pickup_reward
        score += results['successful_drops'] * self.successful_drop_reward

        # Apply penalty for excess moves
        if moves_made > self.max_path_length:
            excess_moves = moves_made - self.max_path_length
            score -= excess_moves * self.distance_penalty

        return score

    def _print_debug_info(self, path_score: float, health_metrics: Dict,
                          health_delta_reward: float, final_fitness: float):
        """Print detailed debug information about fitness calculation."""
        print("\nFitness Calculation Details:")
        print(f"Path Score: {path_score:.2f}")
        print(f"Health Delta Reward: {health_delta_reward:.2f}")
        print(f"Current Step Reward: {self.current_step_reward:.3f}")
        print(f"Stagnation Counter: {self.stagnation_counter}")
        print(f"Network Health: {health_metrics['combined_health']:.1f}%")
        print(f"Final Fitness: {final_fitness:.2f}")

    def get_stats(self) -> Dict:
        """Get current evaluator statistics."""
        return {
            'stagnation_counter': self.stagnation_counter,
            'current_step_reward': self.current_step_reward,
            'evaluation_count': self.evaluation_count,
            'last_health': self.last_health
        }