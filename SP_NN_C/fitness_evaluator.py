from typing import List, Dict
from network_evolution import NetworkEvolution


class FitnessEvaluator:
    """
    Evaluates fitness for spatial neural network paths with dynamic rewards and stagnation detection.
    Rewards health improvements and scales movement rewards based on navigation scale.
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
        self.pickup_reward = movement_rewards.get('pickup_reward', 20.0)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 22.0)
        self.distance_penalty = movement_rewards.get('distance_penalty', 25.0)

        # Scale-based parameters
        self.min_scale_threshold = movement_rewards.get('min_scale_threshold', 0.5)  # Scale below this gets penalized
        self.scale_penalty_factor = movement_rewards.get('scale_penalty_factor', 0.2)  # How much to reduce rewards

        # Dynamic reward parameters
        self.base_step_reward = movement_rewards.get('base_step_reward', 0.8)
        self.health_change_reward_factor = movement_rewards.get('health_change_reward_factor', 100.0)
        self.stagnation_threshold = movement_rewards.get('stagnation_threshold', 50)
        self.stagnation_penalty_rate = movement_rewards.get('stagnation_penalty_rate', 0.015)
        self.health_delta_threshold = movement_rewards.get('health_delta_threshold', 1.0)

        # State tracking
        self.last_health = None
        self.stagnation_counter = 0
        self.evaluation_count = 0
        self.current_reward_scale = 2.0

    def compute_fitness(self, path: List[str]) -> float:
        try:
            results = self.evolution_system.execute_movement_sequence(path)

            # Get current network health
            health_metrics = results['network_health']
            current_health = health_metrics['combined_health']

            # Calculate health delta and reward
            health_delta_reward = 0
            if self.last_health is not None:
                health_delta = current_health - self.last_health
                if health_delta > 0:
                    health_delta_reward = health_delta * self.health_change_reward_factor

                if abs(health_delta) > self.health_delta_threshold:
                    self.stagnation_counter = 0
                    self.current_reward_scale = 1.0
                else:
                    self.stagnation_counter += 1
                    if self.stagnation_counter > self.stagnation_threshold:
                        excess_stagnation = self.stagnation_counter - self.stagnation_threshold
                        self.current_reward_scale = 1.0 / (1.0 + (excess_stagnation * self.stagnation_penalty_rate))

            self.last_health = current_health

            # Calculate base path score with scale penalties
            path_score = self.evaluate_path_execution(results)

            # Combine scores
            final_fitness = path_score
            if health_delta_reward > 0:
                final_fitness += health_delta_reward * self.current_reward_scale

            if self.debug:
                self._print_debug_info(path_score, health_metrics, health_delta_reward,
                                       final_fitness, self.current_reward_scale)

            self.evaluation_count += 1
            return final_fitness

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return -100.0

    def evaluate_path_execution(self, results: Dict) -> float:
        score = 0.0
        moves_made = results.get('moves_made', 0)

        # Process command history to calculate scale-adjusted rewards
        for command, scale in results.get('command_history', []):
            # Only apply scale penalty to movement commands
            if command in ['U', 'D', 'F', 'B', 'L', 'R']:
                # Calculate scale penalty if below threshold
                if scale < self.min_scale_threshold:
                    scale_multiplier = self.scale_penalty_factor + (scale / self.min_scale_threshold) * (
                                1 - self.scale_penalty_factor)
                else:
                    scale_multiplier = 1.0

                score += self.base_step_reward * scale_multiplier

        # Add pickup and drop rewards (these aren't affected by scale)
        score += results['pickups_made'] * self.pickup_reward
        score += results['successful_drops'] * self.successful_drop_reward

        # Apply penalty for excess moves
        if moves_made > self.max_path_length:
            excess_moves = moves_made - self.max_path_length
            penalty = excess_moves * self.distance_penalty
            score -= penalty

        return score

    def _print_debug_info(self, path_score: float, health_metrics: Dict,
                          health_delta_reward: float, final_fitness: float,
                          reward_scale: float):
        print("\nFitness Calculation Details:")
        print(f"Base Path Score: {path_score:.2f}")
        print(f"Health Delta Reward: {health_delta_reward:.2f}")
        print(f"Stagnation Counter: {self.stagnation_counter}")
        print(f"Reward Scale: {reward_scale:.3f}")
        print(f"Network Health: {health_metrics['combined_health']:.1f}%")
        print(f"Final Fitness: {final_fitness:.2f}")

    def get_stats(self) -> Dict:
        return {
            'stagnation_counter': self.stagnation_counter,
            'reward_scale': self.current_reward_scale,
            'evaluation_count': self.evaluation_count,
            'last_health': self.last_health
        }