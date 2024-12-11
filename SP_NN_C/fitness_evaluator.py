from typing import List, Dict
from network_evolution import NetworkEvolution


class FitnessEvaluator:
    def __init__(
            self,
            evolution_system: NetworkEvolution,
            movement_rewards: Dict,
            debug: bool = False
    ):
        self.evolution_system = evolution_system
        self.debug = debug

        # Core parameters
        self.step_reward = movement_rewards.get('step_reward', 1.0)
        self.step_penalty = movement_rewards.get('step_penalty', 500)
        self.max_steps = movement_rewards.get('max_path_length', 60)
        self.pickup_reward = movement_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 5.0)
        self.scale_reduction_factor = movement_rewards.get('scale_reduction_factor', 0.30)

        # State tracking
        self.last_health = None
        self.stagnation_counter = 0
        self.evaluation_count = 0
        self.current_reward_scale = 2.0
        self.picked_neurons = set()

    def evaluate_path_execution(self, results: Dict) -> float:
        score = 0.0
        moves_made = results.get('moves_made', 0)
        command_history = results.get('command_history', [])

        if moves_made <= self.max_steps:
            # Calculate rewards only if within length limit
            movement_rewards = 0.0
            for _, scale in command_history:
                if scale < 1.0:
                    scale_steps = -int(np.log2(scale))
                    scale_multiplier = (1.0 - self.scale_reduction_factor) ** scale_steps
                    movement_rewards += self.step_reward * scale_multiplier
                else:
                    movement_rewards += self.step_reward

            score += movement_rewards

            # Add pickup and drop rewards only if within length limit
            score += results['pickups_made'] * self.pickup_reward
            score += results['successful_drops'] * self.successful_drop_reward
        else:
            # If over length limit, only apply penalties
            excess_steps = moves_made - self.max_steps
            score -= excess_steps * self.step_penalty

        if self.debug:
            print(f"\nFitness Calculation:")
            print(f"Moves Made: {moves_made}")
            print(f"Max Steps: {self.max_steps}")
            if moves_made <= self.max_steps:
                print(f"Movement Rewards: {movement_rewards:.2f}")
                print(f"Pickups: {results['pickups_made']} * {self.pickup_reward}")
                print(f"Drops: {results['successful_drops']} * {self.successful_drop_reward}")
            else:
                print("Over length limit - no rewards given")
                print(f"Step Penalties: {-excess_steps * self.step_penalty:.2f}")
            print(f"Final Score: {score:.2f}")

        return score

    def compute_fitness(self, path: List[str]) -> float:
        try:
            # Filter already picked neurons
            for neuron_id in list(self.evolution_system.pickup_bag):
                if neuron_id in self.picked_neurons:
                    self.evolution_system.pickup_bag.remove(neuron_id)

            # Execute path and get results
            results = self.evolution_system.execute_movement_sequence(path)

            # Update picked neurons tracking
            for neuron_id in results.get('pickup_bag', []):
                self.picked_neurons.add(neuron_id)

            # Calculate base path score
            path_score = self.evaluate_path_execution(results)

            # Get network health impact
            health_metrics = results['network_health']
            current_health = health_metrics['combined_health']

            # Calculate health delta reward
            health_delta_reward = 0
            if self.last_health is not None:
                health_delta = current_health - self.last_health
                if health_delta > 0:
                    health_delta_reward = health_delta * 20.0
                    self.stagnation_counter = 0
                    self.current_reward_scale = 1.0
                else:
                    self.stagnation_counter += 1

            # Combine scores
            final_fitness = path_score
            if health_delta_reward > 0:
                final_fitness += health_delta_reward * self.current_reward_scale

            # Update state
            self.evaluation_count += 1
            self.last_health = current_health

            return final_fitness

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return -100.0

    def reset_generation(self):
        """Reset picked neurons for new generation"""
        self.picked_neurons.clear()