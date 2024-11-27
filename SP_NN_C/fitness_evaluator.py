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

        self.max_path_length = movement_rewards.get('max_path_length', 60)
        self.rotation_reward = movement_rewards.get('rotation_reward', 0.5)
        self.pickup_reward = movement_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 5.0)
        self.distance_penalty = movement_rewards.get('distance_penalty', 0.1)
        self.proximity_factor = movement_rewards.get('proximity_factor', 1.2)

    def compute_fitness(self, path: List[int]) -> float:
        try:
            initial_health = self.evolution_system.get_network_stats()['health_metrics']
            initial_combined_health = initial_health['combined_health']
            initial_proximity = initial_health['proximity_score']

            results = self.evolution_system.execute_movement_sequence(path)
            path_score = self.evaluate_path_execution(results)

            final_health = results['network_health']
            final_combined_health = final_health['combined_health']
            final_proximity = final_health['proximity_score']

            health_delta = final_combined_health - initial_combined_health
            proximity_delta = final_proximity - initial_proximity

            normalized_health_delta = health_delta / 100.0
            normalized_proximity_delta = proximity_delta / 100.0

            combined_delta = (0.8 * normalized_health_delta +
                              0.2 * normalized_proximity_delta)

            if combined_delta > 0:
                if path_score < 0:
                    final_fitness = path_score / (1 + combined_delta)
                else:
                    final_fitness = path_score * (1 + combined_delta)
            else:
                if path_score < 0:
                    final_fitness = path_score * (1 + abs(combined_delta))
                else:
                    final_fitness = path_score * (1 + combined_delta)

            if self.debug:
                self._print_debug_info(path_score, initial_health, final_health,
                                       health_delta, proximity_delta, final_fitness)

            final_fitness *= (final_combined_health * 3)

            return final_fitness

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return -100.0

    def evaluate_path_execution(self, results: Dict) -> float:
        score = 0.0

        if 'proximity_metrics' in results:
            proximity = results['proximity_metrics']
            avg_near = proximity.get('avg_near_connections', 0)
            score += avg_near * self.proximity_factor

        score += results['rotations_made'] * self.rotation_reward
        score += results['pickups_made'] * self.pickup_reward
        score += results['successful_drops'] * self.successful_drop_reward

        total_distance = results.get('path_length', 0)
        if total_distance <= self.max_path_length:
            score += total_distance
        else:
            score += self.max_path_length
            distance_over = total_distance - self.max_path_length
            penalty = distance_over * self.distance_penalty
            score -= penalty

        return score

    def _print_debug_info(self, path_score: float, initial_health: Dict,
                          final_health: Dict, health_delta: float,
                          proximity_delta: float, final_fitness: float):
        print("\nFitness Calculation Breakdown:")
        print(f"Path Score: {path_score}")
        print("\nNetwork Health Changes:")
        print(f"Initial Combined Health: {initial_health['combined_health']:.1f}")
        print(f"Initial Proximity Score: {initial_health['proximity_score']:.1f}")
        print(f"Final Combined Health: {final_health['combined_health']:.1f}")
        print(f"Final Proximity Score: {final_health['proximity_score']:.1f}")
        print(f"\nHealth Delta: {health_delta:.1f}")
        print(f"Proximity Delta: {proximity_delta:.1f}")
        print(f"Final Fitness: {final_fitness:.3f}")

    def get_stats(self) -> Dict:
        if hasattr(self.evolution_system, 'get_current_state'):
            return self.evolution_system.get_current_state()
        return {}