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
        self.max_path_length = movement_rewards.get('max_path_length', 200)
        self.pickup_reward = movement_rewards.get('pickup_reward', 3.0)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 6.0)
        self.distance_penalty = movement_rewards.get('distance_penalty', 20.0)
        self.proximity_factor = movement_rewards.get('proximity_factor', 1.5)

    def compute_fitness(self, path: List[str]) -> float:
        try:
            # Execute movement sequence and get results
            results = self.evolution_system.execute_movement_sequence(path)
            path_score = self.evaluate_path_execution(results)

            # Get network health metrics
            health_metrics = results['network_health']
            combined_health = health_metrics['combined_health']

            # Final fitness is path score scaled by network health
            final_fitness = path_score * (combined_health * 3)

            if self.debug:
                self._print_debug_info(path_score, health_metrics, final_fitness)

            return final_fitness

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return -100.0

    def evaluate_path_execution(self, results: Dict) -> float:
        """Evaluate the execution results of a movement sequence."""
        score = 0.0

        # Reward proximity-based interactions
        if 'proximity_metrics' in results:
            proximity = results['proximity_metrics']
            avg_near = proximity.get('avg_near_connections', 0)
            score += avg_near * self.proximity_factor

        # Core rewards for successful actions
        score += results['pickups_made'] * self.pickup_reward
        score += results['successful_drops'] * self.successful_drop_reward

        # Movement efficiency
        moves_made = results.get('moves_made', 0)
        if moves_made <= self.max_path_length:
            score += moves_made
        else:
            score += self.max_path_length
            score -= (moves_made - self.max_path_length) * self.distance_penalty

        return score

    def _print_debug_info(self, path_score: float, health_metrics: Dict, final_fitness: float):
        if not self.debug:
            return

        print("\nNetwork Health Stats:")
        print(f"Structural: {health_metrics['structural_connectivity']:.1f}%")
        print(f"Density: {health_metrics['connection_density']:.1f}%")
        print(f"Health: {health_metrics['combined_health']:.1f}%")
        print(f"Final Score: {final_fitness:.2f}\n")

    def get_stats(self) -> Dict:
        return self.evolution_system.get_current_state()