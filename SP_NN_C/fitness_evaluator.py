from typing import List, Dict, Optional, Callable
from SP_NN import create_network, SpatialNeuralNetwork, NetworkParameters, Position
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

        # Core reward parameters
        self.max_path_length = movement_rewards.get('max_path_length', 60)
        self.rotation_reward = movement_rewards.get('rotation_reward', 0.5)
        self.pickup_reward = movement_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 5.0)
        self.distance_penalty = movement_rewards.get('distance_penalty', 0.1)

        # For tracking
        self.current_generation = 0
        self._last_health_metrics = None

    def compute_network_fitness(self) -> float:
        """
        Computes and combines structural connectivity and density scores for fitness.
        Returns single composite score for use in final fitness calculation.
        """
        metrics = self.evolution_system.network.compute_fitness_metrics()

        # Get raw structural and density scores
        structural = metrics['raw_values']['structural']
        density = metrics['raw_values']['density']

        # Simple addition for now - easy to modify combination logic
        composite_score = structural + density

        if self.debug:
            print(f"\nNetwork Fitness Components:")
            print(f"  Structural Score: {structural:.2f}")
            print(f"  Density Score: {density:.2f}")
            print(f"  Composite Score: {composite_score:.2f}")

        return composite_score

    def compute_fitness(self, path: List[int]) -> float:
        try:
            # Execute path and get results
            results = self.evolution_system.execute_movement_sequence(path)

            # Calculate raw path score
            path_score = self.evaluate_path_execution(results)

            # Get composite network score
            network_score = self.compute_network_fitness()

            # Combine path and network scores
            final_score = path_score ** network_score

            if self.debug:
                self._print_debug_info(results, path_score, network_score, final_score)

            return max(0.1, final_score)

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return 0.1

    def evaluate_path_execution(self, results: Dict) -> float:
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
            print(excess_distance, ' Over distance.')

        return score

    def get_stats(self) -> Dict:
        return {
            'current_generation': self.current_generation,
            'rewards': {
                'rotation_reward': self.rotation_reward,
                'pickup_reward': self.pickup_reward,
                'drop_reward': self.successful_drop_reward,
                'distance_penalty': self.distance_penalty,
                'max_path_length': self.max_path_length
            },
            'last_health_metrics': self._last_health_metrics
        }

    def _print_debug_info(self, results: Dict, path_score: float,
                          network_score: float, final_fitness: float):
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

        print(f"\nFinal Scores:")
        print(f"  Path Score: {path_score:.2f}")
        print(f"  Network Score: {network_score:.2f}")
        print(f"  Final Fitness: {final_fitness:.2f}")