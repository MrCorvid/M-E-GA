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

        # Core reward parameters
        self.max_path_length = movement_rewards.get('max_path_length', 60)
        self.rotation_reward = movement_rewards.get('rotation_reward', 0.5)
        self.pickup_reward = movement_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 5.0)
        self.distance_penalty = movement_rewards.get('distance_penalty', 0.1)

    def compute_fitness(self, path: List[int]) -> float:
        """
        Compute fitness based on path execution and network health impact.
        The path score is modified based on its raw impact on network health.
        """
        try:
            # Get initial network health
            initial_health = self.evolution_system.get_network_stats()['health_metrics']
            initial_combined_health = initial_health['combined_health']
    
            # Execute path and get results
            results = self.evolution_system.execute_movement_sequence(path)
    
            # Calculate path execution score
            path_score = self.evaluate_path_execution(results)
    
            # Get final network health
            final_health = results['network_health']
            final_combined_health = final_health['combined_health']
    
            # Calculate raw health delta and normalize
            health_delta = final_combined_health - initial_combined_health
            normalized_health_delta = health_delta / 100.0  # Assuming health ranges from 0 to 100
    
            # Apply score modification based on health impact
            if health_delta > 0:
                if path_score < 0:
                    # Bad path with positive health impact - reduce penalty
                    final_fitness = path_score / (1 + normalized_health_delta)
                else:
                    # Good path with positive health impact - amplify reward
                    final_fitness = path_score * (1 + normalized_health_delta)
            elif health_delta < 0:
                if path_score < 0:
                    # Bad path with negative health impact - amplify penalty
                    final_fitness = path_score * (1 + abs(normalized_health_delta))
                else:
                    # Good path with negative health impact - reduce reward
                    final_fitness = path_score * (1 + normalized_health_delta)
            else:
                # No change in health
                final_fitness = path_score
    
            if self.debug:
                self._print_debug_info(path_score, initial_health, final_health,
                                       health_delta, final_fitness)
    
            # Multiply by final_combined_health (keeping your existing scaling)
            final_fitness *= (final_combined_health * 2)
    
            return final_fitness
    
        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return -100.0

    def evaluate_path_execution(self, results: Dict) -> float:
        """
        Evaluate path execution with distance rewards and penalties.
        - Rewards all movement up to max_path_length
        - Penalizes each unit over max_path_length
        """
        score = 0.0

        # Add base rewards
        score += results['rotations_made'] * self.rotation_reward
        score += results['pickups_made'] * self.pickup_reward
        score += results['successful_drops'] * self.successful_drop_reward

        # Handle distance reward and penalties
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
                          final_health: Dict, health_delta: float, final_fitness: float):
        """Print detailed debug information about fitness calculation"""
        print("\nFitness Calculation Breakdown:")
        print(f"Path Score: {path_score}")

        print("\nNetwork Health Changes:")
        print("  Initial:")
        print(f"    Combined Health: {initial_health['combined_health']:.1f}")

        print("  Final:")
        print(f"    Combined Health: {final_health['combined_health']:.1f}")

        print(f"\nRaw Health Delta: {health_delta:.1f}")
        print(f"Final Fitness: {final_fitness:.3f}")

    def get_stats(self) -> Dict:
        """Get latest execution statistics"""
        if hasattr(self.evolution_system, 'get_current_state'):
            return self.evolution_system.get_current_state()
        return {}
