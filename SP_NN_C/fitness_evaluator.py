from typing import List, Dict
from network_evolution import NetworkEvolution
import numpy as np


class FitnessEvaluator:
    def __init__(
            self,
            evolution_system: NetworkEvolution,
            movement_rewards: Dict,
            debug: bool = False
    ):
        """
        Initialize the fitness evaluator with continuous movement support.

        Args:
            evolution_system: NetworkEvolution instance
            movement_rewards: Dictionary of reward parameters
            debug: Enable debug output
        """
        self.evolution_system = evolution_system
        self.debug = debug

        # Movement reward parameters
        self.max_path_length = movement_rewards.get('max_path_length', 60.0)
        self.path_step_reward = movement_rewards.get('path_step_reward', 1.00)
        self.pickup_reward = movement_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = movement_rewards.get('successful_drop_reward', 5.0)
        self.failed_drop_penalty = movement_rewards.get('failed_drop_penalty', -0.0)
        self.empty_bag_reward = movement_rewards.get('empty_bag_reward', 10.00)
        self.step_penalty = movement_rewards.get('step_penalty', -0.5)

        # Continuous movement parameters
        self.heading_change_cost = movement_rewards.get('heading_change_cost', 0.1)
        self.optimal_step_size = movement_rewards.get('optimal_step_size', 2.0)
        self.path_efficiency_weight = movement_rewards.get('path_efficiency_weight', 0.3)
        self.smoothness_weight = movement_rewards.get('smoothness_weight', 0.2)

        # Generation tracking
        self.current_generation = 0
        self._last_connectivity = 0

        # Movement analysis buffers
        self.command_history = []
        self.path_metrics = {}

    def compute_fitness(self, path: List[int]) -> float:
        """
        Compute fitness for a given path with continuous movement consideration.

        Args:
            path: List of integers that will be converted to binary commands

        Returns:
            float: Computed fitness score
        """
        try:
            # Execute path and get results
            movement_results = self.evolution_system.execute_movement_sequence(path)

            # Store command history for analysis
            self.command_history = movement_results.get('command_history', [])

            # Calculate path metrics
            self.path_metrics = self._calculate_path_metrics(movement_results)

            # Evaluate different aspects
            path_score = self.evaluate_path_execution(movement_results)
            structure_score = self.evaluate_network_structure()
            efficiency_score = self._calculate_efficiency_score(movement_results)
            smoothness_score = self._calculate_path_smoothness(movement_results)

            # Combine scores for final fitness
            fitness = self._calculate_final_fitness(
                path_score=path_score,
                structure_score=structure_score,
                efficiency_score=efficiency_score,
                smoothness_score=smoothness_score,
                results=movement_results
            )

            if self.debug:
                self._print_debug_info(
                    movement_results,
                    path_score,
                    structure_score,
                    efficiency_score,
                    smoothness_score,
                    fitness
                )

            return max(0.0, fitness)  # Ensure non-negative fitness

        except Exception as e:
            if self.debug:
                print(f"Error computing fitness: {e}")
            return 0.0

    def evaluate_path_execution(self, results: Dict) -> float:
        """
        Evaluate the execution of a path based on continuous movement results.
        """
        path_score = 0.0

        # Base movement rewards
        path_score += results['moves_made'] * self.path_step_reward
        path_score += results['pickups_made'] * self.pickup_reward
        path_score += results['successful_drops'] * self.successful_drop_reward
        path_score += results['failed_drops'] * self.failed_drop_penalty

        # Calculate continuous path length
        total_distance = self._calculate_path_length(results['positions'])

        # Apply distance-based penalties
        if total_distance > self.max_path_length:
            excess_distance = total_distance - self.max_path_length
            path_score += excess_distance * self.step_penalty

        # Reward empty pickup bag
        if results['pickup_bag_size'] == 0:
            path_score += self.empty_bag_reward

        # Analyze heading changes
        total_heading_change = self._calculate_total_heading_change()
        path_score -= total_heading_change * self.heading_change_cost

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

    def _calculate_path_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive path metrics for analysis."""
        positions = results['positions']

        # Calculate step distances
        step_distances = [
            positions[i - 1].distance_to(positions[i])
            for i in range(1, len(positions))
        ]

        # Calculate heading changes between consecutive movements
        heading_changes = []
        for i in range(2, len(positions)):
            prev_vector = np.array([
                positions[i - 1].x - positions[i - 2].x,
                positions[i - 1].y - positions[i - 2].y,
                positions[i - 1].z - positions[i - 2].z
            ])
            curr_vector = np.array([
                positions[i].x - positions[i - 1].x,
                positions[i].y - positions[i - 1].y,
                positions[i].z - positions[i - 1].z
            ])

            # Normalize vectors
            prev_norm = np.linalg.norm(prev_vector)
            curr_norm = np.linalg.norm(curr_vector)

            if prev_norm > 0 and curr_norm > 0:
                prev_vector = prev_vector / prev_norm
                curr_vector = curr_vector / curr_norm
                # Calculate angle between vectors
                cos_angle = np.clip(np.dot(prev_vector, curr_vector), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                heading_changes.append(angle)

        return {
            'step_distances': step_distances,
            'heading_changes': heading_changes,
            'total_distance': sum(step_distances),
            'average_step': np.mean(step_distances) if step_distances else 0,
            'total_heading_change': sum(heading_changes) if heading_changes else 0
        }

    def _calculate_path_length(self, positions: List) -> float:
        """Calculate total path length with continuous movement."""
        return sum(
            positions[i - 1].distance_to(positions[i])
            for i in range(1, len(positions))
        )

    def _calculate_total_heading_change(self) -> float:
        """Calculate total heading change from command history."""
        total_change = 0.0
        for cmd, magnitude in self.command_history:
            if cmd in [0b1111, 0b1110, 0b1100]:  # Heading changes
                total_change += abs(magnitude)
        return total_change

    def _calculate_efficiency_score(self, results: Dict) -> float:
        """Calculate movement efficiency score."""
        if not self.path_metrics:
            return 0.0

        # Calculate path efficiency metrics
        avg_step = self.path_metrics['average_step']
        optimal_ratio = avg_step / self.optimal_step_size
        efficiency = np.exp(-abs(optimal_ratio - 1.0))

        # Penalize excessive heading changes
        heading_penalty = np.exp(-self.path_metrics['total_heading_change'] / (2 * np.pi))

        return (efficiency + heading_penalty) / 2

    def _calculate_path_smoothness(self, results: Dict) -> float:
        """Calculate path smoothness score."""
        if not self.path_metrics.get('heading_changes'):
            return 0.0

        # Calculate variance in heading changes
        heading_changes = self.path_metrics['heading_changes']
        smoothness = np.exp(-np.var(heading_changes) if len(heading_changes) > 1 else 0)

        return smoothness

    def _calculate_final_fitness(self, path_score: float, structure_score: float,
                                 efficiency_score: float, smoothness_score: float,
                                 results: Dict) -> float:
        """Calculate final fitness combining all scores."""
        # Base fitness from path execution
        fitness = path_score

        # Scale based on connectivity achievement
        connectivity_factor = (structure_score * 0.01)  # Convert to 0-1 range
        fitness *= (1.0 + connectivity_factor)

        # Add efficiency and smoothness contributions
        fitness += (efficiency_score * self.path_efficiency_weight * fitness)
        fitness += (smoothness_score * self.smoothness_weight * fitness)

        # Apply final distance-based scaling
        total_distance = self.path_metrics.get('total_distance', 0)
        if total_distance > self.max_path_length:
            efficiency_penalty = 1.0 - (total_distance - self.max_path_length) / self.max_path_length
            efficiency_penalty = max(0.1, efficiency_penalty)
            fitness *= efficiency_penalty

        return fitness

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
                'step_penalty': self.step_penalty,
                'heading_change_cost': self.heading_change_cost
            },
            'path_metrics': self.path_metrics,
            'last_connectivity': self._last_connectivity
        }

    def _print_debug_info(self, results: Dict, path_score: float,
                          structure_score: float, efficiency_score: float,
                          smoothness_score: float, final_fitness: float):
        """Print detailed debug information about fitness calculation"""
        if not self.debug:
            return

        print("\nFitness Calculation Details:")
        print(f"Total Distance: {self.path_metrics['total_distance']:.2f} units")
        print(f"Average Step Size: {self.path_metrics['average_step']:.2f} units")
        print(f"Total Heading Change: {self.path_metrics['total_heading_change']:.2f} radians")
        print(f"Path Efficiency Score: {efficiency_score:.3f}")
        print(f"Path Smoothness Score: {smoothness_score:.3f}")
        print(f"Moves Made: {results['moves_made']}")
        print(f"Pickups: {results['pickups_made']}")
        print(f"Successful Drops: {results['successful_drops']}")
        print(f"Failed Drops: {results['failed_drops']}")
        print(f"Path Score: {path_score:.2f}")
        print(f"Network Connectivity: {structure_score:.1f}%")
        print(f"Final Fitness: {final_fitness:.2f}")