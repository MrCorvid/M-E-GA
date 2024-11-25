# SP_NN_Fitness.py

from typing import List, Dict, Optional, Callable
from SP_NN import (
    create_network,
    SpatialNeuralNetwork,
    NetworkParameters,
    Position,
    NeuronType
)
from spatial_navigation import NavigationSystem
from network_monitor import NetworkMonitor
from network_evolution import NetworkEvolution
from fitness_evaluator import FitnessEvaluator
import threading
import time
import sys
import queue
import numpy as np
from scipy.spatial import KDTree


class NetworkEvolutionFitness:
    def __init__(
            self,
            config: Dict[str, any],
            update_best_func: Optional[Callable[[any, float], None]] = None,
            debug: bool = False
    ):
        """
        Initialize Network Evolution Fitness focusing on spatial navigation and connectivity.

        Args:
            config (Dict[str, any]): Configuration dictionary containing all parameters.
            update_best_func (Callable, optional): Callback function to update the best organism.
            debug (bool, optional): Flag to enable debug mode.
        """
        # Extract network parameters and path rewards from config
        network_params = config.get('network_params', {})
        path_rewards = config.get('path_rewards', {})

        # Initialize NetworkParameters
        self.network_params = NetworkParameters(
            volume_size=network_params.get('volume_size', 8.0),
            num_input=network_params.get('num_input', 100),
            num_output=network_params.get('num_output', 4),
            total_neurons=network_params.get('total_neurons', 500),
            max_radius=network_params.get('max_radius', 3.0),
            min_radius=network_params.get('min_radius', 0.1),
            input_radius_factor=network_params.get('input_radius_factor', 0.25),
            interface_radius_factor=network_params.get('interface_radius_factor', 0.3),
            hidden_radius_range=network_params.get('hidden_radius_range', (0.10, 0.80)),
            interface_offset=network_params.get('interface_offset', 1.0),
            activation_budget=network_params.get('activation_budget', 1000),
            time_window_size=network_params.get('time_window_size', 100),
            base_radius_shrink_rate=network_params.get('base_radius_shrink_rate', 0.95),
            activation_radius_factor=network_params.get('activation_radius_factor', 0.2),
            activation_threshold=network_params.get('activation_threshold', 0.5)
        )

        # Initialize the spatial neural network
        self.network = create_network(self.network_params)

        # Initialize path reward parameters with simplified rewards structure
        self.max_path_length = path_rewards.get('max_path_length', 60)
        self.rotation_reward = path_rewards.get('rotation_reward', 0.5)
        self.pickup_reward = path_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = path_rewards.get('successful_drop_reward', 5.0)
        self.distance_penalty = path_rewards.get('distance_penalty', 0.1)

        # Debug and Update Function
        self.debug = debug
        self.update_best = update_best_func

        # Initialize modular components
        self.navigator = NavigationSystem(volume_size=self.network_params.volume_size)

        # Create monitor window first
        try:
            self.monitor = NetworkMonitor(volume_size=self.network_params.volume_size)
            # Give the window a moment to initialize
            time.sleep(0.5)
            # Initialize path visualization with starting position
            if self.monitor.visualization_enabled:
                self.monitor.update_path_step(Position(0.0, 0.0, 0.0))
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not create monitor window: {e}")
            self.monitor = None

        # Initialize evolution system
        self.evolution = NetworkEvolution(
            network=self.network,
            navigator=self.navigator,
            monitor=self.monitor,
            params=self.network_params
        )

        # Initialize fitness evaluator with simplified rewards
        self.evaluator = FitnessEvaluator(
            evolution_system=self.evolution,
            movement_rewards={
                'max_path_length': self.max_path_length,
                'rotation_reward': self.rotation_reward,
                'pickup_reward': self.pickup_reward,
                'successful_drop_reward': self.successful_drop_reward,
                'distance_penalty': self.distance_penalty
            },
            debug=debug
        )

        # Define genes for continuous navigation
        # Each gene is a single digit [0-9] that will be concatenated
        # to form the binary command stream
        self.genes = [str(i) for i in range(10)]  # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # Initialize last health metrics
        self._last_health_metrics = None

    def compute(self, encoded_individual, ga_instance) -> float:
        """
        Fitness computation with comprehensive network health evaluation.

        Args:
            encoded_individual: The encoded genome representing the sequence of digits.
            ga_instance: The genetic algorithm instance.

        Returns:
            float: The calculated fitness score.
        """
        # Convert encoded individual to list of integers
        if ga_instance:
            # Decode organism and convert string digits to integers
            path_str = ga_instance.decode_organism(encoded_individual)
            path = [int(digit) for digit in path_str if digit.isdigit()]
        else:
            # Direct conversion if no GA instance
            path = [int(digit) for digit in encoded_individual if str(digit).isdigit()]

        # Ensure path is not empty
        if not path:
            return 0.0

        # Compute fitness using evaluator
        fitness = self.evaluator.compute_fitness(path)

        # Get comprehensive health metrics
        health_metrics = self.evolution.get_network_stats()['health_metrics']

        # Store for debugging and monitoring
        self._last_health_metrics = health_metrics

        # Check if we've achieved optimal network health
        if health_metrics['combined_health'] >= 95:  # Using combined health threshold
            if self.debug:
                print("\nExiting Phase 1: Achieved optimal network health")
                print("\nFinal Health Metrics:")
                print(f"Structural Connectivity: {health_metrics['structural_connectivity']:.1f}%")
                print(f"Connection Density: {health_metrics['connection_density']:.1f}%")
                print(f"Combined Health: {health_metrics['combined_health']:.1f}%")
                print("\nNetwork Statistics:")
                stats = self.get_stats()
                print(f"Total Neurons: {stats['network']['total_neurons']}")
                print(f"Path Length: {len(path)}")
                print(f"Path: {''.join(map(str, path))}")

            # Exit the program since optimal health is achieved
            sys.exit(0)

        if self.debug:
            self._print_debug_info(path, fitness, health_metrics)

        if self.update_best:
            self.update_best(encoded_individual, fitness)

        return fitness

    def _print_debug_info(self, path: List[int], fitness: float, health_metrics: Dict):
        """Print comprehensive debug information about fitness and health metrics"""
        if not self.debug:
            return

        print("\nFitness Calculation Details:")
        print(f"Path Length: {len(path)}")
        print(f"Path: {''.join(map(str, path))}")
        print("\nNetwork Health Metrics:")
        print(f"Structural Connectivity: {health_metrics['structural_connectivity']:.1f}%")
        print(f"Connection Density: {health_metrics['connection_density']:.1f}%")
        print(f"Combined Health: {health_metrics['combined_health']:.1f}%")
        print(f"Final Fitness: {fitness:.2f}")

        # Print command analysis if available
        if hasattr(self.navigator, 'command_history') and self.navigator.command_history:
            print("\nCommand Execution:")
            command_names = {
                self.navigator.HEADING_X: "X-Heading",
                self.navigator.HEADING_Y: "Y-Heading",
                self.navigator.HEADING_Z: "Z-Heading",
                self.navigator.MOVE: "Move",
                self.navigator.DROP: "Drop"
            }
            for cmd, mag in self.navigator.command_history:
                cmd_name = command_names.get(cmd, "Unknown")
                print(f"{cmd_name}: {mag:.2f}")

        # Print detailed health metadata if available
        if 'metadata' in health_metrics:
            meta = health_metrics['metadata']
            print("\nHealth Calculation Details:")
            print(f"Density Margin: {meta['density_margin']}")
            print(f"Structural Weight: {meta['structural_weight']}")
            print(f"Density Weight: {meta['density_weight']}")
            print(f"Raw Density: {meta['raw_density']:.1f}%")
            print(f"Adjusted Density: {meta['adjusted_density']:.1f}%")

    def get_stats(self) -> Dict:
        """Get comprehensive statistics from all components"""
        network_stats = self.evolution.get_network_stats()
        return {
            'health_metrics': network_stats['health_metrics'],
            'path': self.evaluator.get_stats(),
            'network': network_stats,
            'current_state': self.evolution.get_current_state()
        }