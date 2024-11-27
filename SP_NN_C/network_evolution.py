from typing import List, Dict, Set
from SP_NN import Position, SpatialNeuralNetwork, NetworkParameters, NeuronType
from spatial_navigation import NavigationSystem
from network_monitor import NetworkMonitor
import threading
import time
from collections import deque
import numpy as np
from scipy.spatial import KDTree


class NetworkEvolution:
    AGENT_RADIUS = 1.0  # Interaction radius for agent
    MAX_PICKUP_BAG = 500  # Maximum neurons we can carry at once

    def __init__(
            self,
            network: SpatialNeuralNetwork,
            navigator: NavigationSystem,
            monitor: NetworkMonitor,
            params: NetworkParameters
    ):
        """
        Initialize NetworkEvolution with required components.

        Args:
            network: SpatialNeuralNetwork instance
            navigator: NavigationSystem instance
            monitor: NetworkMonitor instance
            params: NetworkParameters instance
        """
        self.network = network
        self.navigator = navigator
        self.monitor = monitor
        self.params = params

        # Evolution state
        self.pickup_bag = deque(maxlen=self.MAX_PICKUP_BAG)  # FIFO queue for pickups
        self.processed_neurons = set()
        self.last_command_time = 0
        self._last_health_metrics = None

        # Start background updates if monitor exists
        if self.monitor:
            self._start_background_updates()

    def execute_movement_sequence(self, path: List[int]) -> Dict:
        """Execute a sequence of movements encoded in the number list"""
        try:
            # Initialize visualization if enabled
            should_visualize = self.monitor and self.monitor.visualization_enabled
            if should_visualize:
                self.monitor.clear_drops()
                self.monitor.clear_path()
                self.monitor.update_path_step(self.navigator.current_pos)

            # Reset state
            self.pickup_bag.clear()
            self.navigator.reset_position()

            # Get current network state
            state = self.network.get_network_state()
            neuron_positions = state['neuron_positions']
            position_updates = {}

            # Execute movement sequence
            nav_results = self.navigator.execute_movement_sequence(path)
            positions = nav_results['positions']
            command_history = nav_results['command_history']
            command_positions = nav_results['command_positions']

            # Create initial KD-tree for efficient neighbor searching
            available_neurons = [
                (nid, data['position'])
                for nid, data in neuron_positions.items()
                if (nid not in self.pickup_bag and
                    nid not in position_updates and
                    self.network.neurons[nid].type == NeuronType.HIDDEN)
            ]

            if available_neurons:
                neuron_ids, positions_array = zip(*available_neurons)
                neuron_pos_array = np.array([
                    [p[0], p[1], p[2]] for p in positions_array
                ])
                tree = KDTree(neuron_pos_array)

            # Process each command position
            for cmd_idx, (command, magnitude) in enumerate(command_history):
                pos_idx = command_positions[cmd_idx]
                current_pos = positions[pos_idx]

                # Check for pickups at every position
                if available_neurons and len(self.pickup_bag) < self.MAX_PICKUP_BAG:
                    # Query for nearby neurons
                    nearby_indices = tree.query_ball_point(
                        [current_pos.x, current_pos.y, current_pos.z],
                        self.AGENT_RADIUS
                    )

                    for idx in nearby_indices:
                        if len(self.pickup_bag) >= self.MAX_PICKUP_BAG:
                            break

                        nid = neuron_ids[idx]
                        if nid not in self.pickup_bag and nid not in position_updates:
                            self.pickup_bag.append(nid)

                    # Update available neurons if any were picked up
                    if nearby_indices:
                        # Create mask for remaining neurons
                        mask = np.ones(len(neuron_ids), dtype=bool)
                        for idx in nearby_indices:
                            mask[idx] = False

                        if any(mask):
                            neuron_pos_array = neuron_pos_array[mask]
                            neuron_ids = [nid for i, nid in enumerate(neuron_ids) if mask[i]]
                            tree = KDTree(neuron_pos_array)
                        else:
                            available_neurons = []

                # Handle DROP command
                if command == self.navigator.DROP and self.pickup_bag:
                    heading = self.navigator.heading
                    drop_pos = Position(
                        x=current_pos.x - heading[0] * self.AGENT_RADIUS,
                        y=current_pos.y - heading[1] * self.AGENT_RADIUS,
                        z=current_pos.z - heading[2] * self.AGENT_RADIUS
                    )

                    # Check if drop position is clear
                    position_clear = True
                    if available_neurons:
                        nearby = tree.query_ball_point(
                            [drop_pos.x, drop_pos.y, drop_pos.z],
                            self.AGENT_RADIUS
                        )
                        position_clear = len(nearby) == 0

                    if position_clear:
                        neuron_id = self.pickup_bag.popleft()  # FIFO order
                        position_updates[neuron_id] = (drop_pos.x, drop_pos.y, drop_pos.z)

                        if should_visualize:
                            self.monitor.update_drop_locations([drop_pos])

                # Update visualization
                if should_visualize:
                    self.monitor.update_path_step(current_pos)
                    time.sleep(0.05)

            # Apply position updates and update network health
            if position_updates:
                self.network.update_neuron_positions(position_updates)
                self._update_network_health()

            # Clear visualization after execution
            if should_visualize:
                time.sleep(0.1)
                self.monitor.clear_drops()

            # Return comprehensive results
            return {
                'moves_made': nav_results['moves_made'],
                'pickups_made': len(nav_results['move_positions']),
                'successful_drops': nav_results['drops_made'],
                'failed_drops': 0,  # Maintained for compatibility
                'neurons_moved': len(position_updates),
                'rotations_made': nav_results['rotations_made'],
                'total_neurons': len(self.network.neurons),
                'pickup_bag_size': len(self.pickup_bag),
                'total_steps': len(positions),
                'commands_executed': len(command_history),
                'positions': positions,
                'path_length': nav_results['path_length'],  # Get the actual distance from navigator
                'network_health': self._last_health_metrics
            }

        except Exception as e:
            import traceback
            traceback.print_exc()

            if should_visualize:
                self.monitor.clear_drops()
                self.monitor.clear_path()

            return {
                'moves_made': 0,
                'pickups_made': 0,
                'successful_drops': 0,
                'failed_drops': 0,
                'neurons_moved': 0,
                'rotations_made': 0,
                'total_neurons': len(self.network.neurons),
                'pickup_bag_size': len(self.pickup_bag),
                'total_steps': 0,
                'commands_executed': 0,
                'positions': [self.navigator.current_pos],
                'path_length': 0.0,
                'network_health': self._get_network_health()
            }

    def calculate_connectivity(self) -> float:
        """
        Legacy method renamed to match new health metrics system.
        Returns structural connectivity score.
        """
        return self.get_network_stats()['health_metrics']['structural_connectivity']

    def get_network_stats(self) -> Dict:
        """Get current network statistics"""
        state = self.network.get_network_state()
        health_metrics = self.network.compute_network_health()
        state.update({
            'health_metrics': health_metrics,
            'pickup_bag_size': len(self.pickup_bag)
        })
        return state

    def _get_network_health(self) -> dict:
        return self.network.compute_network_health()

    def _update_network_health(self):
        """Update network health metrics and monitor"""
        self._last_health_metrics = self._get_network_health()
        if self.monitor:
            self.monitor.update_health_metrics(self._last_health_metrics)

    def get_current_state(self) -> Dict:
        """Get current evolution state"""
        health_metrics = self._get_network_health()
        return {
            'pickup_bag_size': len(self.pickup_bag),
            'processed_neurons': len(self.processed_neurons),
            'current_position': self.navigator.get_current_position(),
            'health_metrics': health_metrics
        }

    def _start_background_updates(self, update_interval: float = 1.0):
        """Start background thread for monitor updates"""
        def update_loop():
            while True:
                try:
                    if self.monitor and self.monitor.visualization_enabled:
                        current_state = self.network.get_network_state()
                        self.monitor.update_neuron_positions(current_state['neuron_positions'])
                        self._update_network_health()
                except Exception as e:
                    print(f"Error in background update loop: {e}")
                time.sleep(update_interval)

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
