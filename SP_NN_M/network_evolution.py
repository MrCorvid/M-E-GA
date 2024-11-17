# network_evolution.py

from typing import List, Dict, Set
from SP_NN import Position, SpatialNeuralNetwork, NetworkParameters, NeuronType
from spatial_navigation import NavigationSystem
from network_monitor import NetworkMonitor
import threading
import time


class NetworkEvolution:
    def __init__(
            self,
            network: SpatialNeuralNetwork,
            navigator: NavigationSystem,
            monitor: NetworkMonitor,
            params: NetworkParameters
    ):
        self.network = network
        self.navigator = navigator
        self.monitor = monitor
        self.params = params

        # Evolution state
        self.pickup_bag = []
        self.processed_neurons = set()

        # Start background updates if monitor exists
        if self.monitor:
            self._start_background_updates()

    def execute_movement_sequence(self, path: List[str]) -> Dict:
        """Execute a sequence of movements and return results"""
        # Check if visualization should be enabled
        should_visualize = self.monitor and self.monitor.visualization_enabled

        # Only clear drops if visualization is enabled
        if should_visualize:
            self.monitor.clear_drops()

        # Reset state
        self.pickup_bag = []
        current_pos = Position(0.0, 0.0, 0.0)  # Start at origin
        self.navigator.current_pos = current_pos

        # Only reset path visualization if enabled
        if should_visualize:
            self.monitor.path_steps = []
            self.monitor.update_path_step(current_pos)

        # Get current network state
        state = self.network.get_network_state()
        neuron_positions = state['neuron_positions']
        position_updates = {}

        # Tracking metrics
        moves_made = 0
        pickups_made = 0
        successful_drops = 0
        failed_drops = 0
        neurons_moved = 0
        total_steps = 0

        for command in path:
            if command == 'DR' and self.pickup_bag:
                total_steps += 1
                current_pos = self.navigator.get_current_position()
                current_pos_tuple = (
                    round(current_pos.x),
                    round(current_pos.y),
                    round(current_pos.z)
                )

                # Check position availability
                is_position_empty = not any(
                    tuple(round(x) for x in pos['position']) == current_pos_tuple
                    for nid, pos in neuron_positions.items()
                    if nid not in self.pickup_bag and nid not in position_updates
                )

                if is_position_empty:
                    # Successful drop
                    neuron_id = self.pickup_bag.pop(0)
                    position_updates[neuron_id] = current_pos_tuple
                    neurons_moved += 1
                    successful_drops += 1

                    # Only update visualization if enabled
                    if should_visualize:
                        self.monitor.update_drop_locations([current_pos])

                else:
                    # Failed drop
                    failed_drops += 1

            else:
                # Execute movement
                new_pos = self.navigator.execute_movement(command)
                if new_pos:
                    total_steps += 1
                    moves_made += 1
                    current_pos = new_pos

                    # Check for pickups
                    current_pos_tuple = (
                        round(current_pos.x),
                        round(current_pos.y),
                        round(current_pos.z)
                    )

                    for nid, data in neuron_positions.items():
                        neuron = self.network.neurons[nid]
                        if (
                                neuron.type == NeuronType.HIDDEN and
                                nid not in self.pickup_bag and
                                nid not in position_updates
                        ):
                            neuron_pos = tuple(round(x) for x in data['position'])
                            if neuron_pos == current_pos_tuple:
                                self.pickup_bag.append(nid)
                                pickups_made += 1
                                break

                    # Only update path visualization if enabled
                    if should_visualize:
                        self.monitor.update_path_step(current_pos)

            # Only sleep for visualization if enabled
            if should_visualize:
                time.sleep(0.05)  # Visualization delay

        # Apply position updates
        if position_updates:
            self.network.update_neuron_positions(position_updates)

        return {
            'moves_made': moves_made,
            'pickups_made': pickups_made,
            'successful_drops': successful_drops,
            'failed_drops': failed_drops,
            'neurons_moved': neurons_moved,
            'total_neurons': len(self.network.neurons),
            'pickup_bag_size': len(self.pickup_bag),
            'total_steps': total_steps
        }

    def evaluate_network_structure(self) -> Dict:
        """Calculate current evolution metrics"""
        return {
            'connectivity': self.calculate_connectivity(),
            'processed_neurons': len(self.processed_neurons),
            'total_neurons': len(self.network.neurons),
            'pickup_bag_size': len(self.pickup_bag)
        }

    def calculate_connectivity(self) -> float:
        """Calculate current network connectivity"""
        return self.network.compute_connectivity_score()

    def update_monitor_connectivity(self, value: float):
        """Update monitor with current connectivity"""
        if self.monitor:
            self.monitor.update_connectivity(value)

    def get_network_stats(self) -> Dict:
        """Get current network statistics"""
        return self.network.get_network_state()

    def get_current_state(self) -> Dict:
        """Get current evolution state"""
        return {
            'pickup_bag_size': len(self.pickup_bag),
            'processed_neurons': len(self.processed_neurons),
            'current_position': self.navigator.get_current_position()
        }

    def _start_background_updates(self, update_interval: float = 1.0):
        """Start background thread for monitor updates"""

        def update_loop():
            while True:
                try:
                    # Only update visualization if enabled
                    if self.monitor and self.monitor.visualization_enabled:
                        current_state = self.network.get_network_state()
                        self.monitor.update_neuron_positions(current_state['neuron_positions'])
                        connectivity = self.calculate_connectivity()
                        self.monitor.update_connectivity(connectivity)
                except Exception as e:
                    print(f"Error in background update loop: {e}")
                time.sleep(update_interval)

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()