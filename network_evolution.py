from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import threading
import time
import numpy as np

from spatial_neural_network import SpatialNeuralNetwork, NetworkParameters, Position, NeuronType
from navigation_system import NavigationSystem
from network_monitor import NetworkMonitor

class NetworkEvolution:
    def __init__(self, 
                 params: NetworkParameters,
                 monitor: Optional[NetworkMonitor] = None,
                 debug: bool = False):
        """
        Initialize the network evolution system.
        
        Args:
            params: Network parameters configuration
            monitor: Optional visualization monitor
            debug: Enable debug output
        """
        self.params = params
        self.debug = debug
        self.monitor = monitor
        
        # Initialize core components
        self.network = SpatialNeuralNetwork(params)
        self.navigator = NavigationSystem(volume_size=params.volume_size)
        
        # Evolution state
        self.pickup_bag: List[int] = []  # Currently held neurons
        self.processed_neurons: Set[int] = set()  # Neurons that have been moved
        
        # Start background updates if monitor is present
        if self.monitor:
            self._start_background_updates()

    def execute_movement_sequence(self, path: List[str]) -> Dict:
        """
        Execute a sequence of movement commands and return results.
        
        Args:
            path: List of movement commands ('U', 'D', 'F', 'B', 'L', 'R', 'DR')
            
        Returns:
            Dictionary containing execution results
        """
        if self.monitor:
            self.monitor.clear_drops()
        
        metrics = {
            'moves_made': 0,
            'pickups_made': 0,
            'successful_drops': 0,
            'failed_drops': 0,
            'neurons_moved': 0,
            'total_steps': 0
        }
        
        # Execute each movement command
        for command in path:
            if command == 'DR':
                # Handle drop command
                success = self._handle_drop_command(self.navigator.get_current_position())
                if success:
                    metrics['successful_drops'] += 1
                else:
                    metrics['failed_drops'] += 1
            else:
                # Execute movement
                new_pos = self.navigator.execute_movement(command)
                if new_pos:
                    metrics['moves_made'] += 1
                    if self.monitor:
                        self.monitor.update_path_step(new_pos)
                    
                    # Check for pickup opportunity
                    if self._handle_pickup_opportunity(new_pos):
                        metrics['pickups_made'] += 1
            
            metrics['total_steps'] += 1
            
        metrics['neurons_moved'] = len(self.processed_neurons)
        return metrics

    def _handle_pickup_opportunity(self, pos: Position) -> bool:
        """
        Check for and handle neuron pickup at current position.
        
        Args:
            pos: Current position to check
            
        Returns:
            True if pickup successful, False otherwise
        """
        # Get current network state
        state = self.network.get_network_state()
        current_pos = (round(pos.x), round(pos.y), round(pos.z))
        
        # Check for pickups at current position
        for nid, data in state['neuron_positions'].items():
            neuron = self.network.neurons[nid]
            if (neuron.type == NeuronType.HIDDEN and
                nid not in self.pickup_bag and
                nid not in self.processed_neurons):
                
                neuron_pos = tuple(round(x) for x in data['position'])
                if neuron_pos == current_pos:
                    self.pickup_bag.append(nid)
                    return True
        
        return False

    def _handle_drop_command(self, pos: Position) -> bool:
        """
        Handle neuron drop at current position.
        
        Args:
            pos: Position to attempt drop
            
        Returns:
            True if drop successful, False otherwise
        """
        if not self.pickup_bag:
            return False
            
        current_pos = (round(pos.x), round(pos.y), round(pos.z))
        state = self.network.get_network_state()
        
        # Check if position is occupied
        is_position_empty = not any(
            tuple(round(x) for x in pos_data['position']) == current_pos
            for nid, pos_data in state['neuron_positions'].items()
            if nid not in self.pickup_bag and nid not in self.processed_neurons
        )
        
        if is_position_empty:
            # Perform drop
            neuron_id = self.pickup_bag.pop(0)
            self.network.update_neuron_positions({
                neuron_id: current_pos
            })
            self.processed_neurons.add(neuron_id)
            
            # Update visualization
            if self.monitor:
                self.monitor.update_drop_locations([pos])
            
            return True
            
        return False

    def evaluate_network_structure(self) -> float:
        """
        Evaluate the current network structure.
        
        Returns:
            Network connectivity score (0-100)
        """
        connectivity = self.calculate_connectivity()
        
        if self.monitor:
            self.monitor.update_connectivity(connectivity)
            
        return connectivity

    def calculate_connectivity(self) -> float:
        """
        Calculate network connectivity percentage.
        
        Returns:
            Connectivity score (0-100)
        """
        unreachable = self.network.compute_unreachable_neurons()
        total = len(self.network.neurons)
        
        if total == 0:
            return 0.0
            
        return 100.0 * (1.0 - (unreachable / total))

    def get_network_stats(self) -> Dict:
        """Get current network statistics."""
        return {
            'connectivity': {
                'current': self.calculate_connectivity(),
                'unreachable_neurons': self.network.compute_unreachable_neurons()
            },
            'network': {
                'total_neurons': len(self.network.neurons),
                'input_neurons': len([n for n in self.network.neurons.values() 
                                    if n.type == NeuronType.INPUT]),
                'output_neurons': len([n for n in self.network.neurons.values()
                                     if n.type == NeuronType.OUTPUT]),
                'hidden_neurons': len([n for n in self.network.neurons.values()
                                     if n.type == NeuronType.HIDDEN])
            },
            'current_state': {
                'pickup_bag_size': len(self.pickup_bag),
                'processed_neurons': len(self.processed_neurons),
                'current_position': (
                    self.navigator.current_pos.x,
                    self.navigator.current_pos.y,
                    self.navigator.current_pos.z
                )
            }
        }

    def _start_background_updates(self, update_interval: float = 1.0):
        """Start background thread for monitor updates."""
        def update_loop():
            while True:
                try:
                    if self.monitor and self.monitor.visualization_enabled:
                        state = self.network.get_network_state()
                        self.monitor.update_neuron_positions(state['neuron_positions'])
                except Exception as e:
                    if self.debug:
                        print(f"Error in background update: {e}")
                time.sleep(update_interval)

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
