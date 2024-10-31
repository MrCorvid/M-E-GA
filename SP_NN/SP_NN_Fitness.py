from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from SP_NN import Position, NetworkParameters, create_network, NeuronType
import sys

class NetworkEvolutionFitness:
    def __init__(
        self,
        network_params: Dict[str, any],
        update_best_func=None,
        max_path_length: int = 200,
        path_step_reward: float = 0.5,
        debug: bool = False
    ):
        """
        Initialize Phase 1 fitness evaluation focusing on network connectivity structure.

        Parameters:
            network_params (dict): Dictionary containing network initialization parameters.
            update_best_func (callable, optional): Function to update the best individuals.
            max_path_length (int): Maximum allowed path length before applying penalties.
            path_step_reward (float): Reward for each successful path step.
            debug (bool): Enable debug output printing.
        """
        self.debug = debug
        self.update_best = update_best_func

        # Path control parameters
        self.max_path_length = max_path_length
        self.path_step_reward = path_step_reward

        # Define path penalties
        self.failed_drop_penalty = -path_step_reward
        self.step_penalty = -2 * path_step_reward

        # Define genes for path evolution
        self.genes = ['U', 'D', 'F', 'B', 'DR']

        # Create network parameters using the provided network_params dictionary
        self.network_params = NetworkParameters(
            volume_size=network_params.get('volume_size', 8.0),
            num_input=network_params.get('num_input', 100),
            num_output=network_params.get('num_output', 4),
            total_neurons=network_params.get('total_neurons', 800),
            max_radius=network_params.get('max_radius', 3.0),
            min_radius=network_params.get('min_radius', 0.1),
            input_radius_factor=network_params.get('input_radius_factor', 0.25),
            interface_radius_factor=network_params.get('interface_radius_factor', 0.3),
            hidden_radius_range=network_params.get('hidden_radius_range', (0.02, 0.90)),
            interface_offset=network_params.get('interface_offset', 1.0),
            activation_budget=network_params.get('activation_budget', 1000),
            time_window_size=network_params.get('time_window_size', 100),
            base_radius_shrink_rate=network_params.get('base_radius_shrink_rate', 0.95),
            activation_radius_factor=network_params.get('activation_radius_factor', 0.2),
            activation_threshold=network_params.get('activation_threshold', 0.5)
        )

        # Initialize the spatial neural network with the defined parameters
        self.network = create_network(self.network_params)

        # Initialize persistent path state
        self.pickup_bag = []
        self.current_pos = Position(0.0, 0.0, 0.0)

    def calculate_connectivity_score(self) -> float:
        """Calculate network connectivity as a percentage (0-100)"""
        unreachable = self.network.compute_unreachable_neurons()
        total_neurons = len(self.network.neurons)
        if total_neurons == 0:
            return 0.0
        return (1.0 - (unreachable / total_neurons)) * 100

    def execute_path(self, path: List[str]) -> Dict:
        """Execute movement path to modify network structure"""
        state = self.network.get_network_state()
        neuron_positions = state['neuron_positions']
        position_updates = {}

        # Tracking metrics
        successful_steps = 0  # Counts moves and successful drops
        failed_drops = 0      # Counts failed drop attempts
        neurons_moved = 0     # Counts successfully moved neurons

        for command in path:
            if command == 'DR' and self.pickup_bag:
                current_pos_tuple = (
                    round(self.current_pos.x),
                    round(self.current_pos.y),
                    round(self.current_pos.z)
                )

                # Check if position is occupied by other neurons
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
                    successful_steps += 1
                else:
                    # Failed drop attempt
                    failed_drops += 1
            else:
                # Movement command
                new_pos = self.move(command)
                if new_pos:
                    self.current_pos = new_pos
                    successful_steps += 1

                    # Pickup any neurons at current position
                    current_pos_tuple = (
                        round(self.current_pos.x),
                        round(self.current_pos.y),
                        round(self.current_pos.z)
                    )

                    # Only pick up hidden neurons
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
                                break

        if position_updates:
            self.network.update_neuron_positions(position_updates)

        # Calculate path score
        path_score = 0.0

        # Add rewards for successful steps (moves and drops)
        path_score += successful_steps * self.path_step_reward

        # Add penalties for failed drops
        path_score += failed_drops * self.failed_drop_penalty

        # Add penalties for exceeding max path length
        if successful_steps > self.max_path_length:
            excess_steps = successful_steps - self.max_path_length
            path_score += excess_steps * self.step_penalty

        return {
            'path_score': path_score,
            'successful_steps': successful_steps,
            'failed_drops': failed_drops,
            'neurons_moved': neurons_moved,
            'total_neurons': len(self.network.neurons),
            'pickup_bag_size': len(self.pickup_bag)
        }

    def compute(self, encoded_individual, ga_instance) -> float:
        """
        Phase 1 fitness computation. Final fitness is:
        fitness = connectivity_score + path_score

        connectivity_score: 0-100 (percentage of reachable neurons)
        path_score: raw score based on steps, drops, and penalties

        Exits Phase 1 when 100% connectivity is achieved.
        """
        # Decode and execute path
        path = ga_instance.decode_organism(encoded_individual)
        movement_results = self.execute_path(path)

        # Get raw path score from execution
        path_score = movement_results['path_score']

        # Calculate connectivity percentage (0-100)
        connectivity_score = int(self.calculate_connectivity_score())

        # Check if we've achieved 100% connectivity
        if connectivity_score == 100:
            if self.debug:
                print("\nExiting Phase 1: Achieved 100% network connectivity")
                print(f"Final Stats:")
                print(f"Neurons Moved: {movement_results['neurons_moved']}")
                print(f"Path Length: {movement_results['successful_steps']}")
                print(f"Failed Drops: {movement_results['failed_drops']}")
            sys.exit(0)  # Clean exit after achieving goal

        # Final fitness is the connectivity percentage plus raw path score
        fitness = connectivity_score + path_score

        if self.debug:
            print(f"\nFitness Calculation Details:")
            print(f"Path Length: {movement_results['successful_steps']}")
            print(f"Successful Drops: {movement_results['neurons_moved']}")
            print(f"Failed Drops: {movement_results['failed_drops']}")
            print(f"Raw Path Score: {path_score:.2f}")
            print(f"Connectivity Score: {connectivity_score}%")
            print(f"Combined Fitness: {fitness:.2f}")

        if self.update_best:
            self.update_best(encoded_individual, fitness)

        return fitness

    def move(self, direction: str) -> Optional[Position]:
        """Calculate new position after movement"""
        moves = {
            'U': (0, 1, 0),
            'D': (0, -1, 0),
            'F': (1, 0, 0),
            'B': (-1, 0, 0)
        }

        if direction in moves:
            dx, dy, dz = moves[direction]
            new_x = round(self.current_pos.x + dx) % round(self.network_params.volume_size)
            new_y = round(self.current_pos.y + dy) % round(self.network_params.volume_size)
            new_z = round(self.current_pos.z + dz) % round(self.network_params.volume_size)
            return Position(float(new_x), float(new_y), float(new_z))
        return None

    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'connectivity': {
                'current': self.calculate_connectivity_score(),
                'unreachable_neurons': self.network.compute_unreachable_neurons()
            },
            'path': {
                'max_length': self.max_path_length,
                'step_reward': self.path_step_reward,
                'failed_drop_penalty': self.failed_drop_penalty,
                'step_penalty': self.step_penalty
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
                'current_position': (self.current_pos.x, self.current_pos.y, self.current_pos.z)
            }
        }