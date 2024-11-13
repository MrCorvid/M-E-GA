import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Deque
from dataclasses import dataclass
from collections import deque
from enum import Enum
from scipy.spatial import KDTree
from threading import Lock, RLock
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# Neuron type enumeration
class NeuronType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


# 3D position class
@dataclass
class Position:
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Position') -> float:
        return np.sqrt((self.x - other.x) ** 2 +
                      (self.y - other.y) ** 2 +
                      (self.z - other.z) ** 2)


@dataclass
class NetworkParameters:
    volume_size: float = 100.0
    num_input: int = 10
    num_output: int = 20
    total_neurons: int = 100
    max_radius: float = 3.0
    min_radius: float = 0.1
    input_radius_factor: float = 0.25
    interface_radius_factor: float = 1
    hidden_radius_range: Tuple[float, float] = (.50, 1.0)
    interface_offset: float = 1.
    activation_budget: int = 1000
    time_window_size: int = 100
    base_radius_shrink_rate: float = 0.95
    activation_radius_factor: float = 0.2
    activation_threshold: float = 0.5

    @property
    def num_hidden(self) -> int:
        return self.total_neurons - (self.num_input * 2) - self.num_output


class Neuron:
    def __init__(self, id: int, type: NeuronType, position: Position, params: NetworkParameters):
        self.id = id
        self.type = type
        self.position = position
        self.radius = 0.0
        self.activation = 0.0
        self.connections: Set['Neuron'] = set()
        self.activation_history = deque(maxlen=10)
        self.activation_threshold = params.activation_threshold
        self.radius_mutable = True
        self.min_radius = params.min_radius
        self.max_radius = params.max_radius
        self.radius_shrink_rate = params.base_radius_shrink_rate
        self.activation_radius_factor = params.activation_radius_factor
        self.last_activation_time = 0

    def calculate_weight(self, target: 'Neuron') -> float:
        distance = self.position.distance_to(target.position)
        # Add small epsilon to prevent division by zero
        return np.clip(1.0 / (1.0 + max(distance, 1e-10)), 0.0, 1.0)

    def can_connect_to(self, target: 'Neuron') -> bool:
        if self.type == NeuronType.OUTPUT:
            return False
        if target.type == NeuronType.INPUT:
            return False
        distance = self.position.distance_to(target.position)
        return distance <= self.radius

    def update_radius(self, current_time: int) -> None:
        if not self.radius_mutable:
            return

        recent_activity = (sum(self.activation_history) / len(self.activation_history)
                           if self.activation_history else 0.0)

        # Clip recent activity to prevent excessive growth
        recent_activity = np.clip(recent_activity, 0.0, 1.0)

        self.radius *= self.radius_shrink_rate
        activity_growth = recent_activity * self.activation_radius_factor * self.max_radius
        self.radius = np.clip(self.radius + activity_growth, self.min_radius, self.max_radius)

    def try_activate(self, input_value: float, network: 'SpatialNeuralNetwork') -> bool:
        # Clip input value to prevent overflow
        input_value = np.clip(input_value, 0.0, 1.0)

        if network.can_spend_activation():
            if input_value >= self.activation_threshold:
                self.activation = input_value
                self.activation_history.append(input_value)
                self.last_activation_time = network.time_step
                network.spend_activation()
                return True
        return False


class SpatialNeuralNetwork:
    def __init__(self, params: NetworkParameters):
        # Network parameters
        self.params = params

        # Neuron collections
        self.neurons: Dict[int, Neuron] = {}
        self.input_neurons: Set[Neuron] = set()
        self.hidden_neurons: Set[Neuron] = set()
        self.output_neurons: Set[Neuron] = set()
        self.interface_neurons: Set[Neuron] = set()

        # Timing and activation tracking
        self.time_step = 0
        self.window_start_time = 0
        self.window_activations = 0
        self.window_history: Deque[dict] = deque(maxlen=100)
        self.activation_times: Deque[int] = deque()

        # Thread safety controls
        self.neuron_lock = RLock()  # Reentrant lock for neuron operations
        self.connection_lock = Lock()  # Lock for connection updates
        self.activation_lock = Lock()  # Lock for activation budget
        self.collection_lock = Lock()  # Lock for modifying neuron collections
        self.window_lock = Lock()  # Lock for window operations

        # Thread pool for parallel operations
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(50, (params.total_neurons // 50) + 1)
        )

    def add_neuron(self, neuron: Neuron):
        with self.collection_lock:
            self.neurons[neuron.id] = neuron
            if neuron.type == NeuronType.INPUT:
                self.input_neurons.add(neuron)
            elif neuron.type == NeuronType.HIDDEN:
                self.hidden_neurons.add(neuron)
                if not neuron.radius_mutable:
                    self.interface_neurons.add(neuron)
            else:
                self.output_neurons.add(neuron)

    def get_network_state(self) -> dict:
        with self.neuron_lock:
            return {
                'volume_size': self.params.volume_size,
                'neuron_positions': {
                    neuron.id: {
                        'position': (neuron.position.x, neuron.position.y, neuron.position.z),
                        'type': neuron.type,
                        'radius': neuron.radius,
                        'activation': neuron.activation
                    }
                    for neuron in self.neurons.values()
                }
            }

    def update_neuron_positions(self, new_positions: dict) -> None:
        with self.neuron_lock:
            for neuron_id, new_pos in new_positions.items():
                if neuron_id in self.neurons:
                    self.neurons[neuron_id].position = Position(
                        x=new_pos[0],
                        y=new_pos[1],
                        z=new_pos[2]
                    )
        self.update_connections()

    def get_hidden_neuron_positions(self) -> Dict[int, Tuple[float, float, float]]:
        with self.neuron_lock:
            return {neuron.id: (neuron.position.x, neuron.position.y, neuron.position.z)
                    for neuron in self.hidden_neurons}

    def update_hidden_neuron_positions(self, new_positions: Dict[int, Tuple[float, float, float]]) -> None:
        with self.neuron_lock:
            for neuron_id, pos in new_positions.items():
                if neuron_id in self.neurons and self.neurons[neuron_id].type == NeuronType.HIDDEN:
                    self.neurons[neuron_id].position = Position(
                        x=pos[0],
                        y=pos[1],
                        z=pos[2]
                    )
        self.update_connections()

    def update_connections(self):
        with self.connection_lock:
            with self.neuron_lock:
                positions = np.array([[n.position.x, n.position.y, n.position.z]
                                      for n in self.neurons.values()])
                neuron_list = list(self.neurons.values())

            if len(positions) == 0:
                return

            tree = KDTree(positions)

            def update_neuron_connections(neuron_idx):
                neuron = neuron_list[neuron_idx]
                if neuron.type == NeuronType.OUTPUT:
                    return

                indices = tree.query_ball_point(positions[neuron_idx], r=neuron.radius)
                new_connections = set()

                for idx in indices:
                    if idx != neuron_idx:
                        target = neuron_list[idx]
                        if neuron.can_connect_to(target):
                            new_connections.add(target)

                with self.neuron_lock:
                    neuron.connections = new_connections

            futures = [
                self.pool.submit(update_neuron_connections, i)
                for i in range(len(neuron_list))
            ]
            concurrent.futures.wait(futures)

    def run_cycle(self, inputs: Optional[List[float]] = None) -> List[float]:
        with self.activation_lock:
            self.time_step += 1

        if inputs:
            def process_input(input_pair):
                input_neuron, input_value = input_pair
                with self.neuron_lock:
                    return input_neuron.try_activate(input_value, self)

            input_pairs = list(zip(self.input_neurons,
                                   [np.clip(x, 0.0, 1.0) for x in inputs]))
            futures = [
                self.pool.submit(process_input, pair)
                for pair in input_pairs
            ]
            concurrent.futures.wait(futures)

        def process_hidden_neuron(neuron):
            with self.neuron_lock:
                weighted_inputs = []
                for n in self.neurons.values():
                    if neuron in n.connections:
                        weight = n.calculate_weight(neuron)
                        contribution = n.activation * weight
                        weighted_inputs.append(contribution)

                if weighted_inputs:
                    total_input = np.clip(np.sum(weighted_inputs), 0.0, 1.0)
                    if len(weighted_inputs) > 1:
                        total_input /= len(weighted_inputs)
                else:
                    total_input = 0.0

                return neuron.try_activate(total_input, self)

        futures = [
            self.pool.submit(process_hidden_neuron, neuron)
            for neuron in self.hidden_neurons
        ]
        concurrent.futures.wait(futures)

        with self.neuron_lock:
            outputs = [neuron.activation for neuron in self.output_neurons]

        if self.update_window():
            def update_neuron_radius(neuron):
                with self.neuron_lock:
                    neuron.update_radius(self.time_step)

            futures = [
                self.pool.submit(update_neuron_radius, neuron)
                for neuron in self.neurons.values()
            ]
            concurrent.futures.wait(futures)

            self.update_connections()

        return outputs

    def run_evaluation(self, steps: int) -> dict:
        results = self.run_window(steps)
        return {
            'activation_count': self.window_activations,
            'window_stats': self.get_window_stats(),
            'network_state': self.get_network_state()
        }

    def update_window(self) -> bool:
        with self.window_lock:
            if (self.time_step - self.window_start_time) >= self.params.time_window_size:
                window_stats = {
                    'start_time': self.window_start_time,
                    'end_time': self.time_step,
                    'total_activations': self.window_activations,
                    'activation_times': list(self.activation_times)
                }
                self.window_history.append(window_stats)

                self.window_start_time = self.time_step
                self.window_activations = 0
                self.activation_times.clear()
                return True
            return False

    def can_spend_activation(self) -> bool:
        with self.activation_lock:
            return self.window_activations < self.params.activation_budget

    def spend_activation(self) -> None:
        with self.activation_lock:
            self.window_activations += 1
            self.activation_times.append(self.time_step)

    def run_window(self, cycles: int, input_sequence: Optional[List[List[float]]] = None) -> dict:
        window_outputs = []
        completed_cycles = 0

        for i in range(cycles):
            with self.activation_lock:
                if self.window_activations >= self.params.activation_budget:
                    break

            cycle_inputs = input_sequence[i] if input_sequence else None
            outputs = self.run_cycle(cycle_inputs)
            window_outputs.append(outputs)
            completed_cycles = i + 1

        return {
            'outputs': window_outputs,
            'activations': self.window_activations,
            'completed_cycles': completed_cycles
        }

    def get_window_stats(self) -> dict:
        with self.window_lock:
            return {
                'total_activations': self.window_activations,
                'window_progress': (self.time_step - self.window_start_time) / self.params.time_window_size,
                'activation_density': len(self.activation_times) / self.params.time_window_size,
                'remaining_budget': self.params.activation_budget - self.window_activations
            }

    def compute_unreachable_neurons(self) -> float:
        """
        Computes the connectivity score of neurons using component counting.
        The score ranges from 0 to 100:
            - 0 indicates low connectivity (high fragmentation).
            - 100 indicates full connectivity (single connected component).

        Returns:
            float: Connectivity score between 0 and 100.
        """
        with self.neuron_lock:
            N = len(self.neurons)
            if N <= 1:
                # If there are 0 or 1 neurons, the network is trivially fully connected
                return 100.0

            # Initialize Union-Find (Disjoint Set Union) structure
            parent = {neuron.id: neuron.id for neuron in self.neurons.values()}
            rank = {neuron.id: 0 for neuron in self.neurons.values()}

            def find(x):
                """Finds the root parent of x with path compression."""
                while parent[x] != x:
                    parent[x] = parent[parent[x]]  # Path compression
                    x = parent[x]
                return x

            def union(x, y):
                """Unites the sets containing x and y using union by rank."""
                root_x = find(x)
                root_y = find(y)
                if root_x == root_y:
                    return
                if rank[root_x] < rank[root_y]:
                    parent[root_x] = root_y
                else:
                    parent[root_y] = root_x
                    if rank[root_x] == rank[root_y]:
                        rank[root_x] += 1

            # Union neurons based on connections
            for neuron in self.neurons.values():
                neuron_id = neuron.id
                for target in neuron.connections:
                    union(neuron_id, target.id)

            # Count unique connected components
            unique_roots = set(find(x) for x in parent)
            C = len(unique_roots)

            # Calculate connectivity score
            # Formula: ((N - C) / (N - 1)) * 100
            # - C = 1 (fully connected): ((N - 1) / (N - 1)) * 100 = 100%
            # - C = N (fully disconnected): ((0) / (N - 1)) * 100 = 0%
            score = ((N - C) / (N - 1)) * 100

            # Clamp the score to ensure it's within [0, 100]
            score = max(0.0, min(score, 100.0))

            return score


def plot_network_with_weights(network: SpatialNeuralNetwork):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    color_map = {
        NeuronType.INPUT: 'red',
        NeuronType.HIDDEN: 'blue',
        NeuronType.OUTPUT: 'green'
    }
    # Get all weights for normalization
    weights = [neuron.calculate_weight(target)
               for neuron in network.neurons.values()
               for target in neuron.connections]
    if not weights:
        weights = [0.0]  # Avoid errors if weights list is empty
    norm = Normalize(vmin=min(weights), vmax=max(weights))
    cmap = plt.get_cmap('viridis')
    # Create lists to store line segments and their colors
    line_segments = []
    colors = []
    # Plot neurons
    for neuron in network.neurons.values():
        # Plot neuron
        color = 'cyan' if neuron in network.interface_neurons else color_map[neuron.type]
        ax.scatter(neuron.position.x, neuron.position.y, neuron.position.z,
                   color=color, s=50)
        # Prepare connections for Line3DCollection
        for target in neuron.connections:
            xs = [neuron.position.x, target.position.x]
            ys = [neuron.position.y, target.position.y]
            zs = [neuron.position.z, target.position.z]
            line_segments.append(list(zip(xs, ys, zs)))
            weight = neuron.calculate_weight(target)
            colors.append(weight)
    if line_segments:
        # Create a Line3DCollection with the line segments
        lc = Line3DCollection(line_segments, cmap=cmap, norm=norm)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)
        # Add colorbar for weights
        cbar = fig.colorbar(lc, ax=ax, label='Connection Weight')
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               label='Input Neuron', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               label='Hidden Neuron', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
               label='Interface Neuron', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               label='Output Neuron', markersize=10)
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Spatial Neural Network with Connection Weights Heatmap')


def create_network(params: NetworkParameters) -> SpatialNeuralNetwork:
    network = SpatialNeuralNetwork(params)
    id_counter = 0

    half_volume = params.volume_size / 2  # Define half the volume size for centering

    input_radius = params.max_radius * params.input_radius_factor
    interface_radius = params.max_radius * params.interface_radius_factor

    # Create input and interface neurons
    for _ in range(params.num_input):
        # Generate continuous position for input neuron
        input_pos = Position(
            x=np.random.uniform(-half_volume, half_volume),
            y=np.random.uniform(-half_volume, half_volume),
            z=np.random.uniform(-half_volume, half_volume)
        )

        input_neuron = Neuron(id_counter, NeuronType.INPUT, input_pos, params)
        input_neuron.radius = input_radius
        input_neuron.radius_mutable = False
        network.add_neuron(input_neuron)
        id_counter += 1

        # Generate interface neuron position with continuous offset
        dx = np.random.uniform(-params.interface_offset, params.interface_offset)
        dy = np.random.uniform(-params.interface_offset, params.interface_offset)
        dz = np.random.uniform(-params.interface_offset, params.interface_offset)

        # Apply wrapping with centered volume
        new_x = (input_pos.x + dx + half_volume) % params.volume_size - half_volume
        new_y = (input_pos.y + dy + half_volume) % params.volume_size - half_volume
        new_z = (input_pos.z + dz + half_volume) % params.volume_size - half_volume

        interface_pos = Position(
            x=float(new_x),
            y=float(new_y),
            z=float(new_z)
        )

        interface_neuron = Neuron(id_counter, NeuronType.HIDDEN, interface_pos, params)
        interface_neuron.radius = interface_radius
        interface_neuron.radius_mutable = False
        network.add_neuron(interface_neuron)
        id_counter += 1

    # Create hidden neurons
    for _ in range(params.num_hidden):
        pos = Position(
            x=np.random.uniform(-half_volume, half_volume),
            y=np.random.uniform(-half_volume, half_volume),
            z=np.random.uniform(-half_volume, half_volume)
        )

        neuron = Neuron(id_counter, NeuronType.HIDDEN, pos, params)
        min_hidden_radius = params.hidden_radius_range[0] * params.max_radius
        max_hidden_radius = params.hidden_radius_range[1] * params.max_radius
        neuron.radius = np.random.uniform(min_hidden_radius, max_hidden_radius)
        network.add_neuron(neuron)
        id_counter += 1

    # Create output neurons
    for _ in range(params.num_output):
        pos = Position(
            x=np.random.uniform(-half_volume, half_volume),
            y=np.random.uniform(-half_volume, half_volume),
            z=np.random.uniform(-half_volume, half_volume)
        )

        output_neuron = Neuron(id_counter, NeuronType.OUTPUT, pos, params)
        output_neuron.radius = 0.0
        output_neuron.radius_mutable = False
        network.add_neuron(output_neuron)
        id_counter += 1

    network.update_connections()
    return network


if __name__ == "__main__":
    # Define consistent network parameters
    network_params = {
        'volume_size': 8.0,  # Example volume size
        'num_input': 100,     # Align with fitness function
        'num_output': 4,      # Align with fitness function
        'total_neurons': 500, # Align with fitness function
        'activation_budget': 1000,  # Ensure all necessary parameters are included
        'time_window_size': 100
    }

    # Create network with consistent parameters
    params = NetworkParameters(
        volume_size=network_params['volume_size'],
        num_input=network_params['num_input'],
        num_output=network_params['num_output'],
        total_neurons=network_params['total_neurons'],
        activation_budget=network_params['activation_budget'],
        time_window_size=network_params['time_window_size']
    )
    network = create_network(params)

    # Compute number of unreachable neurons
    unreachable_neurons = network.compute_unreachable_neurons()
    print(f"Unreachable Neurons: {unreachable_neurons}")

    # Get positions of hidden neurons
    hidden_positions = network.get_hidden_neuron_positions()

    # Example: Modify positions of hidden neurons (e.g., move them randomly)
    modified_positions = {}
    for neuron_id, pos in hidden_positions.items():
        dx = np.random.uniform(-1.0, 1.0)
        dy = np.random.uniform(-1.0, 1.0)
        dz = np.random.uniform(-1.0, 1.0)
        new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
        # Ensure the new position stays within the volume bounds
        half_volume = params.volume_size / 2
        new_pos = (
            np.clip(new_pos[0], -half_volume, half_volume - 1),
            np.clip(new_pos[1], -half_volume, half_volume - 1),
            np.clip(new_pos[2], -half_volume, half_volume - 1)
        )
        modified_positions[neuron_id] = new_pos

    # Update hidden neuron positions in the network
    network.update_hidden_neuron_positions(modified_positions)

    # Recompute number of unreachable neurons after modification
    unreachable_neurons_after = network.compute_unreachable_neurons()
    print(f"Unreachable Neurons After Modification: {unreachable_neurons_after}")

    # Plot the network
    plot_network_with_weights(network)
    plt.tight_layout()
    plt.show()
