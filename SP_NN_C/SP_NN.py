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
import numpy as np


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

        recent_activity = np.clip(recent_activity, 0.0, 1.0)

        self.radius *= self.radius_shrink_rate
        activity_growth = recent_activity * self.activation_radius_factor * self.max_radius
        self.radius = np.clip(self.radius + activity_growth, self.min_radius, self.max_radius)

    def try_activate(self, input_value: float, network: 'SpatialNeuralNetwork') -> bool:
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
        self.params = params
        self.debug = False

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
        self.neuron_lock = RLock()
        self.connection_lock = Lock()
        self.activation_lock = Lock()
        self.collection_lock = Lock()
        self.window_lock = Lock()

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

    def compute_network_health(self) -> dict:
        STRUCTURAL_WEIGHT = 0.30
        DENSITY_WEIGHT = 0.50
        PROXIMITY_WEIGHT = 0.20

        structural = self.compute_structural_connectivity()
        density = self.compute_connection_density()
        proximity = self._compute_proximity_score()

        combined_health = (
                STRUCTURAL_WEIGHT * structural +
                DENSITY_WEIGHT * density +
                PROXIMITY_WEIGHT * proximity
        )

        return {
            'structural_connectivity': structural,
            'connection_density': density,
            'proximity_score': proximity,
            'combined_health': combined_health,
            'metadata': {
                'structural_weight': STRUCTURAL_WEIGHT,
                'density_weight': DENSITY_WEIGHT,
                'proximity_weight': PROXIMITY_WEIGHT,
                'raw_density': density,
                'raw_proximity': proximity
            }
        }

    def _compute_proximity_score(self) -> float:
        total_potential = 0
        hidden_neurons = [n for n in self.neurons.values()
                          if n.type == NeuronType.HIDDEN]

        if len(hidden_neurons) <= 1:
            return 0.0

        positions = np.array([[n.position.x, n.position.y, n.position.z]
                              for n in hidden_neurons])
        tree = KDTree(positions)

        proximity_radius = self.params.max_radius * 1.2

        for i, neuron in enumerate(hidden_neurons):
            neighbors = tree.query_ball_point(
                [neuron.position.x, neuron.position.y, neuron.position.z],
                proximity_radius
            )

            if i in neighbors:
                neighbors.remove(i)

            total_potential += len(neighbors)

        max_potential = len(hidden_neurons) * (len(hidden_neurons) - 1) / 2
        proximity_score = (total_potential / (2 * max_potential)) * 100

        return np.clip(proximity_score, 0.0, 100.0)

    def compute_structural_connectivity(self) -> float:
        """
        Computes structural connectivity using component counting via Union-Find algorithm.
        """
        with self.neuron_lock:
            N = len(self.neurons)
            if N <= 1:
                return 100.0  # Trivially connected for 0 or 1 neurons

            parent = {neuron.id: neuron.id for neuron in self.neurons.values()}
            rank = {neuron.id: 0 for neuron in self.neurons.values()}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]  # Path compression
                    x = parent[x]
                return x

            def union(x, y):
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

            for neuron in self.neurons.values():
                neuron_id = neuron.id
                for target in neuron.connections:
                    union(neuron_id, target.id)

            unique_roots = set(find(x) for x in parent)
            C = len(unique_roots)

            score = ((N - C) / (N - 1)) * 100

            return max(0.0, min(score, 100.0))  # Ensure score is in [0,100]

    def compute_connection_density(self) -> float:
        """
        Computes connection density using validated mathematical framework.
        """
        with self.neuron_lock:
            hidden_neurons = [n for n in self.neurons.values()
                              if n.type == NeuronType.HIDDEN and
                              n in self.hidden_neurons and
                              n not in self.interface_neurons]

            if not hidden_neurons:
                return 0.0

            E_actual = sum(len([c for c in n.connections
                                if c in hidden_neurons])
                           for n in hidden_neurons)

            radii = [n.radius for n in hidden_neurons if n.radius > 0]
            if not radii:
                return 0.0

            R_max = self.params.max_radius
            R_max_actual = max(radii)
            R_min_actual = min(radii)

            N = len(hidden_neurons)
            E_fully_connected = N * (N - 1)
            MARGIN = 0.90

            try:
                density = (E_actual * R_max ** 3) / (E_fully_connected *
                                                     (R_max_actual ** 3 - R_min_actual ** 3) * MARGIN) * 100

                return max(0.0, min(density, 100.0))
            except ZeroDivisionError:
                return 0.0

def plot_network_with_weights(network: SpatialNeuralNetwork):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    color_map = {
        NeuronType.INPUT: 'red',
        NeuronType.HIDDEN: 'blue',
        NeuronType.OUTPUT: 'green'
    }
    weights = [neuron.calculate_weight(target)
               for neuron in network.neurons.values()
               for target in neuron.connections]
    if not weights:
        weights = [0.0]
    norm = Normalize(vmin=min(weights), vmax=max(weights))
    cmap = plt.get_cmap('viridis')
    line_segments = []
    colors = []

    for neuron in network.neurons.values():
        color = 'cyan' if neuron in network.interface_neurons else color_map[neuron.type]
        ax.scatter(neuron.position.x, neuron.position.y, neuron.position.z,
                   color=color, s=50)
        for target in neuron.connections:
            xs = [neuron.position.x, target.position.x]
            ys = [neuron.position.y, target.position.y]
            zs = [neuron.position.z, target.position.z]
            line_segments.append(list(zip(xs, ys, zs)))
            weight = neuron.calculate_weight(target)
            colors.append(weight)

    if line_segments:
        lc = Line3DCollection(line_segments, cmap=cmap, norm=norm)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label='Connection Weight')

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
    half_volume = params.volume_size / 2
    input_radius = params.max_radius * params.input_radius_factor
    interface_radius = params.max_radius * params.interface_radius_factor
    occupied_positions = set()

    def is_position_occupied(position: Position) -> bool:
        position_key = (int(position.x), int(position.y), int(position.z))
        return position_key in occupied_positions

    def add_position(position: Position):
        position_key = (int(position.x), int(position.y), int(position.z))
        occupied_positions.add(position_key)

    # Create input and interface neurons
    for _ in range(params.num_input):
        while True:
            input_pos = Position(
                x=float(np.random.randint(-int(half_volume), int(half_volume))),
                y=float(np.random.randint(-int(half_volume), int(half_volume))),
                z=float(np.random.randint(-int(half_volume), int(half_volume)))
            )
            if not is_position_occupied(input_pos):
                add_position(input_pos)
                break

        input_neuron = Neuron(id_counter, NeuronType.INPUT, input_pos, params)
        input_neuron.radius = input_radius
        input_neuron.radius_mutable = False
        network.add_neuron(input_neuron)
        id_counter += 1

        while True:
            dx = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)
            dz = np.random.randint(-1, 2)
            new_x = (int(input_pos.x) + dx + int(half_volume)) % int(params.volume_size) - int(half_volume)
            new_y = (int(input_pos.y) + dy + int(half_volume)) % int(params.volume_size) - int(half_volume)
            new_z = (int(input_pos.z) + dz + int(half_volume)) % int(params.volume_size) - int(half_volume)
            interface_pos = Position(x=float(new_x), y=float(new_y), z=float(new_z))
            if not is_position_occupied(interface_pos):
                add_position(interface_pos)
                break

        interface_neuron = Neuron(id_counter, NeuronType.HIDDEN, interface_pos, params)
        interface_neuron.radius = interface_radius
        interface_neuron.radius_mutable = False
        network.add_neuron(interface_neuron)
        id_counter += 1

    # Create hidden neurons
    for _ in range(params.num_hidden):
        while True:
            pos = Position(
                x=float(np.random.randint(-int(half_volume), int(half_volume))),
                y=float(np.random.randint(-int(half_volume), int(half_volume))),
                z=float(np.random.randint(-int(half_volume), int(half_volume)))
            )
            if not is_position_occupied(pos):
                add_position(pos)
                break

        neuron = Neuron(id_counter, NeuronType.HIDDEN, pos, params)
        min_hidden_radius = params.hidden_radius_range[0] * params.max_radius
        max_hidden_radius = params.hidden_radius_range[1] * params.max_radius
        neuron.radius = np.random.uniform(min_hidden_radius, max_hidden_radius)
        network.add_neuron(neuron)
        id_counter += 1

    # Create output neurons
    for _ in range(params.num_output):
        while True:
            pos = Position(
                x=float(np.random.randint(-int(half_volume), int(half_volume))),
                y=float(np.random.randint(-int(half_volume), int(half_volume))),
                z=float(np.random.randint(-int(half_volume), int(half_volume)))
            )
            if not is_position_occupied(pos):
                add_position(pos)
                break

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
        'volume_size': 8.0,
        'num_input': 100,
        'num_output': 4,
        'total_neurons': 500,
        'activation_budget': 1000,
        'time_window_size': 100
    }

    # Create network with consistent parameters
    params = NetworkParameters(**network_params)
    network = create_network(params)

    # Get initial network health metrics
    health_metrics = network.compute_network_health()
    print("\nInitial Network Health Metrics:")
    print(f"Structural Connectivity: {health_metrics['structural_connectivity']:.1f}%")
    print(f"Connection Density: {health_metrics['connection_density']:.1f}%")
    print(f"Proximity Score: {health_metrics['proximity_score']:.1f}%")
    print(f"Combined Health: {health_metrics['combined_health']:.1f}%")

    # Get positions of hidden neurons
    hidden_positions = network.get_hidden_neuron_positions()

    # Modify positions of hidden neurons
    modified_positions = {}
    for neuron_id, pos in hidden_positions.items():
        dx = np.random.uniform(-1.0, 1.0)
        dy = np.random.uniform(-1.0, 1.0)
        dz = np.random.uniform(-1.0, 1.0)
        new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
        half_volume = params.volume_size / 2
        new_pos = (
            np.clip(new_pos[0], -half_volume, half_volume - 1),
            np.clip(new_pos[1], -half_volume, half_volume - 1),
            np.clip(new_pos[2], -half_volume, half_volume - 1)
        )
        modified_positions[neuron_id] = new_pos

    # Update hidden neuron positions
    network.update_hidden_neuron_positions(modified_positions)

    # Compute updated health metrics
    health_metrics_after = network.compute_network_health()
    print("\nNetwork Health Metrics After Position Updates:")
    print(f"Structural Connectivity: {health_metrics_after['structural_connectivity']:.1f}%")
    print(f"Connection Density: {health_metrics_after['connection_density']:.1f}%")
    print(f"Proximity Score: {health_metrics_after['proximity_score']:.1f}%")
    print(f"Combined Health: {health_metrics_after['combined_health']:.1f}%")

    # Print detailed metadata
    print("\nHealth Calculation Details:")
    meta = health_metrics_after['metadata']
    print(f"Structural Weight: {meta['structural_weight']}")
    print(f"Density Weight: {meta['density_weight']}")
    print(f"Proximity Weight: {meta['proximity_weight']}")
    print(f"Raw Density: {meta['raw_density']:.1f}%")
    print(f"Raw Proximity: {meta['raw_proximity']:.1f}%")

    # Plot the network
    plot_network_with_weights(network)
    plt.tight_layout()
    plt.show()
