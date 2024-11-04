import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for faster rendering
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum
from scipy.spatial import KDTree
from threading import Lock, RLock
import concurrent.futures
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio
import os
import shutil
import atexit

# Neuron type enumeration
class NeuronType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"

# 3D position class with integer coordinates
@dataclass(frozen=True)
class Position:
    x: int
    y: int
    z: int

    def distance_to(self, other: 'Position') -> float:
        return np.sqrt((self.x - other.x) ** 2 +
                       (self.y - other.y) ** 2 +
                       (self.z - other.z) ** 2)

@dataclass
class NetworkParameters:
    volume_size: float
    num_input: int
    num_output: int
    total_neurons: int
    max_radius: float
    min_radius: float
    input_radius_factor: float
    interface_radius_factor: float
    hidden_radius_range: Tuple[float, float]
    interface_offset: float
    activation_budget: int
    time_window_size: int
    base_radius_shrink_rate: float
    activation_radius_factor: float
    activation_threshold: float

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
        return True  # Distance is already handled by KDTree

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
    def __init__(self, params: NetworkParameters, save_gif: bool = True):
        # Network parameters
        self.params = params

        # Neuron collections
        self.neurons: Dict[int, Neuron] = {}
        self.input_neurons: List[Neuron] = []
        self.hidden_neurons: Set[Neuron] = set()
        self.output_neurons: Set[Neuron] = set()
        self.interface_neurons: List[Neuron] = []
        self.input_to_interface_map: Dict[int, int] = {}  # Mapping from input neuron ID to interface neuron ID

        # Timing and activation tracking
        self.time_step = 0
        self.window_start_time = 0
        self.window_activations = 0
        self.window_history: deque = deque(maxlen=100)
        self.activation_times: deque = deque()

        # Thread safety controls
        self.neuron_lock = RLock()  # Reentrant lock for neuron operations
        self.activation_lock = Lock()  # Lock for activation budget
        self.collection_lock = Lock()  # Lock for modifying neuron collections
        self.window_lock = Lock()  # Lock for window operations

        # GIF saving controls
        self.save_gif = save_gif  # Flag to control plotting and GIF creation
        if self.save_gif:
            self.gif_directory = "network_frames"
            self.cleanup_frames()  # Clean up frames at the start
            self.frame_count = 0
            # Initialize GIF writer
            gif_path = os.path.join(self.gif_directory, "network_evolution.gif")
            self.gif_writer = imageio.get_writer(gif_path, mode='I', duration=0.5)
            atexit.register(self.save_gif_final)  # Ensure GIF writer is closed on exit

        # Initialize occupied_positions set
        self._occupied_positions: Set[Tuple[int, int, int]] = set()

    def cleanup_frames(self):
        """Remove old frames to start fresh for a new run."""
        shutil.rmtree(self.gif_directory, ignore_errors=True)
        os.makedirs(self.gif_directory, exist_ok=True)

    def add_neuron(self, neuron: Neuron):
        with self.collection_lock:
            self.neurons[neuron.id] = neuron
            if neuron.type == NeuronType.INPUT:
                self.input_neurons.append(neuron)
            elif neuron.type == NeuronType.HIDDEN:
                self.hidden_neurons.add(neuron)
                if not neuron.radius_mutable:
                    self.interface_neurons.append(neuron)
            else:
                self.output_neurons.add(neuron)

    def get_network_state(self) -> dict:
        with self.neuron_lock:
            return {
                'volume_size': self.params.volume_size,
                'neuron_positions': {
                    neuron.id: {
                        'position': (neuron.position.x, neuron.position.y, neuron.position.z),
                        'type': neuron.type.value,
                        'radius': neuron.radius,
                        'activation': neuron.activation
                    }
                    for neuron in self.neurons.values()
                }
            }

    def update_neuron_positions(self, new_positions: dict) -> None:
        changed_neurons = []
        with self.neuron_lock:
            for neuron_id, new_pos in new_positions.items():
                if neuron_id in self.neurons:
                    # Ensure positions are integers
                    new_x = int(round(new_pos[0]))
                    new_y = int(round(new_pos[1]))
                    new_z = int(round(new_pos[2]))
                    # Enforce volume bounds
                    new_x = max(0, min(new_x, int(self.params.volume_size) - 1))
                    new_y = max(0, min(new_y, int(self.params.volume_size) - 1))
                    new_z = max(0, min(new_z, int(self.params.volume_size) - 1))
                    new_position = Position(x=new_x, y=new_y, z=new_z)
                    # Check if the new position is already occupied
                    if not self.is_position_occupied(new_position):
                        self.update_position_occupied(self.neurons[neuron_id].position, False)
                        self.neurons[neuron_id].position = new_position
                        self.update_position_occupied(new_position, True)
                        changed_neurons.append(self.neurons[neuron_id])
        self.update_connections(changed_neurons)
        # After updating positions, save the network state if GIF saving is enabled
        if self.save_gif:
            self.plot_and_save_network()

    def get_hidden_neuron_positions(self) -> Dict[int, Tuple[int, int, int]]:
        with self.neuron_lock:
            return {neuron.id: (neuron.position.x, neuron.position.y, neuron.position.z)
                    for neuron in self.hidden_neurons}

    def update_hidden_neuron_positions(self, new_positions: Dict[int, Tuple[int, int, int]]) -> None:
        self.update_neuron_positions(new_positions)

    def update_connections(self, neurons_to_update=None):
        with self.neuron_lock:
            neuron_list = list(self.neurons.values())
            if not neuron_list:
                return
            positions = np.array([[n.position.x, n.position.y, n.position.z] for n in neuron_list])
            neuron_id_to_index = {neuron.id: idx for idx, neuron in enumerate(neuron_list)}

        tree = KDTree(positions)

        if neurons_to_update is None:
            neurons_to_update = self.neurons.values()
        else:
            neurons_to_update = list(neurons_to_update)

        def update_neuron_connections(neuron):
            if neuron.type in {NeuronType.INPUT, NeuronType.OUTPUT}:
                return  # Skip input and output neurons

            neuron_idx = neuron_id_to_index[neuron.id]
            indices = tree.query_ball_point(positions[neuron_idx], r=neuron.radius)
            new_connections = set()

            for idx in indices:
                if idx == neuron_idx:
                    continue  # Skip self
                target = neuron_list[idx]
                if target.type == NeuronType.INPUT:
                    continue  # Skip input neurons
                if neuron.can_connect_to(target):
                    new_connections.add(target)

            neuron.connections = new_connections  # No lock needed here

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(update_neuron_connections, neurons_to_update)

    def run_cycle(self, inputs: Optional[List[float]] = None) -> List[float]:
        with self.activation_lock:
            self.time_step += 1

        if inputs:
            # Activate input neurons
            input_pairs = zip(self.input_neurons, [np.clip(x, 0.0, 1.0) for x in inputs])

            def process_input(input_pair):
                input_neuron, input_value = input_pair
                return input_neuron.try_activate(input_value, self)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(process_input, input_pairs)

            # Activate interface neurons based on their connected input neuron
            def process_interface(interface_neuron):
                input_neuron_id = self.input_to_interface_map.get(interface_neuron.id)
                if input_neuron_id is not None:
                    input_neuron = self.neurons.get(input_neuron_id)
                    if input_neuron:
                        input_activation = input_neuron.activation
                        return interface_neuron.try_activate(input_activation, self)
                return False

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(process_interface, self.interface_neurons)

        def process_hidden_neuron(neuron):
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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_hidden_neuron, self.hidden_neurons)

        outputs = [neuron.activation for neuron in self.output_neurons]

        if self.update_window():
            def update_neuron_radius(neuron):
                neuron.update_radius(self.time_step)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(update_neuron_radius, self.neurons.values())

            self.update_connections()

        # After each cycle, save the network state if GIF saving is enabled
        if self.save_gif:
            self.plot_and_save_network()

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

            cycle_inputs = input_sequence[i] if input_sequence and i < len(input_sequence) else None
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

    def compute_unreachable_neurons(self) -> int:
        """
        Computes the number of neurons that cannot be reached from the input neurons.
        """
        with self.neuron_lock:
            reachable_neurons = set()

            # Initialize queue with input neurons
            queue = deque(self.input_neurons)
            for input_neuron in self.input_neurons:
                reachable_neurons.add(input_neuron.id)

            # Track neurons we've already processed to avoid cycles
            processed = set()

            while queue:
                current_neuron = queue.popleft()
                if current_neuron.id in processed:
                    continue

                processed.add(current_neuron.id)

                # Traverse outgoing connections from the current neuron
                for target in current_neuron.connections:
                    if target.id not in reachable_neurons:
                        reachable_neurons.add(target.id)
                        queue.append(target)

            # Calculate unreachable neurons
            total_neurons = len(self.neurons)
            unreachable = total_neurons - len(reachable_neurons)

        # Plot the network and save the frame after computing connectivity
        if self.save_gif:
            self.plot_and_save_network()

        return unreachable

    def plot_and_save_network(self):
        """Plot the network and save the plot as an image for GIF creation."""
        try:
            fig = plt.figure(figsize=(8, 6), dpi=80)
            ax = fig.add_subplot(111, projection='3d')
            color_map = {
                NeuronType.INPUT: 'red',
                NeuronType.HIDDEN: 'blue',
                NeuronType.OUTPUT: 'green'
            }
            # Prepare data for plotting
            neuron_positions = np.array([[neuron.position.x, neuron.position.y, neuron.position.z]
                                         for neuron in self.neurons.values()])
            neuron_colors = ['cyan' if neuron in self.interface_neurons else color_map[neuron.type]
                             for neuron in self.neurons.values()]

            # Plot neurons
            ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1], neuron_positions[:, 2],
                       c=neuron_colors, s=20, alpha=0.6)

            # Prepare connections
            line_segments = []
            colors = []
            for neuron in self.neurons.values():
                for target in neuron.connections:
                    xs = [neuron.position.x, target.position.x]
                    ys = [neuron.position.y, target.position.y]
                    zs = [neuron.position.z, target.position.z]
                    line_segments.append(list(zip(xs, ys, zs)))
                    weight = neuron.calculate_weight(target)
                    colors.append(weight)

            if line_segments:
                norm = Normalize(vmin=0, vmax=1)  # Normalize weights between 0 and 1
                cmap = plt.get_cmap('viridis')
                lc = Line3DCollection(line_segments, cmap=cmap, norm=norm, linewidths=0.5, alpha=0.5)
                lc.set_array(np.array(colors))
                ax.add_collection(lc)

            # Remove axes for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.grid(False)
            ax.set_title(f'3D Spatial Neural Network at Step {self.time_step}', fontsize=10)

            # Save the figure
            frame_filename = os.path.join(self.gif_directory, f"frame_{self.frame_count:04d}.png")
            plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Append to GIF
            if self.save_gif:
                if os.path.exists(frame_filename):
                    frame = imageio.imread(frame_filename)
                    self.gif_writer.append_data(frame)
                    self.frame_count += 1
        except Exception:
            pass  # Silently ignore any plotting errors to maintain performance

    def save_gif_final(self):
        """Close the GIF writer."""
        try:
            if self.save_gif:
                self.gif_writer.close()
        except Exception:
            pass  # Silently ignore any closing errors

    def is_position_occupied(self, position: Position) -> bool:
        position_key = (position.x, position.y, position.z)
        return position_key in self._occupied_positions

    def update_position_occupied(self, position: Position, occupied: bool) -> None:
        position_key = (position.x, position.y, position.z)
        if occupied:
            self._occupied_positions.add(position_key)
        else:
            self._occupied_positions.discard(position_key)

def create_network(params: NetworkParameters, save_gif: bool = False) -> SpatialNeuralNetwork:
    network = SpatialNeuralNetwork(params, save_gif=save_gif)
    id_counter = 0

    input_radius = params.max_radius * params.input_radius_factor
    interface_radius = params.max_radius * params.interface_radius_factor

    # Create input and interface neurons
    for _ in range(params.num_input):
        # Generate unique position for input neuron using integer positions
        while True:
            input_pos = Position(
                x=np.random.randint(0, int(params.volume_size)),
                y=np.random.randint(0, int(params.volume_size)),
                z=np.random.randint(0, int(params.volume_size))
            )
            if not network.is_position_occupied(input_pos):
                network.update_position_occupied(input_pos, True)
                break

        input_neuron = Neuron(id_counter, NeuronType.INPUT, input_pos, params)
        input_neuron.radius = input_radius
        input_neuron.radius_mutable = False
        network.add_neuron(input_neuron)
        input_neuron_id = id_counter
        id_counter += 1

        # Generate unique position for interface neuron adjacent to input
        while True:
            # Pick a random adjacent position (including diagonals)
            dx = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)
            dz = np.random.randint(-1, 2)
            new_x = (input_pos.x + dx) % int(params.volume_size)
            new_y = (input_pos.y + dy) % int(params.volume_size)
            new_z = (input_pos.z + dz) % int(params.volume_size)
            interface_pos = Position(
                x=new_x,
                y=new_y,
                z=new_z
            )
            if not network.is_position_occupied(interface_pos):
                network.update_position_occupied(interface_pos, True)
                break

        interface_neuron = Neuron(id_counter, NeuronType.HIDDEN, interface_pos, params)
        interface_neuron.radius = interface_radius
        interface_neuron.radius_mutable = False
        network.add_neuron(interface_neuron)
        interface_neuron_id = id_counter
        id_counter += 1

        # Establish direct connection between input neuron and its interface neuron
        input_neuron.connections.add(interface_neuron)

        # Map input neuron ID to interface neuron ID for activation
        network.input_to_interface_map[input_neuron_id] = interface_neuron_id

        # Map interface neuron ID to input neuron ID for activation
        if not hasattr(network, 'interface_to_input_map'):
            network.interface_to_input_map = {}
        network.interface_to_input_map[interface_neuron_id] = input_neuron_id

    # Create hidden neurons
    for _ in range(params.num_hidden):
        while True:
            pos = Position(
                x=np.random.randint(0, int(params.volume_size)),
                y=np.random.randint(0, int(params.volume_size)),
                z=np.random.randint(0, int(params.volume_size))
            )
            if not network.is_position_occupied(pos):
                network.update_position_occupied(pos, True)
                break

        neuron = Neuron(id_counter, NeuronType.HIDDEN, pos, params)
        min_hidden_radius = params.hidden_radius_range[0] * params.max_radius
        max_hidden_radius = params.hidden_radius_range[1] * params.max_radius
        # Using a uniform distribution for radius
        neuron.radius = np.random.uniform(min_hidden_radius, max_hidden_radius)
        network.add_neuron(neuron)
        id_counter += 1

    # Create output neurons
    for _ in range(params.num_output):
        while True:
            pos = Position(
                x=np.random.randint(0, int(params.volume_size)),
                y=np.random.randint(0, int(params.volume_size)),
                z=np.random.randint(0, int(params.volume_size))
            )
            if not network.is_position_occupied(pos):
                network.update_position_occupied(pos, True)
                break

        output_neuron = Neuron(id_counter, NeuronType.OUTPUT, pos, params)
        output_neuron.radius = 0.0
        output_neuron.radius_mutable = False
        network.add_neuron(output_neuron)
        id_counter += 1

    network.update_connections()
    # After initial creation, save the network state if GIF saving is enabled
    if network.save_gif:
        network.plot_and_save_network()
    return network

# End of SP_NN module
