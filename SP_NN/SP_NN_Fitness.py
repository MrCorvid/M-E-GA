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
import sys
import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


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
    volume_size: float = 25.0
    num_input: int = 100
    num_output: int = 20
    total_neurons: int = 400
    max_radius: float = 3.0
    min_radius: float = 0.1
    input_radius_factor: float = 0.25
    interface_radius_factor: float = 1
    hidden_radius_range: Tuple[float, float] = (0.10, 0.80)
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

    def compute_unreachable_neurons(self) -> int:
        """
        Computes unreachable neurons using component counting.
        Formula: ((N-C)/(N-1)) maps to connectivity
        Returns number of unreachable neurons.
        """
        with self.neuron_lock:
            N = len(self.neurons)
            if N == 0:
                return 0

            # Initialize Union-Find (using neuron IDs)
            parent = {neuron.id: neuron.id for neuron in self.neurons.values()}

            # Find with path compression
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            # Union neurons based on connections
            for neuron in self.neurons.values():
                neuron_id = neuron.id
                for target in neuron.connections:
                    # Connect both ways
                    pid = find(neuron_id)
                    tid = find(target.id)
                    if pid != tid:
                        parent[tid] = pid

            # Count unique components
            C = len(set(find(x) for x in parent))

            # Directly calculate unreachable neurons from component count
            # When C = 1 (fully connected), unreachable = 0
            # When C = N (fully disconnected), unreachable = N
            return N - int((N - C) * N / (N - 1)) if N > 1 else 0


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

    input_radius = params.max_radius * params.input_radius_factor
    interface_radius = params.max_radius * params.interface_radius_factor

    # Set to keep track of occupied positions
    occupied_positions = set()

    def is_position_occupied(position: Position) -> bool:
        position_key = (int(position.x), int(position.y), int(position.z))
        return position_key in occupied_positions

    def add_position(position: Position):
        position_key = (int(position.x), int(position.y), int(position.z))
        occupied_positions.add(position_key)

    # Create input and interface neurons
    for _ in range(params.num_input):
        # Generate unique position for input neuron using integer positions
        while True:
            input_pos = Position(
                x=float(np.random.randint(0, int(params.volume_size))),
                y=float(np.random.randint(0, int(params.volume_size))),
                z=float(np.random.randint(0, int(params.volume_size)))
            )
            if not is_position_occupied(input_pos):
                add_position(input_pos)
                break

        input_neuron = Neuron(id_counter, NeuronType.INPUT, input_pos, params)
        input_neuron.radius = input_radius
        input_neuron.radius_mutable = False
        network.add_neuron(input_neuron)
        id_counter += 1

        # Generate unique position for interface neuron adjacent to input
        while True:
            # Pick a random adjacent position (including diagonals)
            dx = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)
            dz = np.random.randint(-1, 2)
            interface_pos = Position(
                x=float((int(input_pos.x) + dx) % int(params.volume_size)),
                y=float((int(input_pos.y) + dy) % int(params.volume_size)),
                z=float((int(input_pos.z) + dz) % int(params.volume_size))
            )
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
                x=float(np.random.randint(0, int(params.volume_size))),
                y=float(np.random.randint(0, int(params.volume_size))),
                z=float(np.random.randint(0, int(params.volume_size)))
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
                x=float(np.random.randint(0, int(params.volume_size))),
                y=float(np.random.randint(0, int(params.volume_size))),
                z=float(np.random.randint(0, int(params.volume_size)))
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


class NetworkMonitorWindow:
    def __init__(self, volume_size: float = 10.0, fig_size: int = 5):
        """
        Initialize the Network Monitor Window.

        Args:
            volume_size (float): The size of the spatial volume.
            fig_size (int): The size of the matplotlib figure.
        """
        self.queue = queue.Queue()
        self.volume_size = volume_size
        self.fig_size = fig_size
        self.neuron_scatter = None
        self.drop_scatter = None
        self.path_line = None
        self.processed_drops = []
        self.path_steps = []
        self.lock = threading.Lock()

        # **Visualization Enabled Flag**
        self.visualization_enabled = True  # Default to enabled

        # Start window in separate thread
        self.thread = threading.Thread(target=self.create_window, daemon=True)
        self.thread.start()

    def create_window(self):
        """Create the window and its components"""
        try:
            self.root = tk.Tk()
            self.root.title("Network Health Monitor")
            self.root.geometry("800x800")  # Increased size to accommodate 3D plot
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

            # Configure main frame
            self.frame = ttk.Frame(self.root, padding="10")
            self.frame.pack(expand=True, fill=tk.BOTH)

            # Connectivity label
            self.connectivity_label = ttk.Label(self.frame, text="Network Connectivity: 0%")
            self.connectivity_label.pack(pady=5)

            # Progress bar
            self.progress = ttk.Progressbar(
                self.frame,
                length=400,
                mode='determinate',
                maximum=100
            )
            self.progress.pack(pady=5)

            # Status label
            self.status_label = ttk.Label(self.frame, text="Status: Initializing...")
            self.status_label.pack(pady=5)

            # **Visualization Toggle Slider**
            self.toggle_label = ttk.Label(self.frame, text="Enable Visualization")
            self.toggle_label.pack(pady=5)

            self.visualization_scale = ttk.Scale(
                self.frame,
                from_=0,
                to=1,
                orient='horizontal',
                command=self.toggle_visualization,
                length=200  # Adjust the length as needed
            )
            self.visualization_scale.set(1)  # Default to enabled
            self.visualization_scale.pack(pady=5)

            # Matplotlib Figure
            self.fig = plt.Figure(figsize=(6, 6), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(0, self.volume_size)
            self.ax.set_ylim(0, self.volume_size)
            self.ax.set_zlim(0, self.volume_size)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('3D Network Visualization')

            # Initialize scatter plots
            self.neuron_scatter = self.ax.scatter([], [], [], c=[], cmap='viridis', marker='o', s=20, label='Neurons')
            self.drop_scatter = self.ax.scatter([], [], [], c='red', marker='^', s=50, label='Drops')
            self.path_line, = self.ax.plot([], [], [], c='blue', linewidth=2, label='Path')

            self.ax.legend(loc='upper right')

            # Embed the matplotlib figure in Tkinter
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

            # Set up periodic queue check
            self.check_queue()

            # Start mainloop
            self.root.mainloop()

        except Exception as e:
            print(f"Error creating monitor window: {e}")

    def on_closing(self):
        """Handle window closing"""
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

    def toggle_visualization(self, value):
        """Handle visualization toggle slider movement"""
        try:
            if float(value) >= 0.5:
                self.visualization_enabled = True
                print("Visualization Enabled")
            else:
                self.visualization_enabled = False
                print("Visualization Disabled")
        except Exception as e:
            print(f"Error toggling visualization: {e}")

    def check_queue(self):
        """Check for updates in the queue"""
        try:
            while True:
                try:
                    message = self.queue.get_nowait()
                    self._handle_message(message)
                except queue.Empty:
                    break

            # Schedule next check
            try:
                self.root.after(100, self.check_queue)
            except:
                pass  # Window might be closed
        except Exception as e:
            print(f"Error checking queue: {e}")

    def update_connectivity(self, value: float):
        """Thread-safe method to update the connectivity display"""
        try:
            self.queue.put({'type': 'connectivity', 'value': value})
        except Exception as e:
            print(f"Error updating connectivity: {e}")

    def update_neuron_positions(self, neurons: Dict[int, Dict]):
        """Thread-safe method to update neuron positions"""
        try:
            self.queue.put({'type': 'neurons', 'data': neurons})
        except Exception as e:
            print(f"Error updating neuron positions: {e}")

    def update_drop_locations(self, drops: List[Position]):
        """Thread-safe method to update drop locations"""
        try:
            self.queue.put({'type': 'drops', 'data': drops})
        except Exception as e:
            print(f"Error updating drop locations: {e}")

    def update_path_step(self, position: Position):
        """Thread-safe method to update the path display"""
        try:
            self.queue.put({'type': 'path_step', 'position': position})
        except Exception as e:
            print(f"Error updating path step: {e}")

    def _handle_message(self, message: Dict):
        """Handle different types of messages"""
        message_type = message.get('type')
        if message_type == 'connectivity':
            self._update_connectivity_display(message.get('value'))
        elif message_type == 'neurons':
            self._update_neuron_positions_display(message.get('data'))
        elif message_type == 'drops':
            self._update_drop_locations_display(message.get('data'))
        elif message_type == 'path_step':
            self._update_path_display(message.get('position'))

    def _update_connectivity_display(self, value: float):
        """Update connectivity GUI elements"""
        try:
            self.connectivity_label.config(text=f"Network Connectivity: {value:.1f}%")
            self.progress['value'] = value

            # Update status text and color based on connectivity
            if value >= 70:
                status = "Good"
                color = "green"
            elif value >= 30:
                status = "Improving"
                color = "orange"
            else:
                status = "Poor"
                color = "red"

            self.status_label.config(text=f"Status: {status}", foreground=color)
        except Exception as e:
            print(f"Error updating connectivity display: {e}")

    def _update_neuron_positions_display(self, neurons: Dict[int, Dict]):
        """Update neuron positions in the 3D plot"""
        if not self.visualization_enabled:
            return  # **Respect the Visualization Toggle**

        try:
            xs = []
            ys = []
            zs = []
            colors = []
            for nid, data in neurons.items():
                pos = data['position']
                neuron_type = data['type']
                xs.append(pos[0])
                ys.append(pos[1])
                zs.append(pos[2])
                if neuron_type == NeuronType.INPUT:
                    colors.append('green')
                elif neuron_type == NeuronType.OUTPUT:
                    colors.append('blue')
                elif neuron_type == NeuronType.HIDDEN:
                    colors.append('purple')
                else:
                    colors.append('gray')  # Default color for unknown types

            self.neuron_scatter._offsets3d = (xs, ys, zs)
            self.neuron_scatter.set_color(colors)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating neuron positions display: {e}")

    def _update_drop_locations_display(self, drops: List[Position]):
        """Update drop locations in the 3D plot"""
        if not self.visualization_enabled:
            return  # **Respect the Visualization Toggle**

        try:
            xs = [drop.x for drop in drops]
            ys = [drop.y for drop in drops]
            zs = [drop.z for drop in drops]

            # Append new drops
            self.drop_scatter._offsets3d = (xs, ys, zs)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating drop locations display: {e}")

    def _update_path_display(self, position: Position):
        """Update the path visualization on the 3D plot"""
        if not self.visualization_enabled:
            return  # **Respect the Visualization Toggle**

        try:
            self.path_steps.append((position.x, position.y, position.z))
            if len(self.path_steps) > 1:
                xs, ys, zs = zip(*self.path_steps)
                self.path_line.set_data(xs, ys)
                self.path_line.set_3d_properties(zs)
                self.canvas.draw()
        except Exception as e:
            print(f"Error updating path display: {e}")


class NetworkEvolutionFitness:
    def __init__(
            self,
            network_params: Dict[str, any],
            update_best_func=None,
            max_path_length: int = 60,
            path_step_reward: float = 1.00,
            pickup_reward: float = 1.,
            successful_drop_reward: float = 4.,
            failed_drop_penalty: float = -2.,
            empty_bag_reward: float = 10.00,
            step_penalty: float = -100.00,
            debug: bool = False
    ):
        """
        Initialize Phase 1 fitness evaluation focusing on network connectivity structure.
        """
        # Centralized Reward, Penalty, and Parameter Definitions
        self.max_path_length = max_path_length
        self.path_step_reward = path_step_reward
        self.pickup_reward = pickup_reward
        self.successful_drop_reward = successful_drop_reward
        self.failed_drop_penalty = failed_drop_penalty
        self.empty_bag_reward = empty_bag_reward
        self.step_penalty = step_penalty

        # Debug and Update Function
        self.debug = debug
        self.update_best = update_best_func

        # Define genes for path evolution
        self.genes = ['U', 'D', 'F', 'B', 'L', 'R', 'DR']

        # Create network parameters using the provided network_params dictionary
        self.network_params = NetworkParameters(
            volume_size=network_params.get('volume_size', 10.0),
            num_input=network_params.get('num_input', 100),
            num_output=network_params.get('num_output', 4),
            total_neurons=network_params.get('total_neurons', 800),
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

        # Initialize the spatial neural network with the defined parameters
        self.network = create_network(self.network_params)

        # Initialize persistent path state
        self.pickup_bag = []
        self.current_pos = Position(0.0, 0.0, 0.0)  # Initial position at origin

        # Create monitor window first
        try:
            self.monitor = NetworkMonitorWindow(volume_size=self.network_params.volume_size)
            # Give the window a moment to initialize
            time.sleep(0.5)
            # Initialize path visualization with starting position
            if self.monitor.visualization_enabled:
                self.monitor.update_path_step(self.current_pos)
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not create monitor window: {e}")
            self.monitor = None

        # Initialize last connectivity score
        self._last_connectivity = 0

        # Start the background monitor update thread
        self._start_background_updates()

    def _start_background_updates(self, update_interval: float = 1.0):
        """
        Start a background thread to continuously update the monitor window
        with the full network state at regular intervals.
        """
        def update_loop():
            while True:
                try:
                    current_state = self.network.get_network_state()
                    if self.monitor:
                        # Only send updates if visualization is enabled
                        if self.monitor.visualization_enabled:
                            self.monitor.update_neuron_positions(current_state['neuron_positions'])
                except Exception as e:
                    if self.debug:
                        print(f"Error in background update loop: {e}")
                time.sleep(update_interval)  # Update every `update_interval` seconds

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()

    def calculate_connectivity_score(self) -> float:
        """Calculate network connectivity as a percentage (0-100)"""
        unreachable = self.network.compute_unreachable_neurons()
        total_neurons = len(self.network.neurons)
        if total_neurons == 0:
            connectivity = 0.0
        else:
            connectivity = (1.0 - (unreachable / total_neurons)) * 100

        # Update monitor if it exists and change is significant
        try:
            if self.monitor and abs(connectivity - self._last_connectivity) > 1:
                self._last_connectivity = connectivity
                self.monitor.update_connectivity(connectivity)
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not update monitor: {e}")

        return connectivity

    def execute_path(self, path: List[str]) -> Dict:
        """Execute movement path to modify network structure"""
        # Use a local set to track processed neurons within this evaluation
        processed_neurons = set()

        # Clear pickup bag at start of each path
        self.pickup_bag = []

        # Reset path in the monitor
        if self.monitor and self.monitor.visualization_enabled:
            self.monitor.path_steps = []
            self.monitor.update_path_step(self.current_pos)

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
        path_score = 0.0

        # Track when we pass the step limit
        step_limit_passed = False

        # Collect drop positions for visualization
        drop_positions = []

        for command in path:
            # Check if we're passing the step limit
            if total_steps == self.max_path_length:
                step_limit_passed = True
                if self.debug:
                    print(f"Step limit {self.max_path_length} passed - subsequent actions cost double")

            if command == 'DR' and self.pickup_bag:
                total_steps += 1
                current_pos_tuple = (
                    round(self.current_pos.x),
                    round(self.current_pos.y),
                    round(self.current_pos.z)
                )

                # Check if position is occupied by other neurons
                is_position_empty = not any(
                    tuple(round(x) for x in pos['position']) == current_pos_tuple
                    for nid, pos in neuron_positions.items()
                    if nid not in self.pickup_bag and nid not in position_updates and nid not in processed_neurons
                )

                if is_position_empty:
                    # Successful drop
                    neuron_id = self.pickup_bag.pop(0)
                    position_updates[neuron_id] = current_pos_tuple
                    neurons_moved += 1
                    successful_drops += 1

                    # Mark neuron as processed
                    processed_neurons.add(neuron_id)

                    # Record drop position
                    drop_pos = Position(*current_pos_tuple)
                    drop_positions.append(drop_pos)

                    if not step_limit_passed:
                        # Reward for successful drop within limit
                        path_score += self.successful_drop_reward
                    else:
                        # Penalty for successful drop beyond limit
                        path_score += self.failed_drop_penalty
                else:
                    # Failed drop
                    failed_drops += 1
                    path_score += self.failed_drop_penalty

            else:
                # Movement command
                new_pos = self.move(command)
                if new_pos:
                    total_steps += 1
                    moves_made += 1
                    self.current_pos = new_pos

                    if not step_limit_passed:
                        # Reward for move within limit
                        path_score += self.path_step_reward

                        # Check for pickups at current position
                        current_pos_tuple = (
                            round(self.current_pos.x),
                            round(self.current_pos.y),
                            round(self.current_pos.z)
                        )

                        # Only pick up hidden neurons that have not been processed
                        for nid, data in neuron_positions.items():
                            neuron = self.network.neurons[nid]
                            if (
                                    neuron.type == NeuronType.HIDDEN and
                                    nid not in self.pickup_bag and
                                    nid not in position_updates and
                                    nid not in processed_neurons
                            ):
                                neuron_pos = tuple(round(x) for x in data['position'])
                                if neuron_pos == current_pos_tuple:
                                    self.pickup_bag.append(nid)
                                    pickups_made += 1
                                    # Apply pickup reward
                                    path_score += self.pickup_reward
                                    break
                    else:
                        # Penalty for move beyond limit
                        path_score += self.step_penalty

                    # Send the new position to the monitor if visualization is enabled
                    if self.monitor and self.monitor.visualization_enabled:
                        self.monitor.update_path_step(self.current_pos)

                else:
                    if self.debug:
                        print(f"Invalid movement command: {command}")

            # **Conditional Sleeping Based on Visualization**
            if self.monitor and self.monitor.visualization_enabled:
                time.sleep(0.05)  # Adjust as needed

        # Apply reward for empty pickup bag at the end
        if not self.pickup_bag:
            path_score += self.empty_bag_reward
            if self.debug:
                print("Empty pickup bag reward applied.")
        else:
            if self.debug:
                print(f"Pickup bag not empty. Remaining neurons: {len(self.pickup_bag)}")

        # Send drop locations to the monitor if visualization is enabled
        if self.monitor and self.monitor.visualization_enabled and drop_positions:
            self.monitor.update_drop_locations(drop_positions)

        if position_updates:
            self.network.update_neuron_positions(position_updates)

        return {
            'path_score': path_score,
            'moves_made': moves_made,
            'pickups_made': pickups_made,
            'successful_drops': successful_drops,
            'failed_drops': failed_drops,
            'neurons_moved': neurons_moved,
            'total_neurons': len(self.network.neurons),
            'pickup_bag_size': len(self.pickup_bag),
            'total_steps': total_steps,
            'excess_steps': total_steps - self.max_path_length if step_limit_passed else 0
        }

    def compute(self, encoded_individual, ga_instance) -> float:
        """
        Phase 1 fitness computation with network visualization at 100% connectivity.
        """
        # Decode and execute path
        path = ga_instance.decode_organism(encoded_individual) if ga_instance else encoded_individual
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
                print(f"Path Length: {movement_results['total_steps']}")
                print(f"Failed Drops: {movement_results['failed_drops']}")

            # Exit the program since connectivity is achieved
            sys.exit(0)

        # Final fitness calculation
        fitness = abs(path_score) ** (connectivity_score * 0.001)

        if self.debug:
            print(f"\nFitness Calculation Details:")
            print(f"Path Length: {movement_results['total_steps']}")
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
            'U': (0, 0, 1),    # Up along Z-axis
            'D': (0, 0, -1),   # Down along Z-axis
            'F': (1, 0, 0),    # Forward along X-axis
            'B': (-1, 0, 0),   # Backward along X-axis
            'L': (0, 1, 0),    # Left along Y-axis
            'R': (0, -1, 0)    # Right along Y-axis
            # You can add more directions here for full 3D movement
        }

        if direction in moves:
            dx, dy, dz = moves[direction]
            new_x = (self.current_pos.x + dx) % self.network_params.volume_size
            new_y = (self.current_pos.y + dy) % self.network_params.volume_size
            new_z = (self.current_pos.z + dz) % self.network_params.volume_size
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
                'path_step_reward': self.path_step_reward,
                'pickup_reward': self.pickup_reward,
                'successful_drop_reward': self.successful_drop_reward,
                'failed_drop_penalty': self.failed_drop_penalty,
                'empty_bag_reward': self.empty_bag_reward,
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


def main():
    # Main Execution Block

    # Define consistent network parameters
    network_params = {
        'volume_size': 8.0,
        'num_input': 100,  # Align with fitness function
        'num_output': 4,  # Align with fitness function
        'total_neurons': 500,  # Align with fitness function
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
    print(f"Initial number of unreachable neurons: {unreachable_neurons}")

    # Get positions of hidden neurons
    hidden_positions = network.get_hidden_neuron_positions()

    # Example: Modify positions of hidden neurons (e.g., move them randomly)
    modified_positions = {}
    for neuron_id, pos in hidden_positions.items():
        dx = np.random.uniform(-1.0, 1.0)
        dy = np.random.uniform(-1.0, 1.0)
        dz = np.random.uniform(-1.0, 1.0)
        new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
        # Ensure new positions are within volume boundaries
        new_pos = (
            max(0.0, min(new_pos[0], params.volume_size)),
            max(0.0, min(new_pos[1], params.volume_size)),
            max(0.0, min(new_pos[2], params.volume_size))
        )
        modified_positions[neuron_id] = new_pos

    # Update hidden neuron positions in the network
    network.update_hidden_neuron_positions(modified_positions)

    # Recompute number of unreachable neurons after modification
    unreachable_neurons_after = network.compute_unreachable_neurons()
    print(f"Number of unreachable neurons after modification: {unreachable_neurons_after}")

    # Initialize the Network Evolution Fitness
    fitness_evaluator = NetworkEvolutionFitness(
        network_params=network_params,
        debug=True  # Set to True to enable debug messages
    )

    # Example: Run multiple fitness evaluations with sample paths
    sample_paths = [
        ['F', 'F', 'U', 'F', 'DR', 'B', 'D', 'DR'],
        ['U', 'U', 'F', 'DR', 'F', 'F', 'DR', 'B'],
        ['L', 'L', 'F', 'F', 'DR', 'R', 'D', 'DR'],
        ['F', 'D', 'F', 'F', 'DR', 'B', 'U', 'DR']
        # Add more paths as needed
    ]

    for path in sample_paths:
        fitness_score = fitness_evaluator.compute(path, ga_instance=None)
        print(f"Fitness Score for path {path}: {fitness_score}")

    # Keep the main thread alive to allow background updates
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting main program.")


if __name__ == "__main__":
    main()
