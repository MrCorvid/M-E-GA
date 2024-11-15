# network_evolution_fitness.py

from typing import List, Dict, Optional, Callable, Iterator
from SP_NN import (
    create_network,
    SpatialNeuralNetwork,
    NetworkParameters,
    Position,
    NeuronType
)
import threading
import time
import sys
import queue
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from scipy.spatial import KDTree
import concurrent.futures
from threading import Lock, RLock


class NavigationSystem:
    def __init__(self, network_params):
        self.network_params = network_params
        self.current_heading = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        
        # Command patterns (4-bit identifiers)
        self.COMMANDS = {
            0b1111: 'X-HEADING',  # Rotate around X axis
            0b1110: 'Y-HEADING',  # Rotate around Y axis
            0b1100: 'Z-HEADING',  # Rotate around Z axis
            0b1000: 'MOVE',        # Move in current heading direction
            0b1001: 'DROP'         # Drop carried neuron
        }

    def stream_genome_to_binary(self, genome: List[int]) -> Iterator[int]:
        """
        Convert genome to binary byte stream.
        Example: genome [1,2,3,4] -> "1234" -> binary representation
        """
        # Join digits and convert to integer
        numeric_str = ''.join(map(str, genome))
        numeric_value = int(numeric_str)
        
        # Convert to binary string and pad to byte alignment
        binary_str = bin(numeric_value)[2:]  # Remove '0b' prefix
        padding = (8 - len(binary_str) % 8) % 8
        binary_str = '0' * padding + binary_str
        
        # Yield each byte
        for i in range(0, len(binary_str), 8):
            yield int(binary_str[i:i + 8], 2)

    def process_binary_stream(self, binary_stream: Iterator[int]) -> Iterator[Dict]:
        """Process binary stream into navigation commands."""
        command_buffer = 0
        bits_processed = 0
        current_command = None
        
        for byte in binary_stream:
            for bit_pos in range(7, -1, -1):
                bit = (byte >> bit_pos) & 1
                
                if current_command is None:
                    # Collect command bits (4 bits)
                    command_buffer = ((command_buffer << 1) | bit) & 0xF
                    bits_processed += 1
                    
                    if bits_processed == 4:
                        if command_buffer in self.COMMANDS:
                            current_command = {
                                'type': self.COMMANDS[command_buffer],
                                'magnitude_bits': [],
                                'selector_read': False
                            }
                        command_buffer = 0
                        bits_processed = 0
                        
                else:
                    if not current_command['selector_read']:
                        # Read selector bit
                        current_command['selector'] = bit
                        current_command['selector_read'] = True
                        current_command['required_bits'] = 8 if bit else 6
                    else:
                        # Collect magnitude bits
                        current_command['magnitude_bits'].append(bit)
                        
                        if len(current_command['magnitude_bits']) == current_command['required_bits']:
                            magnitude = 0
                            for mag_bit in current_command['magnitude_bits']:
                                magnitude = (magnitude << 1) | mag_bit
                            
                            yield {
                                'type': current_command['type'],
                                'magnitude': magnitude,
                                'selector': current_command['selector']
                            }
                            current_command = None

    def _normalize_angle(self, magnitude: int, selector: int) -> float:
        """
        Normalize the angle based on magnitude and selector.
        For simplicity, assume magnitude is scaled to a maximum of 360 degrees.
        """
        max_magnitude = (1 << 8) - 1 if selector else (1 << 6) - 1
        return (magnitude / max_magnitude) * 360.0

    def _scale_magnitude_to_distance(self, magnitude: int, selector: int) -> float:
        """
        Scale the magnitude to a distance value.
        For simplicity, assume magnitude scales to a maximum of 1 unit.
        """
        max_magnitude = (1 << 8) - 1 if selector else (1 << 6) - 1
        return (magnitude / max_magnitude) * 1.0

    def update_heading(self, axis: str, angle: float):
        """
        Update the current heading vector based on rotation around the specified axis.
        """
        radians = np.deg2rad(angle)
        if axis == 'X':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(radians), -np.sin(radians)],
                [0, np.sin(radians), np.cos(radians)]
            ])
        elif axis == 'Y':
            rotation_matrix = np.array([
                [np.cos(radians), 0, np.sin(radians)],
                [0, 1, 0],
                [-np.sin(radians), 0, np.cos(radians)]
            ])
        elif axis == 'Z':
            rotation_matrix = np.array([
                [np.cos(radians), -np.sin(radians), 0],
                [np.sin(radians), np.cos(radians), 0],
                [0, 0, 1]
            ])
        else:
            rotation_matrix = np.identity(3)
        
        self.current_heading = rotation_matrix.dot(self.current_heading)
        self.current_heading /= np.linalg.norm(self.current_heading)  # Normalize

    def get_movement_vector(self, distance: float) -> np.ndarray:
        """
        Get the movement vector based on the current heading and distance.
        """
        return self.current_heading * distance


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

        # **Add an attribute to store all drop positions**
        self.all_drops = []

        # **Visualization Enabled Flag**
        self.visualization_enabled = True  # Default to enabled

        # **Track the last position to handle wrapping**
        self.last_position = None

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
            half_volume = self.volume_size / 2
            self.ax.set_xlim(-half_volume, half_volume)
            self.ax.set_ylim(-half_volume, half_volume)
            self.ax.set_zlim(-half_volume, half_volume)
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

    def clear_drops(self):
        """Thread-safe method to clear all drops from the visualization"""
        try:
            self.queue.put({'type': 'clear_drops'})
        except Exception as e:
            print(f"Error clearing drops: {e}")

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
        elif message_type == 'clear_drops':
            self._clear_drops_display()

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
                if neuron_type == NeuronType.INPUT.value:
                    colors.append('green')
                elif neuron_type == NeuronType.OUTPUT.value:
                    colors.append('blue')
                elif neuron_type == NeuronType.HIDDEN.value:
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
            # Append new drops to the accumulated list
            for drop in drops:
                self.all_drops.append(drop)

            # Extract X, Y, Z coordinates from all accumulated drops
            xs = [drop.x for drop in self.all_drops]
            ys = [drop.y for drop in self.all_drops]
            zs = [drop.z for drop in self.all_drops]

            # Update the scatter plot with all drops
            self.drop_scatter._offsets3d = (xs, ys, zs)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating drop locations display: {e}")

    def _update_path_display(self, position: Position):
        """Update the path visualization on the 3D plot, handling toroidal wrapping."""
        if not self.visualization_enabled:
            return  # **Respect the Visualization Toggle**

        try:
            new_pos = (position.x, position.y, position.z)
            self.path_steps.append(new_pos)
            self.last_position = new_pos

            # Update the path_line with new path_steps
            if len(self.path_steps) > 1:
                xs, ys, zs = zip(*self.path_steps)
                self.path_line.set_data(xs, ys)
                self.path_line.set_3d_properties(zs)
                self.canvas.draw()
        except Exception as e:
            print(f"Error updating path display: {e}")

    def _clear_drops_display(self):
        """Clear all drops from the 3D plot"""
        if not self.visualization_enabled:
            return  # **Respect the Visualization Toggle**

        try:
            self.all_drops = []
            self.drop_scatter._offsets3d = ([], [], [])
            self.canvas.draw()
        except Exception as e:
            print(f"Error clearing drops display: {e}")


class NetworkEvolutionFitness:
    def __init__(
            self,
            config: Dict[str, any],
            update_best_func: Optional[Callable[[any, float], None]] = None,
            debug: bool = False
    ):
        """
        Initialize Phase 1 fitness evaluation focusing on network connectivity structure.

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
            volume_size=network_params.get('volume_size', 8.0),  # Match the volume size
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

        # Initialize the spatial neural network with the defined parameters
        self.network = create_network(self.network_params)

        # Initialize path reward parameters
        self.max_path_length = path_rewards.get('max_path_length', 60)
        self.path_step_reward = path_rewards.get('path_step_reward', 1.00)
        self.pickup_reward = path_rewards.get('pickup_reward', 0.5)
        self.successful_drop_reward = path_rewards.get('successful_drop_reward', 5.0)
        self.failed_drop_penalty = path_rewards.get('failed_drop_penalty', -0.0)
        self.empty_bag_reward = path_rewards.get('empty_bag_reward', 10.00)
        self.step_penalty = path_rewards.get('step_penalty', -10.0)

        # Debug and Update Function
        self.debug = debug
        self.update_best = update_best_func

        # Define genes for path evolution
        self.genes = ['U', 'D', 'F', 'B', 'L', 'R', 'DR']

        # Initialize persistent path state
        self.pickup_bag = []
        self.current_pos = Position(0.0, 0.0, 0.0)  # Initial position at origin

        # Initialize Navigation System
        self.navigation_system = NavigationSystem(self.network_params)

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
        connectivity = self.network.compute_unreachable_neurons()

        # Update monitor if it exists and change is significant
        try:
            if self.monitor and abs(connectivity - self._last_connectivity) > 1:
                self._last_connectivity = connectivity
                self.monitor.update_connectivity(connectivity)
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not update monitor: {e}")

        return connectivity

    def execute_path(self, genome: List[int]) -> Dict:
    """
    Execute movement path based on the provided genome using the NavigationSystem.

    Args:
        genome (List[int]): The genome representing the navigation commands.

    Returns:
        Dict: A dictionary containing execution metrics.
    """
    # Clear existing drops from previous evaluation
    if self.monitor:
        self.monitor.clear_drops()

    # Agent radius for neuron pickup
    self.agent_radius = 1.5  # Adjust this value as needed

    # Use NavigationSystem to process genome
    binary_stream = self.navigation_system.stream_genome_to_binary(genome)
    command_stream = self.navigation_system.process_binary_stream(binary_stream)

    # Reset metrics
    self.metrics = {
        'path_distance': 0.0,
        'pickups': 0,
        'successful_drops': 0,
        'failed_drops': 0,
        'path_score': 0.0,
        'path_length_reward': 0.0
    }

    # Clear pickup bag at start of each path
    self.pickup_bag = []

    # Reset current position to the origin at start of evaluation
    self.current_pos = Position(0.0, 0.0, 0.0)

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

    for command in command_stream:
        cmd_type = command['type']
        magnitude = command['magnitude']
        selector = command['selector']

        if total_steps >= self.max_path_length:
            step_limit_passed = True
            if self.debug:
                print(f"Step limit {self.max_path_length} passed - subsequent actions cost double")

        if cmd_type.endswith('HEADING'):
            angle = self.navigation_system._normalize_angle(magnitude, selector)
            axis = cmd_type[0]  # 'X', 'Y', or 'Z'
            self.navigation_system.update_heading(axis, angle)
            if self.debug:
                print(f"Rotated around {axis}-axis by {angle:.2f} degrees")

        elif cmd_type == 'MOVE':
            distance = self.navigation_system._scale_magnitude_to_distance(magnitude, selector)
            movement_vector = self.navigation_system.get_movement_vector(distance)

            # Update position with wrapping in toroidal space
            new_x = (self.current_pos.x + movement_vector[0]) % self.network_params.volume_size
            new_y = (self.current_pos.y + movement_vector[1]) % self.network_params.volume_size
            new_z = (self.current_pos.z + movement_vector[2]) % self.network_params.volume_size
            self.current_pos = Position(new_x, new_y, new_z)

            # Update metrics
            self.metrics['path_distance'] += distance
            moves_made += 1
            total_steps += 1

            if not step_limit_passed:
                # Reward for move within limit
                path_score += self.path_step_reward
            else:
                # Penalty for move beyond limit
                path_score += self.step_penalty

            # Check for pickups at current position
            for nid, data in neuron_positions.items():
                neuron = self.network.neurons[nid]
                if (
                    neuron.type == NeuronType.HIDDEN and
                    nid not in self.pickup_bag and
                    nid not in position_updates
                ):
                    # Calculate actual distance between agent and neuron
                    dx = self.current_pos.x - data['position'][0]
                    dy = self.current_pos.y - data['position'][1]
                    dz = self.current_pos.z - data['position'][2]
                    
                    # Calculate distance in toroidal space
                    dx = min(abs(dx), abs(dx - self.network_params.volume_size), abs(dx + self.network_params.volume_size))
                    dy = min(abs(dy), abs(dy - self.network_params.volume_size), abs(dy + self.network_params.volume_size))
                    dz = min(abs(dz), abs(dz - self.network_params.volume_size), abs(dz + self.network_params.volume_size))
                    
                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if distance <= self.agent_radius:
                        self.pickup_bag.append(nid)
                        pickups_made += 1
                        # Apply pickup reward
                        if not step_limit_passed:
                            path_score += self.pickup_reward
                        else:
                            path_score += self.step_penalty
                        if self.debug:
                            print(f"Picked up neuron {nid} at position {self.current_pos}")
                        break

            # Send the new position to the monitor if visualization is enabled
            if self.monitor and self.monitor.visualization_enabled:
                self.monitor.update_path_step(self.current_pos)

            # Conditional sleeping based on visualization
            if self.monitor and self.monitor.visualization_enabled:
                time.sleep(0.05)  # Adjust as needed

        elif cmd_type == 'DROP':
            if self.pickup_bag:
                neuron_id = self.pickup_bag.pop(0)
                
                # Calculate drop position at back edge of agent's radius
                # Get the opposite of current heading for back edge
                drop_heading = -self.navigation_system.current_heading
                
                # Calculate drop position by moving agent_radius distance in opposite heading
                drop_x = (self.current_pos.x + (drop_heading[0] * self.agent_radius)) % self.network_params.volume_size
                drop_y = (self.current_pos.y + (drop_heading[1] * self.agent_radius)) % self.network_params.volume_size
                drop_z = (self.current_pos.z + (drop_heading[2] * self.agent_radius)) % self.network_params.volume_size
                
                position_updates[neuron_id] = (drop_x, drop_y, drop_z)
                neurons_moved += 1
                successful_drops += 1

                # Send the drop position immediately to the monitor
                drop_pos = Position(drop_x, drop_y, drop_z)
                if self.monitor and self.monitor.visualization_enabled:
                    self.monitor.update_drop_locations([drop_pos])

                if not step_limit_passed:
                    # Reward for successful drop within limit
                    path_score += self.successful_drop_reward
                else:
                    # Penalty for successful drop beyond limit
                    path_score += self.failed_drop_penalty

                if self.debug:
                    print(f"Dropped neuron {neuron_id} at {position_updates[neuron_id]}")
            else:
                # Failed drop
                failed_drops += 1
                path_score += self.failed_drop_penalty
                if self.debug:
                    print("Failed to drop: No neurons to drop")

            total_steps += 1

            # Conditional sleeping based on visualization
            if self.monitor and self.monitor.visualization_enabled:
                time.sleep(0.05)  # Adjust as needed

    if position_updates:
        self.network.update_neuron_positions(position_updates)

    # Update metrics
    self.metrics.update({
        'pickups': pickups_made,
        'successful_drops': successful_drops,
        'failed_drops': failed_drops,
        'path_score': path_score
    })

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
    def process_genome(self, genome: List[int]) -> float:
        """
        Process the genome by executing the path, calculating connectivity, and computing fitness.

        Args:
            genome (List[int]): The genome representing the navigation commands.

        Returns:
            float: The calculated fitness score.
        """
        # Reset metrics for this run
        self.metrics = {
            'path_distance': 0.0,
            'pickups': 0,
            'successful_drops': 0,
            'failed_drops': 0,
            'path_score': 0.0,
            'path_length_reward': 0.0
        }

        # Execute path and get results
        movement_results = self.execute_path(genome)

        # After network updates are complete, calculate final connectivity
        connectivity_score = self.calculate_connectivity_score()

        # Update monitor with final state
        if self.monitor and hasattr(self.monitor, 'visualization_enabled'):
            if self.monitor.visualization_enabled:
                current_state = self.network.get_network_state()
                self.monitor.update_neuron_positions(current_state['neuron_positions'])

        # Calculate final fitness
        fitness = movement_results['path_score'] + connectivity_score

        return fitness

    def compute(self, encoded_individual, ga_instance) -> float:
        """
        Phase 1 fitness computation with network visualization at 100% connectivity.

        Args:
            encoded_individual: The encoded genome representing the path.
            ga_instance: The genetic algorithm instance.

        Returns:
            float: The calculated fitness score.
        """
        # Decode and execute path
        path = ga_instance.decode_organism(encoded_individual) if ga_instance else encoded_individual
        fitness = self.process_genome(path)

        # Check if we've achieved 100% connectivity
        connectivity_score = int(self.calculate_connectivity_score())
        if connectivity_score == 100:
            if self.debug:
                print("\nExiting Phase 1: Achieved 100% network connectivity")
                print(f"Final Stats:")
                print(f"Neurons Moved: {movement_results['neurons_moved']}")
                print(f"Path Length: {movement_results['total_steps']}")
                print(f"Failed Drops: {movement_results['failed_drops']}")

            # Exit the program since connectivity is achieved
            sys.exit(0)

        # Final fitness calculation using configuration parameters
        fitness = fitness + connectivity_score

        if self.debug:
            print(f"\nFitness Calculation Details:")
            print(f"Path Length: {movement_results['total_steps']}")
            print(f"Successful Drops: {movement_results['neurons_moved']}")
            print(f"Failed Drops: {movement_results['failed_drops']}")
            print(f"Raw Path Score: {movement_results['path_score']:.2f}")
            print(f"Connectivity Score: {connectivity_score}%")
            print(f"Combined Fitness: {fitness:.2f}")

        if self.update_best:
            self.update_best(encoded_individual, fitness)

        return fitness

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
