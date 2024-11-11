# network_evolution_fitness.py

from typing import List, Dict, Optional, Callable
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
            if self.last_position is not None:
                wrapped_segments = self.split_wrapped_path(self.last_position, new_pos, self.volume_size)
                for pos in wrapped_segments[:-1]:
                    self.path_steps.append(pos)
                self.path_steps.append(new_pos)
            else:
                self.path_steps.append(new_pos)

            # Update the last_position
            self.last_position = new_pos

            # Update the path_line with new path_steps
            if len(self.path_steps) > 1:
                xs, ys, zs = zip(*self.path_steps)
                self.path_line.set_data(xs, ys)
                self.path_line.set_3d_properties(zs)
                self.canvas.draw()
        except Exception as e:
            print(f"Error updating path display: {e}")

    def split_wrapped_path(self, last_pos: tuple[float, float, float], new_pos: tuple[float, float, float],
                           volume_size: float) -> list[tuple[float, float, float]]:
        """
        Split the path into segments that handle wrapping in toroidal space.

        Args:
            last_pos (Tuple[float, float, float]): The last position (x, y, z).
            new_pos (Tuple[float, float, float]): The new position (x, y, z).
            volume_size (float): The size of the spatial volume.

        Returns:
            List[Tuple[float, float, float]]: A list of positions including intermediate boundary points.
        """
        wrapped_positions = []
        intermediate_positions = []
        x1, y1, z1 = last_pos
        x2, y2, z2 = new_pos

        # Handle wrapping in x
        dx = x2 - x1
        if dx > volume_size / 2:
            # Wrapped around negative side
            intermediate_positions.append((volume_size, y1, z1))
            intermediate_positions.append((0.0, y1, z1))
        elif dx < -volume_size / 2:
            # Wrapped around positive side
            intermediate_positions.append((0.0, y1, z1))
            intermediate_positions.append((volume_size, y1, z1))

        # Handle wrapping in y
        dy = y2 - y1
        if dy > volume_size / 2:
            # Wrapped around negative side
            intermediate_positions.append((x1, volume_size, z1))
            intermediate_positions.append((x1, 0.0, z1))
        elif dy < -volume_size / 2:
            # Wrapped around positive side
            intermediate_positions.append((x1, 0.0, z1))
            intermediate_positions.append((x1, volume_size, z1))

        # Handle wrapping in z
        dz = z2 - z1
        if dz > volume_size / 2:
            # Wrapped around negative side
            intermediate_positions.append((x1, y1, volume_size))
            intermediate_positions.append((x1, y1, 0.0))
        elif dz < -volume_size / 2:
            # Wrapped around positive side
            intermediate_positions.append((x1, y1, 0.0))
            intermediate_positions.append((x1, y1, volume_size))

        # Insert intermediate positions
        for pos in intermediate_positions:
            wrapped_positions.append(pos)

        # Finally, append the new position
        wrapped_positions.append(new_pos)

        return wrapped_positions

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
            connectivity = unreachable

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
        # **Clear existing drops from previous evaluation**
        if self.monitor:
            self.monitor.clear_drops()

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

                    # **Send the drop position immediately to the monitor**
                    drop_pos = Position(*current_pos_tuple)
                    if self.monitor and self.monitor.visualization_enabled:
                        self.monitor.update_drop_locations([drop_pos])

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

        Args:
            encoded_individual: The encoded genome representing the path.
            ga_instance: The genetic algorithm instance.

        Returns:
            float: The calculated fitness score.
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

        # Final fitness calculation using configuration parameters
        fitness = (path_score * self.path_step_reward) + (connectivity_score * 2)

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
            'U': (0, 0, 1),  # Up along Z-axis
            'D': (0, 0, -1),  # Down along Z-axis
            'F': (1, 0, 0),  # Forward along X-axis
            'B': (-1, 0, 0),  # Backward along X-axis
            'L': (0, 1, 0),  # Left along Y-axis
            'R': (0, -1, 0)  # Right along Y-axis
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
