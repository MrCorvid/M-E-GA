from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from SP_NN import Position, NetworkParameters, create_network, NeuronType, plot_network_with_weights
import sys
import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


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

    def update_neuron_positions(self, neurons: Dict[str, Position]):
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

    def _update_neuron_positions_display(self, neurons: Dict[str, Position]):
        """Update neuron positions in the 3D plot"""
        try:
            xs = []
            ys = []
            zs = []
            colors = []
            for nid, pos in neurons.items():
                xs.append(pos.x)
                ys.append(pos.y)
                zs.append(pos.z)
                if pos.type == NeuronType.INPUT:
                    colors.append('green')
                elif pos.type == NeuronType.OUTPUT:
                    colors.append('blue')
                elif pos.type == NeuronType.HIDDEN:
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
            max_path_length: int = 100,
            path_step_reward: float = 1.00,
            pickup_reward: float = 1.,
            successful_drop_reward: float = 3.,
            failed_drop_penalty: float = -0.,
            empty_bag_reward: float = 10.00,
            step_penalty: float = -10.00,
            debug: bool = False
    ):
        """
        Initialize Phase 1 fitness evaluation focusing on network connectivity structure.
        """
        # Create plots directory if it doesn't exist
        self.plots_dir = "network_plots"
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

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
        self.genes = ['U', 'D', 'F', 'B', 'DR']

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
            hidden_radius_range=network_params.get('hidden_radius_range', (0.30, .60)),
            interface_offset=network_params.get('interface_offset', 1.0),
            activation_budget=network_params.get('activation_budget', 1000),
            time_window_size=network_params.get('time_window_size', 100),
            base_radius_shrink_rate=network_params.get('base_radius_shrink_rate', 0.95),
            activation_radius_factor=network_params.get('activation_radius_factor', 0.2),
            activation_threshold=network_params.get('activation_threshold', 0.5)
        )

        # Initialize the spatial neural network with the defined parameters
        self.network = create_network(self.network_params)

        # Plot and save initial network state
        self.save_network_plot("initial")

        # Initialize persistent path state
        self.pickup_bag = []
        self.current_pos = Position(0.0, 0.0, 0.0)  # Initial position at origin

        # Create monitor window first
        try:
            self.monitor = NetworkMonitorWindow(volume_size=self.network_params.volume_size)
            # Give the window a moment to initialize
            time.sleep(0.5)
            # Initialize path visualization with starting position
            self.monitor.update_path_step(self.current_pos)
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not create monitor window: {e}")
            self.monitor = None

        # Initialize last connectivity score
        self._last_connectivity = 0

        # Initialize a set to keep track of processed neurons (picked up and dropped)
        self.processed_neurons = set()

    def save_network_plot(self, stage: str):
        """Save a plot of the network at the specified stage"""
        plt.figure(figsize=(12, 8))
        plot_network_with_weights(self.network)
        plt.tight_layout()

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.plots_dir}/network_{stage}_{timestamp}.png"

        plt.savefig(filename)
        plt.close()

        if self.debug:
            print(f"Saved network plot: {filename}")

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
        # Reset position to origin at start of each path
        self.current_pos = Position(0.0, 0.0, 0.0)
        # Clear pickup bag at start of each path
        self.pickup_bag = []

        # Reset path in the monitor
        if self.monitor:
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
                    if nid not in self.pickup_bag and nid not in position_updates and nid not in self.processed_neurons
                )

                if is_position_empty:
                    # Successful drop
                    neuron_id = self.pickup_bag.pop(0)
                    position_updates[neuron_id] = current_pos_tuple
                    neurons_moved += 1
                    successful_drops += 1

                    # Mark neuron as processed
                    self.processed_neurons.add(neuron_id)

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
                                    nid not in self.processed_neurons
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

                    # Send the new position to the monitor
                    if self.monitor:
                        self.monitor.update_path_step(self.current_pos)

            # Optional: Introduce a small delay for real-time visualization
            if self.monitor:
                time.sleep(0.05)  # Adjust as needed

        # Apply reward for empty pickup bag at the end
        if not self.pickup_bag:
            path_score += self.empty_bag_reward
            if self.debug:
                print("Empty pickup bag reward applied.")
        else:
            if self.debug:
                print(f"Pickup bag not empty. Remaining neurons: {len(self.pickup_bag)}")

        # Send drop locations to the monitor
        if self.monitor and drop_positions:
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

            # Save final network plot before exiting
            self.save_network_plot("final")
            sys.exit(0)

        # Final fitness calculation
        fitness = abs(path_score) ** (connectivity_score * 0.02)

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
            'U': (0, 1, 0),
            'D': (0, -1, 0),
            'F': (1, 0, 0),
            'B': (-1, 0, 0)
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
