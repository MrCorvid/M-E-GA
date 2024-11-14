# network_evolution_fitness.py

import numpy as np
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
import itertools



class NavigationSystem:
    """
    Navigation system for precise movement in discrete 3D space.
    
    The system processes a genome sequence into movement commands using a structured
    binary format. Each command consists of:
    - Command type (4 bits)
    - Selector bit (1 bit)
    - Magnitude (6 or 8 bits, based on selector)
    """

    def __init__(self, network_params):
        """
        Initialize navigation system.
        
        Args:
            network_params: Configuration parameters including volume_size
        """
        self.network_params = network_params
        self.current_heading = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        
        # Command patterns (4-bit identifiers)
        self.COMMANDS = {
            0b1111: 'X-HEADING',  # Rotate around X axis
            0b1110: 'Y-HEADING',  # Rotate around Y axis
            0b1100: 'Z-HEADING',  # Rotate around Z axis
            0b1000: 'MOVE',       # Move in current heading direction
            0b1001: 'DROP'        # Drop carried neuron
        }

    def stream_genome_to_binary(self, genome: List[int]) -> Iterator[int]:
        """
        Convert genome to binary byte stream.
        
        Args:
            genome: List of integers (0-9)
            
        Yields:
            Sequence of bytes (0-255) representing the genome
        """
        # Join digits and convert to integer
        numeric_str = ''.join(map(str, genome))
        numeric_value = int(numeric_str)
        
        # Convert to binary string and pad to byte alignment
        binary_str = bin(numeric_value)[2:]
        padding = (8 - len(binary_str) % 8) % 8
        binary_str = '0' * padding + binary_str
        
        # Yield each byte
        for i in range(0, len(binary_str), 8):
            yield int(binary_str[i:i+8], 2)

    def process_binary_stream(self, binary_stream: Iterator[int]) -> Iterator[Dict]:
        """
        Process binary stream into navigation commands.
        
        Args:
            binary_stream: Iterator yielding bytes (0-255)
            
        Yields:
            Dict containing command type, magnitude, and selector
        """
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
                            # Convert magnitude bits to value
                            magnitude = 0
                            for mag_bit in current_command['magnitude_bits']:
                                magnitude = (magnitude << 1) | mag_bit
                                
                            yield {
                                'type': current_command['type'],
                                'magnitude': magnitude,
                                'selector': current_command['selector']
                            }
                            current_command = None

        # Handle final command if complete
        if current_command and current_command.get('selector_read'):
            if len(current_command['magnitude_bits']) >= current_command['required_bits']:
                magnitude = 0
                for mag_bit in current_command['magnitude_bits'][:current_command['required_bits']]:
                    magnitude = (magnitude << 1) | mag_bit
                    
                yield {
                    'type': current_command['type'],
                    'magnitude': magnitude,
                    'selector': current_command['selector']
                }

    def _scale_magnitude_to_distance(self, magnitude: int, selector: int) -> float:
        """
        Scale magnitude to movement distance.
        
        Args:
            magnitude: Raw magnitude value
            selector: Selector bit (0=6-bit, 1=8-bit magnitude)
            
        Returns:
            Scaled distance value
        """
        max_value = 255 if selector else 63  # 8 or 6 bits
        max_distance = self.network_params.volume_size / 4
        return (magnitude / max_value) * max_distance

    def _normalize_angle(self, magnitude: int, selector: int) -> float:
        """
        Convert magnitude to rotation angle.
        
        Args:
            magnitude: Raw magnitude value
            selector: Selector bit (0=6-bit, 1=8-bit magnitude)
            
        Returns:
            Angle in degrees (0-360)
        """
        max_value = 255 if selector else 63  # 8 or 6 bits
        return (magnitude / max_value) * 360.0

    def update_heading(self, axis: str, angle_deg: float):
        """
        Update heading vector with rotation.
        
        Args:
            axis: Rotation axis ('X', 'Y', or 'Z')
            angle_deg: Rotation angle in degrees
        """
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        
        # Create rotation matrix based on axis
        if axis == 'X':
            rotation = np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ], dtype=np.float64)
        elif axis == 'Y':
            rotation = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ], dtype=np.float64)
        else:  # Z-axis
            rotation = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            
        # Apply rotation and normalize
        self.current_heading = np.dot(rotation, self.current_heading)
        norm = np.linalg.norm(self.current_heading)
        if norm > 0:
            self.current_heading /= norm

    def get_movement_vector(self, distance: float) -> np.ndarray:
        """
        Get movement vector for given distance.
        
        Args:
            distance: Distance to move
            
        Returns:
            3D vector representing movement
        """
        return self.current_heading * distance


class NetworkMonitorWindow:
    def __init__(self, volume_size: float = 35.0, fig_size: int = 5, agent_radius: float = 0.5):
        """Initialize Network Monitor Window with agent radius"""
        self.queue = queue.Queue()
        self.volume_size = volume_size
        self.half_volume = volume_size / 2
        self.fig_size = fig_size
        self.agent_radius = agent_radius
        self.neuron_scatter = None
        self.drop_scatter = None
        self.path_line = None
        self.processed_drops = []
        self.path_steps = []
        self.lock = threading.Lock()
        self.all_drops = []
        self.visualization_enabled = True
        self.last_position = None
        self.root = None

        # Start window in separate thread
        self.thread = threading.Thread(target=self.create_window, daemon=True)
        self.thread.start()

        # Wait for window to be created
        while self.root is None:
            time.sleep(0.1)

    def normalize_position(self, pos):
        """Normalize position to stay within centered volume bounds"""
        x = ((pos[0] + self.half_volume) % self.volume_size) - self.half_volume
        y = ((pos[1] + self.half_volume) % self.volume_size) - self.half_volume
        z = ((pos[2] + self.half_volume) % self.volume_size) - self.half_volume
        return (x, y, z)

    def get_shortest_path(self, pos1, pos2):
        """Get shortest path between two points in centered toroidal space"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]

        # Adjust for wrapping in centered space
        if dx > self.half_volume:
            dx -= self.volume_size
        elif dx < -self.half_volume:
            dx += self.volume_size

        if dy > self.half_volume:
            dy -= self.volume_size
        elif dy < -self.half_volume:
            dy += self.volume_size

        if dz > self.half_volume:
            dz -= self.volume_size
        elif dz < -self.half_volume:
            dz += self.volume_size

        return dx, dy, dz

    def create_window(self):
        """Create the window and its components"""
        try:
            self.root = tk.Tk()
            self.root.title("Network Health Monitor")
            self.root.geometry("800x800")
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

            # Visualization Toggle Button
            self.toggle_button = ttk.Button(
                self.frame,
                text="Visualization: Enabled",
                command=self.toggle_visualization
            )
            self.toggle_button.pack(pady=5)

            # Matplotlib Figure
            self.fig = plt.Figure(figsize=(6, 6), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='3d')

            # Set fixed limits centered around origin
            self.ax.set_xlim(-self.half_volume, self.half_volume)
            self.ax.set_ylim(-self.half_volume, self.half_volume)
            self.ax.set_zlim(-self.half_volume, self.half_volume)

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('3D Network Visualization (Centered Toroidal Space)')

            # Add grid
            self.ax.grid(True)

            # Initialize scatter plots
            self.neuron_scatter = self.ax.scatter([], [], [], c=[], cmap='viridis', marker='o', s=20, label='Neurons')
            self.drop_scatter = self.ax.scatter([], [], [], c='red', marker='^', s=50, label='Drops')
            self.path_line, = self.ax.plot([], [], [], c='blue', linewidth=2, label='Path')

            # Add radius visualization for agent at origin
            phi = np.linspace(0, 2 * np.pi, 32)
            theta = np.linspace(0, np.pi, 16)
            phi, theta = np.meshgrid(phi, theta)

            x = self.agent_radius * np.sin(theta) * np.cos(phi)
            y = self.agent_radius * np.sin(theta) * np.sin(phi)
            z = self.agent_radius * np.cos(theta)

            self.agent_sphere = self.ax.plot_surface(x, y, z, alpha=0.2, color='blue')

            self.ax.legend(loc='upper right')

            # Embed matplotlib figure
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

            # Schedule the first queue check
            self.root.after(100, self.check_queue)

            # Start mainloop
            self.root.mainloop()

        except Exception as e:
            print(f"Error creating monitor window: {e}")
            if self.root:
                self.root.quit()

    def toggle_visualization(self):
        """Toggle visualization state"""
        self.visualization_enabled = not self.visualization_enabled
        button_text = "Visualization: Enabled" if self.visualization_enabled else "Visualization: Disabled"
        self.toggle_button.config(text=button_text)
        print(f"Visualization {'Enabled' if self.visualization_enabled else 'Disabled'}")

    def check_queue(self):
        """Check for updates in the queue"""
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()
                self._handle_message(message)
                self.queue.task_done()

            if self.root and self.root.winfo_exists():
                self.root.after(100, self.check_queue)

        except Exception as e:
            print(f"Error checking queue: {e}")

    def update_connectivity(self, value: float):
        """Thread-safe method to update the connectivity display"""
        if self.root and self.root.winfo_exists():
            self.queue.put({'type': 'connectivity', 'value': value})

    def update_neuron_positions(self, neurons: Dict[int, Dict]):
        """Thread-safe method to update neuron positions"""
        if self.root and self.root.winfo_exists():
            self.queue.put({'type': 'neurons', 'data': neurons})

    def update_drop_locations(self, drops: List[Position]):
        """Thread-safe method to update drop locations"""
        if self.root and self.root.winfo_exists():
            self.queue.put({'type': 'drops', 'data': drops})

    def update_path_step(self, position: Position):
        """Thread-safe method to update the path display"""
        if self.root and self.root.winfo_exists():
            self.queue.put({'type': 'path_step', 'position': position})

    def clear_drops(self):
        """Thread-safe method to clear all drops"""
        if self.root and self.root.winfo_exists():
            self.queue.put({'type': 'clear_drops'})

    def _handle_message(self, message: Dict):
        """Handle different types of messages"""
        try:
            message_type = message.get('type')

            if not self.root or not self.root.winfo_exists():
                return

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

        except Exception as e:
            print(f"Error handling message: {e}")

    def _update_connectivity_display(self, value: float):
        """Update connectivity GUI elements"""
        if not self.visualization_enabled:
            return

        try:
            self.connectivity_label.config(text=f"Network Connectivity: {value:.1f}%")
            self.progress['value'] = value

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
            return
    
        try:
            xs = []
            ys = []
            zs = []
            colors = []
    
            for nid, data in neurons.items():
                pos = data['position']
                neuron_type = data['type']
    
                # Normalize position to centered volume bounds
                norm_pos = self.normalize_position(pos)
    
                xs.append(norm_pos[0])
                ys.append(norm_pos[1])
                zs.append(norm_pos[2])
    
                if neuron_type == NeuronType.INPUT.value:
                    colors.append('green')
                elif neuron_type == NeuronType.OUTPUT.value:
                    colors.append('blue')
                elif neuron_type == NeuronType.HIDDEN.value:
                    colors.append('purple')
                else:
                    colors.append('gray')
    
            # Remove the existing scatter plot
            self.neuron_scatter.remove()
    
            # Create a new scatter plot with updated data
            self.neuron_scatter = self.ax.scatter(
                xs, ys, zs, c=colors, cmap='viridis', marker='o', s=20, label='Neurons'
            )
    
            # Re-add the legend to include the new scatter plot
            self.ax.legend(loc='upper right')
    
            # Redraw the canvas
            self.canvas.draw_idle()
    
        except Exception as e:
            print(f"Error updating neuron positions display: {e}")
            
    def _update_path_display(self, position: Position):
        """Update the path visualization with proper toroidal wrapping"""
        if not self.visualization_enabled:
            return

        try:
            # Normalize the new position to centered space
            new_pos = self.normalize_position((position.x, position.y, position.z))

            # If this is the first point, just add it
            if not self.path_steps:
                self.path_steps.append(new_pos)
                self.last_position = new_pos
                return

            # Get the previous position
            prev_pos = self.path_steps[-1]

            # Calculate the shortest path between points considering wrapping
            dx, dy, dz = self.get_shortest_path(prev_pos, new_pos)

            # If we detect a wrap (distance greater than agent_radius * 2),
            # add intermediate points to show the wrapping
            if abs(dx) > self.agent_radius * 2 or abs(dy) > self.agent_radius * 2 or abs(dz) > self.agent_radius * 2:
                # Add exit point near boundary
                exit_point = (
                    prev_pos[0] + (0.9 * dx),
                    prev_pos[1] + (0.9 * dy),
                    prev_pos[2] + (0.9 * dz)
                )
                self.path_steps.append(exit_point)

                # Add entry point on opposite boundary
                entry_point = (
                    new_pos[0] - (0.9 * dx),
                    new_pos[1] - (0.9 * dy),
                    new_pos[2] - (0.9 * dz)
                )
                self.path_steps.append(entry_point)

            # Add the new position
            self.path_steps.append(new_pos)
            self.last_position = new_pos

            # Update path visualization
            if len(self.path_steps) > 1:
                # Convert path_steps to arrays for plotting
                xs, ys, zs = zip(*self.path_steps)

                # Create continuous line plot
                self.path_line.set_data(xs, ys)
                self.path_line.set_3d_properties(zs)

                # Optional: Add dots at wrapping points for clarity
                wrap_points = [p for i, p in enumerate(self.path_steps[1:], 1)
                               if any(abs(p[j] - self.path_steps[i - 1][j]) > self.agent_radius * 2
                                      for j in range(3))]
                if wrap_points:
                    wrap_xs, wrap_ys, wrap_zs = zip(*wrap_points)
                    if not hasattr(self, 'wrap_points_scatter'):
                        self.wrap_points_scatter = self.ax.scatter([], [], [],
                                                                   c='red', marker='o', s=30, alpha=0.5)
                    self.wrap_points_scatter._offsets3d = (wrap_xs, wrap_ys, wrap_zs)

                # Update display
                self.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating path display: {e}")

    def _update_drop_locations_display(self, drops: List[Position]):
        """Update drop locations in the 3D plot"""
        if not self.visualization_enabled:
            return

        try:
            for drop in drops:
                # Normalize drop position to centered volume bounds
                norm_pos = self.normalize_position((drop.x, drop.y, drop.z))
                self.all_drops.append(Position(*norm_pos))

            xs = [drop.x for drop in self.all_drops]
            ys = [drop.y for drop in self.all_drops]
            zs = [drop.z for drop in self.all_drops]

            self.drop_scatter._offsets3d = (xs, ys, zs)
            self.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating drop locations display: {e}")

    def _clear_drops_display(self):
        """Clear all drops from the 3D plot"""
        if not self.visualization_enabled:
            return

        try:
            self.all_drops = []
            self.drop_scatter._offsets3d = ([], [], [])
            self.canvas.draw_idle()

        except Exception as e:
            print(f"Error clearing drops display: {e}")

    def on_closing(self):
        """Handle window closing"""
        try:
            if self.root:
                self.root.quit()
                self.root.destroy()
        except:
            pass



class NetworkEvolutionFitness:
   def __init__(
           self,
           config: Dict[str, any],
           update_best_func: Optional[Callable[[any, float], None]] = None,
           debug: bool = False
   ):
       # Extract network parameters from config
       network_params = config.get('network_params', {})

       # Initialize NetworkParameters
       self.network_params = NetworkParameters(
           volume_size=network_params.get('volume_size', 35.0),
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
       self.pickup_reward = config.get('pickup_reward', 0.5)
       self.successful_drop_reward = config.get('successful_drop_reward', 5.0)
       self.failed_drop_penalty = config.get('failed_drop_penalty', -0.0)
       self.max_path_length = config.get('max_path_length', 1000.0)
       self.path_length_factor = config.get('path_length_factor', 0.01)

       # Initialize navigation system
       self.navigation = NavigationSystem(self.network_params)

       # Debug and Update Function
       self.debug = debug
       self.update_best = update_best_func

       # Initialize state variables
       self.current_pos = Position(0.0, 0.0, 0.0)
       self.pickup_bag = []  # FIFO queue for picked-up neurons
       self.metrics = {
           'path_distance': 0.0,
           'pickups': 0,
           'successful_drops': 0,
           'failed_drops': 0,
           'path_score': 0.0,
           'path_length_reward': 0.0
       }
       self.position_updates = {}
       self.processed_neurons = set()
       self.agent_radius = 0.5

       # Create monitor window
       try:
           self.monitor = NetworkMonitorWindow(volume_size=self.network_params.volume_size)
           time.sleep(0.5)
           if hasattr(self.monitor, 'visualization_enabled'):
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
       """Start a background thread to continuously update the monitor window."""
       def update_loop():
           while True:
               try:
                   current_state = self.network.get_network_state()
                   if self.monitor and hasattr(self.monitor, 'visualization_enabled'):
                       if self.monitor.visualization_enabled:
                           self.monitor.update_neuron_positions(current_state['neuron_positions'])
               except Exception as e:
                   if self.debug:
                       print(f"Error in background update loop: {e}")
               time.sleep(update_interval)

       update_thread = threading.Thread(target=update_loop, daemon=True)
       update_thread.start()

   def check_for_pickups(self, neuron_positions):
       """Check for pickups at current position - collects all valid neurons within range"""
       current_pos_tuple = (
           round(self.current_pos.x),
           round(self.current_pos.y),
           round(self.current_pos.z)
       )

       for nid, data in neuron_positions.items():
           neuron = self.network.neurons[nid]
           if (
               neuron.type == NeuronType.HIDDEN and
               nid not in self.pickup_bag and
               nid not in self.position_updates and
               nid not in self.processed_neurons
           ):
               neuron_pos = tuple(round(x) for x in data['position'])
               if self.positions_overlap(neuron_pos, current_pos_tuple):
                   # Add to pickup bag (FIFO queue)
                   self.pickup_bag.append(nid)
                   self.metrics['pickups'] += 1
                   self.metrics['path_score'] += self.pickup_reward
                   
                   if self.debug:
                       print(f"[PICKUP] Neuron {nid} at {neuron_pos}, Bag size: {len(self.pickup_bag)}")

   def attempt_drop(self, neuron_positions):
       """Attempt to drop the oldest picked up neuron (FIFO order)"""
       if not self.pickup_bag:
           return

       drop_pos = self.calculate_drop_position()
       pos_tuple = (round(drop_pos.x), round(drop_pos.y), round(drop_pos.z))

       # Check if position is clear
       is_clear = not any(
           self.positions_overlap(pos_tuple, tuple(round(x) for x in pos['position']))
           for nid, pos in neuron_positions.items()
           if nid not in self.pickup_bag and nid not in self.position_updates
       )

       if is_clear:
           neuron_id = self.pickup_bag.pop(0)  # FIFO - drop oldest neuron
           self.position_updates[neuron_id] = pos_tuple
           self.processed_neurons.add(neuron_id)
           self.metrics['successful_drops'] += 1
           self.metrics['path_score'] += self.successful_drop_reward

           if self.debug:
               print(f"[DROP] Success - Neuron {neuron_id} at {pos_tuple}, Remaining: {len(self.pickup_bag)}")

           # Visualization update
           if self.monitor and hasattr(self.monitor, 'visualization_enabled'):
               if self.monitor.visualization_enabled:
                   self.monitor.update_drop_locations([Position(*pos_tuple)])
                   time.sleep(0.05)
       else:
           self.metrics['failed_drops'] += 1
           self.metrics['path_score'] += self.failed_drop_penalty
           
           if self.debug:
               print(f"[DROP] Failed at {pos_tuple}, Bag size: {len(self.pickup_bag)}")

   def positions_overlap(self, pos1: tuple, pos2: tuple) -> bool:
       """Check if two positions overlap considering agent radius"""
       dx = abs(pos1[0] - pos2[0])
       dy = abs(pos1[1] - pos2[1])
       dz = abs(pos1[2] - pos2[2])

       # Adjust for wrapping
       volume_size = self.network_params.volume_size
       dx = min(dx, volume_size - dx)
       dy = min(dy, volume_size - dy)
       dz = min(dz, volume_size - dz)

       distance = np.sqrt(dx * dx + dy * dy + dz * dz)
       return distance <= self.agent_radius

   def calculate_drop_position(self) -> Position:
       """Calculate drop position based on current position"""
       return Position(
           round(self.current_pos.x),
           round(self.current_pos.y),
           round(self.current_pos.z)
       )

   def calculate_connectivity_score(self) -> float:
        """Calculate network connectivity as a percentage (0-100)"""
        unreachable_percentage = self.network.compute_unreachable_neurons()
        total_neurons = len(self.network.neurons)
        
        if total_neurons == 0:
            connectivity = 0.0
        else:
            connectivity = 100.0 - unreachable_percentage
        
        try:
            if self.monitor and abs(connectivity - self._last_connectivity) > 1:
                self._last_connectivity = connectivity
                self.monitor.update_connectivity(connectivity)
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not update monitor: {e}")
        
        return connectivity

   def process_genome(self, genome: List[int]) -> float:
       """Process genome using navigation system"""
       # Reset state
       self.metrics = {
           'path_distance': 0.0,
           'pickups': 0,
           'successful_drops': 0,
           'failed_drops': 0,
           'path_score': 0.0,
           'path_length_reward': 0.0
       }
       self.position_updates = {}
       self.processed_neurons = set()
       self.current_pos = Position(0.0, 0.0, 0.0)
       self.navigation.current_heading = np.array([1.0, 0.0, 0.0], dtype=np.float64)
       self.pickup_bag = []

       if self.debug:
           print("\n[GENOME] Starting genome processing")

       # Reset visualization
       if self.monitor:
           self.monitor.path_steps = []
           self.monitor.update_path_step(self.current_pos)
           self.monitor.clear_drops()

       # Get network state for neuron operations
       neuron_positions = self.network.get_network_state()['neuron_positions']

       # Process genome through navigation system
       binary_stream = self.navigation.stream_genome_to_binary(genome)
       command_stream = self.navigation.process_binary_stream(binary_stream)

       # Execute commands
       for command in command_stream:
           cmd_type = command['type']
           magnitude = command['magnitude']
           selector = command['selector']

           if cmd_type.endswith('HEADING'):
               angle = self.navigation._normalize_angle(magnitude, selector)
               self.navigation.update_heading(cmd_type[0], angle)
               if self.debug:
                   print(f"[NAV] {cmd_type} rotation: {angle:.2f}°")

           elif cmd_type == 'MOVE':
               distance = self.navigation._scale_magnitude_to_distance(magnitude, selector)
               if self.debug:
                   print(f"[NAV] Moving distance: {distance:.2f}")

               start_pos = (self.current_pos.x, self.current_pos.y, self.current_pos.z)
               
               # Update position with wrapping
               self.current_pos = Position(
                   (self.current_pos.x + self.navigation.current_heading[0] * distance) % self.network_params.volume_size,
                   (self.current_pos.y + self.navigation.current_heading[1] * distance) % self.network_params.volume_size,
                   (self.current_pos.z + self.navigation.current_heading[2] * distance) % self.network_params.volume_size
               )

               # Update metrics and check for pickups
               self.metrics['path_distance'] += distance
               self.check_for_pickups(neuron_positions)

               # Visualization update
               if self.monitor and hasattr(self.monitor, 'visualization_enabled'):
                   if self.monitor.visualization_enabled:
                       self.monitor.update_path_step(self.current_pos)
                       time.sleep(0.05)

           elif cmd_type == 'DROP' and self.pickup_bag:
               if self.debug:
                   print(f"[DROP] Attempting drop with {len(self.pickup_bag)} neurons in bag")
               self.attempt_drop(neuron_positions)

       # Apply network updates and calculate fitness
       if self.position_updates:
           self.network.update_neuron_positions(self.position_updates)

       connectivity_score = self.calculate_connectivity_score()

       # Calculate path length reward/penalty
       path_distance = self.metrics['path_distance']
       if path_distance <= self.max_path_length:
           path_length_reward = path_distance * self.path_length_factor
       else:
           over_distance = path_distance - self.max_path_length
           path_length_reward = (self.max_path_length * self.path_length_factor) - (over_distance * self.path_length_factor)

       # Update metrics
       self.metrics['path_length_reward'] = path_length_reward

       # Final fitness calculation
       final_fitness = self.metrics['path_score'] + (connectivity_score * 2) + path_length_reward

       if connectivity_score == 100:
           if self.debug:
               print("\n[SUCCESS] Achieved 100% network connectivity")
               print(f"Final Stats:")
               print(f"Path Distance: {self.metrics['path_distance']:.2f}")
               print(f"Successful Drops: {self.metrics['successful_drops']}")
               print(f"Failed Drops: {self.metrics['failed_drops']}")
               print(f"Pickups: {self.metrics['pickups']}")
           sys.exit(0)

       if self.debug:
           print(f"\n[FITNESS] Calculation Details:")
           print(f"Path Distance: {path_distance:.2f}")
           print(f"Path Length Reward: {path_length_reward:.2f}")
           print(f"Pickups: {self.metrics['pickups']}")
           print(f"Successful Drops: {self.metrics['successful_drops']}")
           print(f"Failed Drops: {self.metrics['failed_drops']}")
           print(f"Raw Path Score: {self.metrics['path_score']:.2f}")
           print(f"Connectivity Score: {connectivity_score}%")
           print(f"Combined Fitness: {final_fitness:.2f}")

       if self.update_best:
           self.update_best(genome, final_fitness)

       return final_fitness

   def compute(self, encoded_individual, ga_instance) -> float:
       """Process the genome and compute fitness"""
       return self.process_genome(encoded_individual)

   def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'connectivity': {
                'current': self.calculate_connectivity_score(),
                'unreachable_neurons': self.network.compute_unreachable_neurons()
            },
            'metrics': self.metrics,
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
                'current_position': (self.current_pos.x, self.current_pos.y, self.current_pos.z),
                'current_heading': self.navigation.current_heading
            }
        } 
