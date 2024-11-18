import threading
import time
import queue
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from SP_NN import Position, NeuronType


class NetworkMonitor:
    def __init__(self, volume_size: float = 10.0):
        """Initialize the Network Monitor Window."""
        self.queue = queue.Queue()
        self.volume_size = volume_size
        self.neuron_scatter = None
        self.drop_scatter = None
        self.path_line = None
        self.path_steps = []
        self.lock = threading.Lock()
        self.all_drops = []
        self.visualization_enabled = True
        self.last_position = None

        # Start window in separate thread
        self.thread = threading.Thread(target=self.create_window, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # Give window time to initialize

    def create_window(self):
        """Create the window and its components"""
        try:
            self.root = tk.Tk()
            self.root.title("Network Health Monitor")
            self.root.geometry("800x800")

            # Configure main frame
            self.frame = ttk.Frame(self.root, padding="10")
            self.frame.pack(expand=True, fill=tk.BOTH)

            # Connectivity display
            self.connectivity_frame = ttk.LabelFrame(self.frame, text="Network Status", padding="5")
            self.connectivity_frame.pack(fill=tk.X, pady=5)

            self.connectivity_label = ttk.Label(
                self.connectivity_frame,
                text="Network Connectivity: 0%",
                font=('Arial', 12, 'bold')
            )
            self.connectivity_label.pack(pady=5)

            self.progress = ttk.Progressbar(
                self.connectivity_frame,
                length=400,
                mode='determinate',
                maximum=100
            )
            self.progress.pack(pady=5)

            self.status_label = ttk.Label(
                self.connectivity_frame,
                text="Status: Initializing...",
                font=('Arial', 10)
            )
            self.status_label.pack(pady=5)

            # Visualization controls
            self.control_frame = ttk.LabelFrame(self.frame, text="Visualization Controls", padding="5")
            self.control_frame.pack(fill=tk.X, pady=5)

            self.visualization_var = tk.BooleanVar(value=True)
            self.visualization_check = ttk.Checkbutton(
                self.control_frame,
                text="Enable Real-time Updates",
                variable=self.visualization_var,
                command=self.toggle_visualization
            )
            self.visualization_check.pack(pady=5)

            # 3D Plot
            self.fig = plt.Figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')

            half_volume = self.volume_size / 2
            self.ax.set_xlim(-half_volume, half_volume)
            self.ax.set_ylim(-half_volume, half_volume)
            self.ax.set_zlim(-half_volume, half_volume)

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('Neural Network Structure')

            # Initialize plots
            self.neuron_scatter = self.ax.scatter([], [], [], c='b', marker='o', s=50)
            self.drop_scatter = self.ax.scatter([], [], [], c='r', marker='^', s=100)
            self.path_line, = self.ax.plot([], [], [], c='g', linewidth=2)

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

            # Start queue checker
            self.check_queue()
            self.root.mainloop()

        except Exception as e:
            print(f"Error creating monitor window: {e}")

    def check_queue(self):
        """Process messages from the queue"""
        try:
            while True:
                try:
                    message = self.queue.get_nowait()
                    self._handle_message(message)
                except queue.Empty:
                    break

            self.root.after(100, self.check_queue)
        except Exception as e:
            print(f"Error checking queue: {e}")

    def update_connectivity(self, value: float):
        """Update connectivity display - always process regardless of visualization state"""
        try:
            self.queue.put({'type': 'connectivity', 'value': value})
        except Exception as e:
            print(f"Error updating connectivity: {e}")

    def update_neuron_positions(self, neurons: dict):
        """Update neuron positions - only if visualization is enabled"""
        if self.visualization_enabled:
            try:
                self.queue.put({'type': 'neurons', 'data': neurons})
            except Exception as e:
                print(f"Error updating neurons: {e}")

    def update_drop_locations(self, drops: list):
        """Update drop locations - only if visualization is enabled"""
        if self.visualization_enabled:
            try:
                self.queue.put({'type': 'drops', 'data': drops})
            except Exception as e:
                print(f"Error updating drops: {e}")

    def update_path_step(self, position: Position):
        """Update path visualization - only if visualization is enabled"""
        if self.visualization_enabled:
            try:
                self.queue.put({'type': 'path_step', 'position': position})
            except Exception as e:
                print(f"Error updating path: {e}")

    def clear_drops(self):
        """Clear all drops"""
        if self.visualization_enabled:
            try:
                self.queue.put({'type': 'clear_drops'})
            except Exception as e:
                print(f"Error clearing drops: {e}")

    def clear_path(self):
        """Clear all path steps from visualization."""
        self.path_steps = []
        self.last_position = None
        if hasattr(self, 'path_line'):
            self.path_line.set_data([], [])
            self.path_line.set_3d_properties([])
            self.canvas.draw()

    def toggle_visualization(self):
        """Toggle real-time visualization updates"""
        self.visualization_enabled = self.visualization_var.get()
        if not self.visualization_enabled:
            # Clear visualization when disabled
            self._clear_visualization()

    def _handle_message(self, message: dict):
        """Process different message types"""
        try:
            message_type = message.get('type')

            # Always process connectivity updates
            if message_type == 'connectivity':
                self._update_connectivity_display(message.get('value'))
            # Only process visualization updates if enabled
            elif self.visualization_enabled:
                if message_type == 'neurons':
                    self._update_neuron_positions_display(message.get('data'))
                elif message_type == 'drops':
                    self._update_drop_locations_display(message.get('data'))
                elif message_type == 'path_step':
                    self._update_path_display(message.get('position'))
                elif message_type == 'clear_drops':
                    self._clear_drops_display()
                elif message_type == 'clear_visualization':
                    self._clear_visualization()

        except Exception as e:
            print(f"Error handling message: {e}")

    def _update_connectivity_display(self, value: float):
        """Update connectivity GUI elements"""
        try:
            self.connectivity_label.config(text=f"Network Connectivity: {value:.1f}%")
            self.progress['value'] = value

            if value >= 70:
                status, color = "Good", "green"
            elif value >= 30:
                status, color = "Improving", "orange"
            else:
                status, color = "Poor", "red"

            self.status_label.config(text=f"Status: {status}", foreground=color)
        except Exception as e:
            print(f"Error updating connectivity display: {e}")

    def _update_neuron_positions_display(self, neurons: dict):
        """Update neuron positions in 3D plot"""
        try:
            xs, ys, zs, colors = [], [], [], []
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
                    colors.append('gray')

            self.neuron_scatter._offsets3d = (xs, ys, zs)
            self.neuron_scatter.set_color(colors)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating neuron positions: {e}")

    def _update_drop_locations_display(self, drops: list):
        """Update drop locations in 3D plot"""
        try:
            for drop in drops:
                self.all_drops.append(drop)

            xs = [drop.x for drop in self.all_drops]
            ys = [drop.y for drop in self.all_drops]
            zs = [drop.z for drop in self.all_drops]

            self.drop_scatter._offsets3d = (xs, ys, zs)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating drop locations: {e}")

    def _update_path_display(self, position: Position):
        """Update path visualization"""
        try:
            new_pos = (position.x, position.y, position.z)
            self.path_steps.append(new_pos)
            self.last_position = new_pos

            if len(self.path_steps) > 1:
                xs, ys, zs = zip(*self.path_steps)
                self.path_line.set_data(xs, ys)
                self.path_line.set_3d_properties(zs)
                self.canvas.draw()
        except Exception as e:
            print(f"Error updating path display: {e}")

    def _clear_drops_display(self):
        """Clear all drops from 3D plot"""
        try:
            self.all_drops = []
            self.drop_scatter._offsets3d = ([], [], [])
            self.canvas.draw()
        except Exception as e:
            print(f"Error clearing drops: {e}")

    def _clear_visualization(self):
        """Clear all visualization elements but keep connectivity display"""
        try:
            if hasattr(self, 'neuron_scatter'):
                self.neuron_scatter._offsets3d = ([], [], [])
            if hasattr(self, 'drop_scatter'):
                self.drop_scatter._offsets3d = ([], [], [])
            if hasattr(self, 'path_line'):
                self.path_line.set_data([], [])
                self.path_line.set_3d_properties([])
            self.path_steps = []
            self.all_drops = []
            self.canvas.draw()
        except Exception as e:
            print(f"Error clearing visualization: {e}")