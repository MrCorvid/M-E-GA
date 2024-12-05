import threading
import time
import queue
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from dataclasses import dataclass


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
        self.show_connections = True
        self.last_position = None
        self.connection_collection = None

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

            # Network Health Metrics Frame
            self.health_frame = ttk.LabelFrame(self.frame, text="Network Health Metrics", padding="5")
            self.health_frame.pack(fill=tk.X, pady=5)

            # Structural Connectivity
            self.structural_label = ttk.Label(
                self.health_frame,
                text="Structural Connectivity: 0%",
                font=('Arial', 10)
            )
            self.structural_label.pack(pady=2)
            self.structural_progress = ttk.Progressbar(
                self.health_frame,
                length=400,
                mode='determinate',
                maximum=100
            )
            self.structural_progress.pack(pady=2)

            # Connection Density
            self.density_label = ttk.Label(
                self.health_frame,
                text="Connection Density: 0%",
                font=('Arial', 10)
            )
            self.density_label.pack(pady=2)
            self.density_progress = ttk.Progressbar(
                self.health_frame,
                length=400,
                mode='determinate',
                maximum=100
            )
            self.density_progress.pack(pady=2)

            # Combined Health
            self.combined_label = ttk.Label(
                self.health_frame,
                text="Combined Health: 0%",
                font=('Arial', 12, 'bold')
            )
            self.combined_label.pack(pady=2)
            self.combined_progress = ttk.Progressbar(
                self.health_frame,
                length=400,
                mode='determinate',
                maximum=100
            )
            self.combined_progress.pack(pady=2)

            # Status Label
            self.status_label = ttk.Label(
                self.health_frame,
                text="Status: Initializing...",
                font=('Arial', 10)
            )
            self.status_label.pack(pady=5)

            # Visualization controls
            self.control_frame = ttk.LabelFrame(self.frame, text="Visualization Controls", padding="5")
            self.control_frame.pack(fill=tk.X, pady=5)

            # Main visualization toggle
            self.visualization_var = tk.BooleanVar(value=True)
            self.visualization_check = ttk.Checkbutton(
                self.control_frame,
                text="Enable Real-time Updates",
                variable=self.visualization_var,
                command=self.toggle_visualization
            )
            self.visualization_check.pack(pady=2)

            # Connection toggle
            self.connections_var = tk.BooleanVar(value=True)
            self.connections_check = ttk.Checkbutton(
                self.control_frame,
                text="Show Network Connections",
                variable=self.connections_var,
                command=self.toggle_connections
            )
            self.connections_check.pack(pady=2)

            # Setup 3D plot
            self.setup_3d_plot()

            # Start queue checker
            self.check_queue()
            self.root.mainloop()

        except Exception as e:
            print(f"Error creating monitor window: {e}")

    def setup_3d_plot(self):
        """Setup the 3D visualization plot."""
        self.fig = plt.Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        half_volume = self.volume_size / 2
        self.ax.set_xlim(-half_volume, half_volume)
        self.ax.set_ylim(-half_volume, half_volume)
        self.ax.set_zlim(-half_volume, half_volume)

        # Disable axes, grid, and cube
        self.ax.axis('off')  # Turns off all axes (including labels, ticks, and cube)

        # Initialize plots with smaller neuron nodes
        self.neuron_scatter = self.ax.scatter([], [], [], c='b', marker='o', s=10)  # Smaller marker size
        self.drop_scatter = self.ax.scatter([], [], [], c='r', marker='^', s=30)  # Adjust drop size if needed
        self.path_line, = self.ax.plot([], [], [], c='g', linewidth=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

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

    def toggle_visualization(self):
        """Toggle real-time visualization updates"""
        self.visualization_enabled = self.visualization_var.get()
        if not self.visualization_enabled:
            self._clear_visualization()

    def toggle_connections(self):
        """Toggle network connection visibility"""
        self.show_connections = self.connections_var.get()
        if not self.show_connections:
            self._clear_connections()

    def update_neuron_positions(self, neurons: dict):
        """Update neuron positions and connections"""
        if self.visualization_enabled:
            try:
                self.queue.put({'type': 'neurons', 'data': neurons})
            except Exception as e:
                print(f"Error updating neurons: {e}")

    def update_health_metrics(self, metrics: dict):
        """Update all health metrics displays"""
        try:
            self.queue.put({'type': 'health_metrics', 'data': metrics})
        except Exception as e:
            print(f"Error updating health metrics: {e}")

    def update_drop_locations(self, drops: list):
        """Update drop locations"""
        if self.visualization_enabled:
            try:
                self.queue.put({'type': 'drops', 'data': drops})
            except Exception as e:
                print(f"Error updating drops: {e}")

    def update_path_step(self, position):
        """Update path visualization"""
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
        """Clear path visualization"""
        try:
            self.path_steps = []
            self.last_position = None
            if hasattr(self, 'path_line'):
                self.path_line.set_data_3d([0], [0], [0])
                self.canvas.draw()
        except Exception as e:
            print(f"Error clearing path: {e}")

    def _clear_connections(self):
        """Clear connection visualization without affecting other elements"""
        if self.connection_collection is not None:
            self.connection_collection.remove()
            self.connection_collection = None
            self.canvas.draw()

    def _handle_message(self, message: dict):
        """Process different message types"""
        try:
            message_type = message.get('type')

            if message_type == 'health_metrics':
                self._update_health_metrics_display(message.get('data'))
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

    def _update_health_metrics_display(self, metrics: dict):
        """Update the display for all health metrics"""
        try:
            structural = metrics.get('structural_connectivity', 0)
            density = metrics.get('connection_density', 0)
            combined = metrics.get('combined_health', 0)

            # Update structural connectivity
            self.structural_label.config(text=f"Structural Connectivity: {structural:.1f}%")
            self.structural_progress['value'] = structural

            # Update connection density
            self.density_label.config(text=f"Connection Density: {density:.1f}%")
            self.density_progress['value'] = density

            # Update combined health
            self.combined_label.config(text=f"Combined Health: {combined:.1f}%")
            self.combined_progress['value'] = combined

            # Update status based on combined health
            if combined >= 70:
                status, color = "Good", "green"
            elif combined >= 30:
                status, color = "Improving", "orange"
            else:
                status, color = "Poor", "red"

            self.status_label.config(text=f"Status: {status}", foreground=color)

        except Exception as e:
            print(f"Error updating health metrics display: {e}")

    def _update_neuron_positions_display(self, neurons: dict):
        """Update neuron positions and connections in 3D plot."""
        try:
            positions = {}
            xs, ys, zs = [], [], []

            for nid, data in neurons.items():
                pos = data['position']
                if isinstance(pos, (tuple, list)):
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                else:
                    x, y, z = float(pos.x), float(pos.y), float(pos.z)

                xs.append(x)
                ys.append(y)
                zs.append(z)
                positions[nid] = (x, y, z)

            self.neuron_scatter._offsets3d = (xs, ys, zs)

            if self.show_connections:
                lines = []
                for nid, data in neurons.items():
                    if 'connections' in data:
                        start = positions[nid]
                        for target_id in data['connections']:
                            if target_id in positions:
                                end = positions[target_id]
                                lines.append([start, end])

                # Update or create Line3DCollection
                if self.connection_collection is None:
                    self.connection_collection = Line3DCollection(
                        lines, colors='gray', linewidths=0.5, alpha=0.3
                    )
                    self.ax.add_collection3d(self.connection_collection)
                else:
                    self.connection_collection.set_segments(lines)

            self.canvas.draw()
        except Exception as e:
            print(f"Error updating neuron positions: {e}")

    def _update_drop_locations_display(self, drops: list):
        """Update drop locations in 3D plot"""
        try:
            for drop in drops:
                if hasattr(drop, 'x') and hasattr(drop, 'y') and hasattr(drop, 'z'):
                    self.all_drops.append(drop)

            if self.all_drops:
                xs = [float(drop.x) for drop in self.all_drops]
                ys = [float(drop.y) for drop in self.all_drops]
                zs = [float(drop.z) for drop in self.all_drops]
                self.drop_scatter._offsets3d = (xs, ys, zs)
                self.canvas.draw()

        except Exception as e:
            print(f"Error updating drop locations: {e}")

    def _update_path_display(self, position):
        """Update path visualization"""
        try:
            if hasattr(position, 'x') and hasattr(position, 'y') and hasattr(position, 'z'):
                # Handle Position object
                self.path_steps.append((float(position.x), float(position.y), float(position.z)))
            else:
                # Handle tuple/list
                self.path_steps.append((float(position[0]), float(position[1]), float(position[2])))

            if len(self.path_steps) > 1:
                xs, ys, zs = zip(*self.path_steps)
                self.path_line.set_data_3d(xs, ys, zs)  # Use set_data_3d instead of separate calls
                self.canvas.draw()

        except Exception as e:
            print(f"Error updating path display: {e}")

    def _clear_drops_display(self):
        """Clear all drops from 3D plot"""
        try:
            self.all_drops = []
            # Create empty arrays explicitly for all three dimensions
            empty_drops = np.array([[0, 0, 0]])  # Single point that won't be visible
            self.drop_scatter._offsets3d = (empty_drops[:, 0],
                                            empty_drops[:, 1],
                                            empty_drops[:, 2])
            self.canvas.draw()
        except Exception as e:
            print(f"Error clearing drops: {e}")

    def _clear_visualization(self):
        """Clear all visualization elements"""
        try:
            if hasattr(self, 'neuron_scatter'):
                self.neuron_scatter._offsets3d = ([], [], [])
            if hasattr(self, 'drop_scatter'):
                self.drop_scatter._offsets3d = ([], [], [])
            if hasattr(self, 'path_line'):
                self.path_line.set_data([], [])
                self.path_line.set_3d_properties([])
            self._clear_connections()
            self.path_steps = []
            self.all_drops = []
            self.canvas.draw()
        except Exception as e:
            print(f"Error clearing visualization: {e}")
