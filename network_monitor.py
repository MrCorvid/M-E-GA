import threading
import queue
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Dict, Optional
from dataclasses import dataclass

class NetworkMonitor:
    def __init__(self, volume_size: float = 10.0, fig_size: int = 5):
        """
        Initialize the Network Monitor Window.

        Args:
            volume_size (float): Size of the spatial volume.
            fig_size (int): Size of the matplotlib figure.
        """
        self.queue = queue.Queue()
        self.volume_size = volume_size
        self.fig_size = fig_size
        self.neuron_scatter = None
        self.drop_scatter = None
        self.path_line = None
        self.path_steps = []
        self.all_drops = []
        self.last_position = None
        self.lock = threading.Lock()

        # Visualization state
        self.visualization_enabled = True

        # Start window in separate thread
        self.thread = threading.Thread(target=self.create_window, daemon=True)
        self.thread.start()

    def create_window(self):
        """Create and configure the main window and visualization components."""
        try:
            self.root = tk.Tk()
            self.root.title("Network Health Monitor")
            self.root.geometry("800x800")
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

            # Main frame
            self.frame = ttk.Frame(self.root, padding="10")
            self.frame.pack(expand=True, fill=tk.BOTH)

            # Status indicators
            self.connectivity_label = ttk.Label(self.frame, text="Network Connectivity: 0%")
            self.connectivity_label.pack(pady=5)

            self.progress = ttk.Progressbar(
                self.frame,
                length=400,
                mode='determinate',
                maximum=100
            )
            self.progress.pack(pady=5)

            self.status_label = ttk.Label(self.frame, text="Status: Initializing...")
            self.status_label.pack(pady=5)

            # Visualization controls
            self.toggle_label = ttk.Label(self.frame, text="Enable Visualization")
            self.toggle_label.pack(pady=5)

            self.visualization_scale = ttk.Scale(
                self.frame,
                from_=0,
                to=1,
                orient='horizontal',
                command=self.toggle_visualization,
                length=200
            )
            self.visualization_scale.set(1)
            self.visualization_scale.pack(pady=5)

            # 3D Plot setup
            self.setup_3d_plot()

            # Queue checker
            self.check_queue()

            # Start mainloop
            self.root.mainloop()

        except Exception as e:
            print(f"Error creating monitor window: {e}")

    def setup_3d_plot(self):
        """Initialize the 3D matplotlib plot."""
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, self.volume_size)
        self.ax.set_ylim(0, self.volume_size)
        self.ax.set_zlim(0, self.volume_size)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Network Visualization')

        # Initialize plots
        self.neuron_scatter = self.ax.scatter([], [], [], c=[], cmap='viridis',
                                            marker='o', s=20, label='Neurons')
        self.drop_scatter = self.ax.scatter([], [], [], c='red',
                                          marker='^', s=50, label='Drops')
        self.path_line, = self.ax.plot([], [], [], c='blue',
                                      linewidth=2, label='Path')

        self.ax.legend(loc='upper right')

        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def check_queue(self):
        """Process updates from the message queue."""
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
                pass
        except Exception as e:
            print(f"Error checking queue: {e}")

    def _handle_message(self, message: Dict):
        """Process different types of update messages."""
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
        """Update connectivity indicators."""
        try:
            self.connectivity_label.config(text=f"Network Connectivity: {value:.1f}%")
            self.progress['value'] = value

            # Update status based on connectivity
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

    def _update_neuron_positions_display(self, neurons: Dict):
        """Update neuron positions in the 3D plot."""
        if not self.visualization_enabled:
            return

        try:
            xs, ys, zs, colors = [], [], [], []
            for nid, data in neurons.items():
                pos = data['position']
                neuron_type = data['type']
                xs.append(pos[0])
                ys.append(pos[1])
                zs.append(pos[2])

                # Color based on neuron type
                if neuron_type == "input":
                    colors.append('green')
                elif neuron_type == "output":
                    colors.append('blue')
                elif neuron_type == "hidden":
                    colors.append('purple')
                else:
                    colors.append('gray')

            self.neuron_scatter._offsets3d = (xs, ys, zs)
            self.neuron_scatter.set_color(colors)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating neuron positions: {e}")

    def _update_drop_locations_display(self, drops: List['Position']):
        """Update drop markers in the 3D plot."""
        if not self.visualization_enabled:
            return

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

    def _update_path_display(self, position: 'Position'):
        """Update the path visualization, handling toroidal wrapping."""
        if not self.visualization_enabled:
            return

        try:
            new_pos = (position.x, position.y, position.z)
            if self.last_position is not None:
                wrapped_segments = self._split_wrapped_path(
                    self.last_position, new_pos, self.volume_size)
                for pos in wrapped_segments[:-1]:
                    self.path_steps.append(pos)
                self.path_steps.append(new_pos)
            else:
                self.path_steps.append(new_pos)

            self.last_position = new_pos

            if len(self.path_steps) > 1:
                xs, ys, zs = zip(*self.path_steps)
                self.path_line.set_data(xs, ys)
                self.path_line.set_3d_properties(zs)
                self.canvas.draw()
        except Exception as e:
            print(f"Error updating path display: {e}")

    def _split_wrapped_path(self, last_pos: tuple, new_pos: tuple,
                           volume_size: float) -> List[tuple]:
        """Handle path visualization across volume boundaries."""
        wrapped_positions = []
        x1, y1, z1 = last_pos
        x2, y2, z2 = new_pos

        # Handle x-axis wrapping
        dx = x2 - x1
        if dx > volume_size / 2:
            wrapped_positions.extend([
                (volume_size, y1, z1),
                (0.0, y1, z1)
            ])
        elif dx < -volume_size / 2:
            wrapped_positions.extend([
                (0.0, y1, z1),
                (volume_size, y1, z1)
            ])

        # Handle y-axis wrapping
        dy = y2 - y1
        if dy > volume_size / 2:
            wrapped_positions.extend([
                (x1, volume_size, z1),
                (x1, 0.0, z1)
            ])
        elif dy < -volume_size / 2:
            wrapped_positions.extend([
                (x1, 0.0, z1),
                (x1, volume_size, z1)
            ])

        # Handle z-axis wrapping
        dz = z2 - z1
        if dz > volume_size / 2:
            wrapped_positions.extend([
                (x1, y1, volume_size),
                (x1, y1, 0.0)
            ])
        elif dz < -volume_size / 2:
            wrapped_positions.extend([
                (x1, y1, 0.0),
                (x1, y1, volume_size)
            ])

        wrapped_positions.append(new_pos)
        return wrapped_positions

    def _clear_drops_display(self):
        """Clear all drop markers from the visualization."""
        if not self.visualization_enabled:
            return

        try:
            self.all_drops = []
            self.drop_scatter._offsets3d = ([], [], [])
            self.canvas.draw()
        except Exception as e:
            print(f"Error clearing drops display: {e}")

    def toggle_visualization(self, value):
        """Toggle visualization on/off."""
        try:
            self.visualization_enabled = float(value) >= 0.5
            print(f"Visualization {'Enabled' if self.visualization_enabled else 'Disabled'}")
        except Exception as e:
            print(f"Error toggling visualization: {e}")

    def on_closing(self):
        """Handle window closing."""
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

    # Thread-safe update methods
    def update_connectivity(self, value: float):
        """Thread-safe method to update connectivity display."""
        self.queue.put({'type': 'connectivity', 'value': value})

    def update_neuron_positions(self, neurons: Dict):
        """Thread-safe method to update neuron positions."""
        self.queue.put({'type': 'neurons', 'data': neurons})

    def update_drop_locations(self, drops: List['Position']):
        """Thread-safe method to update drop locations."""
        self.queue.put({'type': 'drops', 'data': drops})

    def update_path_step(self, position: 'Position'):
        """Thread-safe method to update path visualization."""
        self.queue.put({'type': 'path_step', 'position': position})

    def clear_drops(self):
        """Thread-safe method to clear all drops."""
        self.queue.put({'type': 'clear_drops'})
