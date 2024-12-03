from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from SP_NN import Position


class NavigationSystem:
    # Movement commands
    UP = 'U'
    DOWN = 'D'
    FORWARD = 'F'
    BACK = 'B'
    LEFT = 'L'
    RIGHT = 'R'

    # Control commands
    SCALE_UP = 'SU'
    SCALE_DOWN = 'SD'
    TOGGLE_PICKUP = 'TP'
    DROP = 'DR'

    def __init__(self, volume_size: float = 10.0, base_radius: float = 0.50):
        self.volume_size = volume_size
        self.base_radius = base_radius

        # Navigation state
        self.current_pos = Position(0.0, 0.0, 0.0)
        self.scale = 1.0
        self.pickup_enabled = True
        self._current_heading = np.array([1.0, 0.0, 0.0])  # Default heading

        # Movement tracking
        self.command_history = []
        self.total_distance = 0.0
        self.all_positions = []
        self.move_positions = []
        self.moves_made = 0
        self.drops_made = 0

        # Movement vectors for each command
        self.movement_vectors = {
            self.UP: np.array([0, 0, 1]),
            self.DOWN: np.array([0, 0, -1]),
            self.FORWARD: np.array([1, 0, 0]),
            self.BACK: np.array([-1, 0, 0]),
            self.LEFT: np.array([0, 1, 0]),
            self.RIGHT: np.array([0, -1, 0])
        }

    def reset_position(self, start_position: Position = None):
        """Reset navigation state with optional starting position"""
        self.current_pos = start_position if start_position else Position(0.0, 0.0, 0.0)
        self.scale = 1.0
        self.pickup_enabled = True
        self._current_heading = np.array([1.0, 0.0, 0.0])
        self.command_history = []
        self.total_distance = 0.0
        self.all_positions = []
        self.move_positions = []
        self.moves_made = 0
        self.drops_made = 0

    def _update_scale(self, command: str):
        """Update movement scale based on command"""
        if command == self.SCALE_UP:
            self.scale *= 2.0
        elif command == self.SCALE_DOWN:
            self.scale *= 0.5

    def _move(self, command: str) -> Position:
        """Execute movement command and update heading"""
        if command in self.movement_vectors:
            # Update heading to movement direction
            self._current_heading = self.movement_vectors[command]

            # Calculate movement
            movement = self.movement_vectors[command] * self.scale
            new_pos = Position(
                x=self.current_pos.x + movement[0],
                y=self.current_pos.y + movement[1],
                z=self.current_pos.z + movement[2]
            )
            self.current_pos = self._wrap_position(new_pos)
            return self.current_pos
        return self.current_pos

    def _wrap_position(self, pos: Position) -> Position:
        """Implement toroidal wrapping"""
        half_size = self.volume_size / 2

        def wrap_coordinate(coord: float) -> float:
            return ((coord + half_size) % self.volume_size) - half_size

        return Position(
            x=wrap_coordinate(pos.x),
            y=wrap_coordinate(pos.y),
            z=wrap_coordinate(pos.z)
        )

    def _calculate_wrapped_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate the shortest distance between two points in toroidal space"""
        half_size = self.volume_size / 2

        def delta(a: float, b: float) -> float:
            direct = abs(a - b)
            wrapped = self.volume_size - direct
            return min(direct, wrapped)

        dx = delta(pos1.x, pos2.x)
        dy = delta(pos1.y, pos2.y)
        dz = delta(pos1.z, pos2.z)

        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def _update_path_metrics(self, new_pos: Position, is_move: bool = False):
        """Update path tracking metrics"""
        if self.all_positions:
            last_pos = self.all_positions[-1]
            dist = self._calculate_wrapped_distance(new_pos, last_pos)
            if is_move:
                self.total_distance += dist
                self.move_positions.append(new_pos)

        self.all_positions.append(new_pos)

    def execute_movement_sequence(self, commands: List[str]) -> Dict:
        """Execute a sequence of movement commands"""
        start_position = self.current_pos
        self._update_path_metrics(self.current_pos)
        command_positions = []

        for i, cmd in enumerate(commands):
            current_pos_index = len(self.all_positions) - 1

            if cmd in self.movement_vectors:
                new_pos = self._move(cmd)
                self._update_path_metrics(new_pos, is_move=True)
                self.moves_made += 1
                command_positions.append(len(self.all_positions) - 1)
                self.command_history.append((cmd, self.scale))

            elif cmd in [self.SCALE_UP, self.SCALE_DOWN]:
                self._update_scale(cmd)
                self._update_path_metrics(self.current_pos)
                command_positions.append(current_pos_index)
                self.command_history.append((cmd, self.scale))

            elif cmd == self.TOGGLE_PICKUP:
                self.pickup_enabled = not self.pickup_enabled
                self._update_path_metrics(self.current_pos)
                command_positions.append(current_pos_index)
                self.command_history.append((cmd, 0))

            elif cmd == self.DROP:
                self._update_path_metrics(self.current_pos)
                self.drops_made += 1
                command_positions.append(current_pos_index)
                self.command_history.append((cmd, 0))

        return {
            'positions': self.all_positions,
            'move_positions': self.move_positions,
            'path_length': self.total_distance,
            'moves_made': self.moves_made,
            'drops_made': self.drops_made,
            'commands_executed': len(self.command_history),
            'command_history': self.command_history,
            'command_positions': command_positions,
            'start_position': start_position,
            'final_position': self.current_pos,
        }

    def get_current_position(self) -> Position:
        """Get current position"""
        return self.current_pos

    @property
    def heading(self) -> np.ndarray:
        """Return current heading vector"""
        return self._current_heading

    @property
    def agent_radius(self) -> float:
        """Return the scaled interaction radius"""
        return self.base_radius * self.scale