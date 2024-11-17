from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Tuple
from SP_NN import Position  # Import Position from original SP_NN


class NavigationSystem:
    def __init__(self, volume_size: float = 10.0):
        self.volume_size = volume_size
        self.current_pos = Position(0.0, 0.0, 0.0)  # Start at origin

        # Define valid movements
        self.valid_moves = ['U', 'D', 'F', 'B', 'L', 'R', 'DR']

        # Movement mapping - (dx, dy, dz)
        self.movement_map = {
            'U': (0, 0, 1),  # Up
            'D': (0, 0, -1),  # Down
            'F': (1, 0, 0),  # Forward
            'B': (-1, 0, 0),  # Backward
            'L': (0, 1, 0),  # Left
            'R': (0, -1, 0),  # Right
            'DR': (0, 0, 0)  # Drop (no movement)
        }

    def execute_movement(self, command: str) -> Position:
        """Execute a single movement command and return new position"""
        if command not in self.valid_moves:
            return self.current_pos

        if command == 'DR':
            return self.current_pos

        dx, dy, dz = self.movement_map[command]
        new_pos = self._calculate_wrapped_position(
            Position(
                self.current_pos.x + dx,
                self.current_pos.y + dy,
                self.current_pos.z + dz
            )
        )

        self.current_pos = new_pos
        return new_pos

    def validate_path(self, path: List[str]) -> bool:
        """Validate a sequence of movement commands"""
        return all(move in self.valid_moves for move in path)

    def calculate_path_metrics(self) -> Dict:
        """Calculate metrics for the current path"""
        return {
            'current_position': (self.current_pos.x, self.current_pos.y, self.current_pos.z),
            'distance_from_origin': self._distance_from_origin(),
            'is_at_boundary': self._is_at_boundary()
        }

    def is_valid_position(self, position: Position) -> bool:
        """Check if a position is within volume bounds"""
        half_size = self.volume_size / 2
        return all(-half_size <= coord <= half_size for coord in
                   [position.x, position.y, position.z])

    def get_current_position(self) -> Position:
        """Return current position"""
        return self.current_pos

    def _calculate_wrapped_position(self, pos: Position) -> Position:
        """Handle toroidal wrapping at volume boundaries"""
        half_size = self.volume_size / 2

        # Apply wrapping with centered volume
        x = ((pos.x + half_size) % self.volume_size) - half_size
        y = ((pos.y + half_size) % self.volume_size) - half_size
        z = ((pos.z + half_size) % self.volume_size) - half_size

        return Position(x, y, z)

    def _distance_from_origin(self) -> float:
        """Calculate distance from current position to origin"""
        return self.current_pos.distance_to(Position(0.0, 0.0, 0.0))

    def _is_at_boundary(self) -> bool:
        """Check if current position is at volume boundary"""
        half_size = self.volume_size / 2
        threshold = 0.1  # Small threshold for float comparison

        return any(abs(abs(coord) - half_size) < threshold
                   for coord in [self.current_pos.x,
                                 self.current_pos.y,
                                 self.current_pos.z])