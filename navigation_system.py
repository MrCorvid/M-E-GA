from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class Position:
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return np.sqrt((self.x - other.x) ** 2 + 
                      (self.y - other.y) ** 2 + 
                      (self.z - other.z) ** 2)

class NavigationSystem:
    """
    Handles movement and position tracking in 3D space with toroidal wrapping.
    """
    
    def __init__(self, volume_size: float = 10.0):
        self.volume_size = volume_size
        self.current_pos = Position(0.0, 0.0, 0.0)  # Start at origin
        
        # Define valid movement commands and their corresponding deltas
        self.movement_map = {
            'U': (0, 0, 1),    # Up (positive z)
            'D': (0, 0, -1),   # Down (negative z)
            'F': (1, 0, 0),    # Forward (positive x)
            'B': (-1, 0, 0),   # Backward (negative x)
            'L': (0, 1, 0),    # Left (positive y)
            'R': (0, -1, 0),   # Right (negative y)
            'DR': (0, 0, 0)    # Drop (no movement)
        }
        
        self.valid_moves = list(self.movement_map.keys())

    def execute_movement(self, command: str) -> Optional[Position]:
        """
        Execute a single movement command and return the new position.
        
        Args:
            command: Movement command string ('U', 'D', 'F', 'B', 'L', 'R', 'DR')
            
        Returns:
            New Position after movement, or None if command is invalid
        """
        if command not in self.movement_map:
            return None
            
        if command == 'DR':
            return self.current_pos
            
        dx, dy, dz = self.movement_map[command]
        new_pos = Position(
            x=(self.current_pos.x + dx) % self.volume_size,
            y=(self.current_pos.y + dy) % self.volume_size,
            z=(self.current_pos.z + dz) % self.volume_size
        )
        
        self.current_pos = new_pos
        return new_pos

    def validate_path(self, path: List[str]) -> bool:
        """
        Validate a sequence of movement commands.
        
        Args:
            path: List of movement command strings
            
        Returns:
            True if all commands are valid, False otherwise
        """
        return all(cmd in self.valid_moves for cmd in path)

    def calculate_path_metrics(self, path: List[str]) -> Dict:
        """
        Calculate metrics for a movement path.
        
        Args:
            path: List of movement command strings
            
        Returns:
            Dictionary containing path metrics:
                - total_distance: Total Euclidean distance traveled
                - num_drops: Number of drop commands
                - num_moves: Number of movement commands
                - is_valid: Whether the path is valid
        """
        if not self.validate_path(path):
            return {
                'total_distance': 0.0,
                'num_drops': 0,
                'num_moves': 0,
                'is_valid': False
            }
            
        total_distance = 0.0
        num_drops = 0
        num_moves = 0
        current = Position(0.0, 0.0, 0.0)
        
        for cmd in path:
            if cmd == 'DR':
                num_drops += 1
                continue
                
            dx, dy, dz = self.movement_map[cmd]
            next_pos = Position(
                x=(current.x + dx) % self.volume_size,
                y=(current.y + dy) % self.volume_size,
                z=(current.z + dz) % self.volume_size
            )
            
            # Calculate shortest distance considering wrapping
            dx = min((next_pos.x - current.x) % self.volume_size,
                    (current.x - next_pos.x) % self.volume_size)
            dy = min((next_pos.y - current.y) % self.volume_size,
                    (current.y - next_pos.y) % self.volume_size)
            dz = min((next_pos.z - current.z) % self.volume_size,
                    (current.z - next_pos.z) % self.volume_size)
                    
            total_distance += np.sqrt(dx*dx + dy*dy + dz*dz)
            num_moves += 1
            current = next_pos
            
        return {
            'total_distance': total_distance,
            'num_drops': num_drops,
            'num_moves': num_moves,
            'is_valid': True
        }

    def is_valid_position(self, position: Position) -> bool:
        """
        Check if a position is within the valid volume bounds.
        
        Args:
            position: Position to check
            
        Returns:
            True if position is valid, False otherwise
        """
        return (0 <= position.x < self.volume_size and
                0 <= position.y < self.volume_size and
                0 <= position.z < self.volume_size)

    def get_current_position(self) -> Position:
        """Get the current position."""
        return self.current_pos

    def _calculate_wrapped_position(self, pos: Position) -> Position:
        """
        Calculate the wrapped position ensuring coordinates are within bounds.
        
        Args:
            pos: Position to wrap
            
        Returns:
            New Position with wrapped coordinates
        """
        return Position(
            x=pos.x % self.volume_size,
            y=pos.y % self.volume_size,
            z=pos.z % self.volume_size
        )
