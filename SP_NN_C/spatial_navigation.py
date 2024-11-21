from SP_NN import Position
import numpy as np
from typing import List, Dict, Tuple
from collections import deque

class NavigationSystem:
    # Command patterns (4-bit)
    HEADING_X = 0b1111
    HEADING_Y = 0b1110
    HEADING_Z = 0b1100
    MOVE = 0b1000
    DROP = 0b1001

    # Adjustable movement parameters
    ROTATION_BITS = 12  # Adjust as needed
    ROTATION_RANGE = 360.0  # degrees

    MOVEMENT_BITS = 16
    MOVEMENT_RANGE = None  # Will be set based on volume_size

    def __init__(self, volume_size: float = 10.0):
        self.volume_size = volume_size
        self.MOVEMENT_RANGE = self.volume_size  # Directly use volume_size

        self.last_end_position = Position(0.0, 0.0, 0.0)
        self.current_pos = self.last_end_position
        self.heading = np.array([1.0, 0.0, 0.0])  # Initial heading along x-axis

        # Command processing state
        self.command_history = []

        # Path tracking
        self.total_distance = 0.0
        self.all_positions = []
        self.move_positions = []

        # Simple command counters
        self.moves_made = 0
        self.drops_made = 0
        self.rotations_made = 0

        # Constants for magnitude processing
        self.PRECISION = 1e-6

        # Command data lengths
        self.command_data_lengths = {
            self.MOVE: self.MOVEMENT_BITS,
            self.DROP: 0,
            self.HEADING_X: self.ROTATION_BITS,
            self.HEADING_Y: self.ROTATION_BITS,
            self.HEADING_Z: self.ROTATION_BITS,
        }

        # Command pattern lookup for more efficient pattern matching
        self.command_patterns = {
            self.HEADING_X: (self.HEADING_X, 'x'),
            self.HEADING_Y: (self.HEADING_Y, 'y'),
            self.HEADING_Z: (self.HEADING_Z, 'z'),
            self.MOVE: (self.MOVE, None),
            self.DROP: (self.DROP, None)
        }

    def create_bit_stream(self, numbers: List[int]) -> str:
        """Convert list of integers into binary string without chunking."""
        if not all(0 <= n <= 9 for n in numbers):
            raise ValueError("All numbers must be in range [0-9]")

        # Concatenate all numbers into a single string
        num_str = ''.join(str(num) for num in numbers)

        # Convert the concatenated string to an integer
        large_int = int(num_str)

        # Convert the integer to a binary string
        binary_str = bin(large_int)[2:]  # Remove '0b' prefix

        return binary_str

    def process_command_stream(self, binary_str: str) -> List[Tuple[int, List[str]]]:
        """Process binary string into commands and their data bits, discarding incomplete commands at the end."""
        commands = []
        command_stack = []
        window = []

        for bit in binary_str:
            window.append(bit)
            if len(window) > 4:
                window.pop(0)

            if len(window) == 4:
                pattern = int(''.join(window), 2)
                if pattern in self.command_data_lengths:
                    data_length = self.command_data_lengths[pattern]
                    new_command = {'command': pattern, 'data_bits': [], 'data_length': data_length}
                    command_stack.append(new_command)
                    window = []  # Reset window after detecting a command

            # Add bit to all commands in the stack
            for cmd in command_stack:
                cmd['data_bits'].append(bit)

            # Check if any command is complete
            while command_stack and len(command_stack[0]['data_bits']) >= command_stack[0]['data_length']:
                # Remove the command from the stack and append to commands
                complete_cmd = command_stack.pop(0)
                # Trim excess data bits if any
                complete_cmd['data_bits'] = complete_cmd['data_bits'][:complete_cmd['data_length']]
                commands.append((complete_cmd['command'], complete_cmd['data_bits']))

        return commands

    def calculate_magnitude(self, bits: List[str], command: int) -> float:
        """Calculate magnitude value from bits."""
        if not bits:
            return 0.0

        value = int(''.join(bits), 2)
        max_value = (1 << len(bits)) - 1
        normalized = value / max_value

        if command in [self.HEADING_X, self.HEADING_Y, self.HEADING_Z]:
            return normalized * self.ROTATION_RANGE
        elif command == self.MOVE:
            return normalized * self.MOVEMENT_RANGE
        else:
            return 0.0

    def execute_movement_sequence(self, numbers: List[int]) -> Dict:
        """Execute complete movement sequence from input numbers."""
        self.command_history = []
        self.total_distance = 0.0
        self.all_positions = []
        self.move_positions = []
        self.moves_made = 0
        self.drops_made = 0
        self.rotations_made = 0
        command_positions = []
        self.current_pos = self.last_end_position  # Start from last ending position
        self._update_path_metrics(self.current_pos)

        try:
            binary_str = self.create_bit_stream(numbers)
            command_data_pairs = self.process_command_stream(binary_str)

            for command, data_bits in command_data_pairs:
                magnitude = self.calculate_magnitude(data_bits, command)
                current_pos_index = len(self.all_positions) - 1

                cmd_info = self.command_patterns.get(command)
                if cmd_info:
                    cmd_type, axis = cmd_info
                    if axis:  # Heading command
                        self.update_heading(axis, magnitude)
                        self.rotations_made += 1
                        command_positions.append(current_pos_index)
                    elif cmd_type == self.MOVE:
                        new_pos = self.move(magnitude)
                        self._update_path_metrics(new_pos, is_move=True)
                        self.moves_made += 1
                        command_positions.append(len(self.all_positions) - 1)
                    elif cmd_type == self.DROP:
                        self._update_path_metrics(self.current_pos)
                        self.drops_made += 1
                        command_positions.append(len(self.all_positions) - 1)

                self.command_history.append((command, magnitude))

            # Store final position for next sequence
            self.last_end_position = self.current_pos

        except Exception as e:
            print(f"Error during movement sequence: {e}")
            return self._create_error_result()

        return {
            'positions': self.all_positions,
            'move_positions': self.move_positions,
            'path_length': self.total_distance,
            'moves_made': self.moves_made,
            'drops_made': self.drops_made,
            'rotations_made': self.rotations_made,
            'commands_executed': len(self.command_history),
            'command_history': self.command_history,
            'command_positions': command_positions,
            'final_position': self.current_pos,
        }

    def _update_path_metrics(self, new_pos: Position, is_move: bool = False) -> None:
        """Update path metrics with new position."""
        if self.all_positions:
            last_pos = self.all_positions[-1]
            dist = np.sqrt((new_pos.x - last_pos.x) ** 2 +
                           (new_pos.y - last_pos.y) ** 2 +
                           (new_pos.z - last_pos.z) ** 2)
            if is_move:
                self.total_distance += dist
                self.move_positions.append(new_pos)

        self.all_positions.append(new_pos)

    def _create_error_result(self) -> Dict:
        """Create a safe error result."""
        return {
            'positions': [self.current_pos],
            'move_positions': [],
            'path_length': 0.0,
            'moves_made': 0,
            'drops_made': 0,
            'rotations_made': 0,
            'commands_executed': 0,
            'command_history': [],
            'command_positions': [],
            'final_position': self.current_pos,
        }

    def update_heading(self, axis: str, angle: float):
        """Update heading vector based on axis rotation."""
        angle_rad = np.radians(angle % 360)

        if axis == 'x':
            rotation = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            rotation = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        else:  # z-axis
            rotation = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])

        self.heading = np.dot(rotation, self.heading)
        self.heading = self.heading / np.linalg.norm(self.heading)

    def move(self, distance: float) -> Position:
        """Move along current heading vector without constraints."""
        movement = self.heading * distance
        new_pos = Position(
            x=self.current_pos.x + movement[0],
            y=self.current_pos.y + movement[1],
            z=self.current_pos.z + movement[2]
        )
        self.current_pos = self._calculate_wrapped_position(new_pos)
        return self.current_pos

    def _calculate_wrapped_position(self, pos: Position) -> Position:
        """Handle toroidal wrapping at volume boundaries."""
        half_size = self.volume_size / 2

        def wrap_coordinate(coord: float) -> float:
            return ((coord + half_size) % self.volume_size) - half_size

        return Position(
            x=wrap_coordinate(pos.x),
            y=wrap_coordinate(pos.y),
            z=wrap_coordinate(pos.z)
        )

    def get_current_position(self) -> Position:
        """Get current position."""
        return self.current_pos

    def reset_position(self):
        """Reset the navigation system to origin."""
        self.last_end_position = Position(0.0, 0.0, 0.0)
        self.current_pos = self.last_end_position
        self.heading = np.array([1.0, 0.0, 0.0])