from dataclasses import dataclass
from collections import deque
import numpy as np
from typing import List, Dict, Tuple, Generator, Optional, Deque
from SP_NN import Position


class NavigationSystem:
    # Command patterns (4-bit)
    HEADING_X = 0b1111  # 1111 - X-Heading Change
    HEADING_Y = 0b1110  # 1110 - Y-Heading Change
    HEADING_Z = 0b1100  # 1100 - Z-Heading Change
    MOVE = 0b1000      # 1000 - Move
    DROP = 0b1001      # 1001 - Drop

    def __init__(self, volume_size: float = 10.0):
        """Initialize navigation system with volume constraints."""
        self.volume_size = volume_size
        self.current_pos = Position(0.0, 0.0, 0.0)
        self.heading = np.array([1.0, 0.0, 0.0])  # Initial heading along x-axis

        # Command processing state
        self.command_history = []
        self.current_command = None
        self.bit_accumulator = []

        # Constants for magnitude processing
        self.MIN_MAGNITUDE_BITS = 6
        self.MAX_MAGNITUDE_BITS = 16
        self.PRECISION = 1e-6

    def create_bit_stream(self, numbers: List[int]) -> str:
        """
        Convert list of integers into a single binary string.
        Args:
            numbers: List of integers [0-9] to be processed
        Returns:
            Binary string representing the concatenated number
        """
        # Validate input numbers
        if not all(0 <= n <= 9 for n in numbers):
            raise ValueError("All numbers must be in range [0-9]")

        # Concatenate numbers into single integer
        number = int(''.join(map(str, numbers)))

        # Convert to binary string, strip '0b' prefix
        return bin(number)[2:]

    def process_command_stream(self, binary_str: str) -> List[Tuple[int, float]]:
        """
        Process binary string into commands and magnitudes.
        Args:
            binary_str: Binary string to process
        Returns:
            List of (command, magnitude) tuples
        """
        commands = []
        bit_buffer = []
        magnitude_bits = []
        in_magnitude = False
        selector_bit = None

        # Process each bit in the stream
        for bit in binary_str:
            bit = int(bit)

            if not in_magnitude:
                # Building potential command
                bit_buffer.append(bit)

                if len(bit_buffer) == 4:
                    # Check if we have a valid command
                    command = int(''.join(map(str, bit_buffer)), 2)

                    if command in [self.HEADING_X, self.HEADING_Y,
                                 self.HEADING_Z, self.MOVE, self.DROP]:
                        # Valid command found
                        self.current_command = command
                        selector_bit = None  # Will be set with next bit
                        bit_buffer = []
                        in_magnitude = True
                    else:
                        # Not a valid command, shift window
                        bit_buffer = bit_buffer[1:]

            else:  # Processing magnitude
                if selector_bit is None:
                    # This bit is the selector bit
                    selector_bit = bit
                    magnitude_bits = []
                else:
                    magnitude_bits.append(bit)

                    # Check if we have enough magnitude bits
                    min_bits = 8 if selector_bit else 6
                    if len(magnitude_bits) >= min_bits:
                        # Check next 4 bits for potential command
                        if len(magnitude_bits) >= min_bits + 4:
                            potential_cmd = int(''.join(map(str, magnitude_bits[-4:])), 2)

                            if potential_cmd in [self.HEADING_X, self.HEADING_Y,
                                               self.HEADING_Z, self.MOVE, self.DROP]:
                                # Process current magnitude and start new command
                                magnitude = self.calculate_magnitude(
                                    magnitude_bits[:-4],
                                    selector_bit
                                )
                                commands.append((self.current_command, magnitude))

                                # Setup for next command
                                self.current_command = potential_cmd
                                in_magnitude = True
                                selector_bit = None
                                magnitude_bits = []
                                continue

        # Process final command if we have one
        if self.current_command is not None and magnitude_bits:
            magnitude = self.calculate_magnitude(magnitude_bits, selector_bit)
            commands.append((self.current_command, magnitude))

        return commands

    def calculate_magnitude(self, bits: List[int], selector: int) -> float:
        """
        Calculate magnitude value from bits using selector bit.
        Args:
            bits: List of magnitude bits
            selector: Selector bit (0: 6+ bits, 1: 8+ bits)
        Returns:
            Normalized float value for magnitude
        """
        if not bits:
            return 0.0

        # Convert bits to integer
        value = int(''.join(map(str, bits)), 2)

        # Calculate normalization factor based on bit length
        norm_factor = (1 << len(bits)) - 1
        normalized = value / norm_factor

        # Scale based on command type and selector
        if self.current_command in [self.HEADING_X, self.HEADING_Y, self.HEADING_Z]:
            # Heading changes: Scale to degrees
            max_value = 360.0 if selector else 180.0
        else:
            # Movement: Scale to volume size
            max_value = self.volume_size if selector else (self.volume_size / 2)

        return normalized * max_value

    def execute_movement_sequence(self, numbers: List[int]) -> Dict:
        """
        Execute complete movement sequence from input numbers.
        Args:
            numbers: List of integers to process
        Returns:
            Dictionary containing execution results
        """
        # Reset state
        self.command_history = []
        positions = [self.current_pos]
        command_positions = []  # Track positions where commands occur
        moves_made = drops_made = 0

        try:
            # Generate binary string
            binary_str = self.create_bit_stream(numbers)

            # Process commands
            commands = self.process_command_stream(binary_str)

            # Execute each command
            current_pos_index = 0
            for cmd_idx, (command, magnitude) in enumerate(commands):
                current_pos_index = len(positions) - 1  # Get current position index

                if command == self.HEADING_X:
                    self.update_heading('x', magnitude)
                    command_positions.append(current_pos_index)  # Record command position
                elif command == self.HEADING_Y:
                    self.update_heading('y', magnitude)
                    command_positions.append(current_pos_index)  # Record command position
                elif command == self.HEADING_Z:
                    self.update_heading('z', magnitude)
                    command_positions.append(current_pos_index)  # Record command position
                elif command == self.MOVE:
                    new_pos = self.move(magnitude)
                    positions.append(new_pos)
                    moves_made += 1
                    command_positions.append(len(positions) - 1)  # Record command position
                elif command == self.DROP:
                    positions.append(self.current_pos)
                    drops_made += 1
                    command_positions.append(len(positions) - 1)  # Record command position

                self.command_history.append((command, magnitude))

        except Exception as e:
            print(f"Error during movement sequence: {e}")

        return {
            'positions': positions,
            'moves_made': moves_made,
            'drops_made': drops_made,
            'commands_executed': len(self.command_history),
            'command_history': self.command_history,
            'command_positions': command_positions,  # Add command positions to results
            'final_position': self.current_pos
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
        """Move along current heading vector."""
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
            wrapped = ((coord + half_size) % self.volume_size) - half_size
            # Round to remove floating point errors
            return round(wrapped, 6)

        return Position(
            x=wrap_coordinate(pos.x),
            y=wrap_coordinate(pos.y),
            z=wrap_coordinate(pos.z)
        )

    def get_current_position(self) -> Position:
        """Get current position."""
        return self.current_pos