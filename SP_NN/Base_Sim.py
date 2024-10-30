from SP_NN import Position, NetworkParameters, create_network, NeuronType
import numpy as np


class PongFitness:
    """Fitness function for evolving network structure with Pong simulation"""

    def __init__(self, simulation, network_params: dict, update_best_func=None):
        # Store simulation
        self.simulation = simulation
        self.sim_requirements = simulation.get_io_requirements()

        # Available genes for neural net structure evolution
        self.genes = ['U', 'D', 'F', 'B', 'DR']

        # Create network parameters
        self.network_params = NetworkParameters(
            volume_size=network_params['volume_size'],
            num_input=self.sim_requirements['inputs'],
            num_output=self.sim_requirements['outputs'],
            total_neurons=network_params['total_neurons'],
            activation_budget=network_params['activation_budget'],
            time_window_size=network_params['time_window_size']
        )

        # Create network
        self.network = create_network(self.network_params)

        # Path execution state
        self.pickup_bag = []
        self.max_carry = 3
        self.current_pos = Position(0, 0, 0)

        # Best solution tracking
        self.update_best = update_best_func

    def evaluate_network(self) -> float:
        """Run network evaluation"""
        self.simulation.reset()
        total_score = 0
        steps_completed = 0

        # Run for the full time window
        for step in range(self.network_params.time_window_size):
            # Get current game state as input
            current_state = [
                self.simulation.ball_pos[0] / self.simulation.width,
                self.simulation.ball_pos[1] / self.simulation.height,
                self.simulation.paddle_x / self.simulation.width
            ]

            # Run one cycle of the network
            outputs = self.network.run_cycle(current_state)

            # Update simulation with network's decision
            self.simulation.move_paddle(outputs)
            self.simulation.update_ball()

            # Check if ball is still in play
            if self.simulation.ball_pos[1] < self.simulation.paddle_y:
                break

            steps_completed += 1

        # Get final evaluation score
        fitness_score = self.simulation.evaluate()

        # Adjust score based on how long the ball stayed in play
        duration_factor = steps_completed / self.network_params.time_window_size
        final_score = fitness_score * duration_factor

        return final_score

    def execute_path(self, path: list[str]) -> dict:
        """Execute movement path"""
        state = self.network.get_network_state()
        neuron_positions = state['neuron_positions']
        position_updates = {}
        neurons_moved = 0

        for command in path:
            if command == 'DR' and self.pickup_bag:
                current_pos_tuple = (self.current_pos.x, self.current_pos.y, self.current_pos.z)
                if not any(pos['position'] == current_pos_tuple
                           for nid, pos in neuron_positions.items()
                           if nid not in self.pickup_bag):
                    neuron_id = self.pickup_bag.pop(0)
                    position_updates[neuron_id] = current_pos_tuple
                    neurons_moved += 1
            else:
                new_pos = self.move(command)
                if new_pos:
                    self.current_pos = new_pos
                    current_pos_tuple = (self.current_pos.x, self.current_pos.y, self.current_pos.z)
                    for nid, data in neuron_positions.items():
                        if (nid not in self.pickup_bag and
                                nid not in position_updates and
                                data['position'] == current_pos_tuple):
                            self.pickup_bag.append(nid)
                            break

        if position_updates:
            self.network.update_neuron_positions(position_updates)

        return {'neurons_moved': neurons_moved}

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
            return Position(
                x=(self.current_pos.x + dx) % self.network_params.volume_size,
                y=(self.current_pos.y + dy) % self.network_params.volume_size,
                z=(self.current_pos.z + dz) % self.network_params.volume_size
            )
        return None

    def compute(self, encoded_individual, ga_instance) -> float:
        """Main fitness computation"""
        path = ga_instance.decode_organism(encoded_individual)

        # Execute path and reorganize neurons
        movement_stats = self.execute_path(path)

        # Evaluate network performance
        network_score = self.evaluate_network()

        # Calculate final fitness
        movement_score = movement_stats['neurons_moved'] / len(self.network.neurons)
        fitness = movement_score * 0.4 + network_score * 0.6

        if self.update_best:
            self.update_best(encoded_individual, fitness)

        return fitness