import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_network_path(network, path, title="Network Path Visualization"):
    """
    Visualize the path taken through the network space, showing neuron positions
    and the movement path.
    """
    # Get network state
    state = network.get_network_state()
    neuron_positions = state['neuron_positions']

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot neurons
    for nid, data in neuron_positions.items():
        pos = data['position']
        ntype = data['type'].value  # Using the NeuronType enum value

        # Color based on neuron type
        color = {
            'input': 'red',
            'hidden': 'blue',
            'output': 'green'
        }.get(ntype, 'gray')

        ax.scatter(pos[0], pos[1], pos[2], c=color, alpha=0.6)

    # Simulate path and plot
    current_pos = np.array([0, 0, 0])
    path_x, path_y, path_z = [current_pos[0]], [current_pos[1]], [current_pos[2]]

    moves = {
        'U': np.array([0, 1, 0]),
        'D': np.array([0, -1, 0]),
        'F': np.array([1, 0, 0]),
        'B': np.array([-1, 0, 0])
    }

    for move in path:
        if move in moves:
            current_pos = current_pos + moves[move]
            path_x.append(current_pos[0])
            path_y.append(current_pos[1])
            path_z.append(current_pos[2])

    # Plot path
    ax.plot(path_x, path_y, path_z, 'r-', alpha=0.5, linewidth=2)

    # Mark start and end
    ax.scatter(path_x[0], path_y[0], path_z[0], c='green', s=100, label='Start')
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], c='red', s=100, label='End')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Add legend
    ax.legend(['Path', 'Input Neurons', 'Hidden Neurons', 'Output Neurons', 'Start', 'End'])

    plt.show()


def plot_output_activations(network, input_sequence=None, cycles=100):
    """
    Run the network and plot output neuron activations over time.
    """
    # If no input sequence provided, generate random inputs
    if input_sequence is None:
        input_size = len(network.input_neurons)
        input_sequence = [np.random.uniform(0, 1, input_size).tolist() for _ in range(cycles)]

    # Run network
    window_results = network.run_window(cycles=cycles, input_sequence=input_sequence)
    outputs = window_results['outputs']

    # Convert outputs to numpy array for easier plotting
    outputs_array = np.array(outputs)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot each output neuron's activation
    for i in range(outputs_array.shape[1]):
        plt.plot(outputs_array[:, i], label=f'Output {i + 1}')

    plt.xlabel('Time Step')
    plt.ylabel('Activation')
    plt.title('Output Neuron Activations Over Time')
    plt.legend()
    plt.grid(True)

    plt.show()


def visualize_best_solution(ga_instance, best_organism, fitness_function):
    """
    Visualize the best solution found by the GA.
    """
    # Decode the best path
    best_path = ga_instance.decode_organism(best_organism["genome"], format=True)

    print("\n=== Best Solution Visualization ===")
    print(f"Path length: {len(best_path)}")
    print(f"Final fitness: {best_organism['fitness']:.4f}")

    # Plot the path through the network
    plot_network_path(fitness_function.network, best_path,
                      title=f"Best Solution Path (Fitness: {best_organism['fitness']:.4f})")

    # Plot output activations for the best solution
    print("\nRunning network with best solution configuration...")
    plot_output_activations(fitness_function.network)