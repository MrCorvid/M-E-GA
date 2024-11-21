import numpy as np
from numba import cuda, float32
import math
from scipy.stats import gaussian_kde
import random

@cuda.jit(device=True)
def toroidal_distance(a, b, volume):
    dx = min(abs(a[0] - b[0]), volume - abs(a[0] - b[0]))
    dy = min(abs(a[1] - b[1]), volume - abs(a[1] - b[1]))
    dz = min(abs(a[2] - b[2]), volume - abs(a[2] - b[2]))
    return math.sqrt(dx*dx + dy*dy + dz*dz)

@cuda.jit
def calculate_repulsion(particles, forces, volume, repulsion_strength, min_distance):
    i = cuda.grid(1)
    if i < particles.shape[0]:
        pos = particles[i]
        force = cuda.local.array(3, float32)
        force[0], force[1], force[2] = 0.0, 0.0, 0.0

        for j in range(particles.shape[0]):
            if i != j:
                other = particles[j]
                dist = toroidal_distance(pos, other, volume)
                if dist < min_distance:
                    dist = min_distance
                magnitude = repulsion_strength / (dist * dist)

                for k in range(3):
                    diff = (pos[k] - other[k] + volume/2) % volume - volume/2
                    force[k] += magnitude * diff / dist

        for k in range(3):
            forces[i, k] = force[k]

class IDontKnow:
    def __init__(self, volume, num_particles, update_best_func, repulsion_strength=1.0, min_distance=1.0):
        self.update_best = update_best_func
        self.volume = volume
        self.num_particles = num_particles
        self.repulsion_strength = repulsion_strength
        self.min_distance = min_distance

        self.genes = ['R', 'L', 'U', 'D', 'F', 'B', 'DR']
        self.directions = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}

        self.particles = self.initialize_particles()

        self.step_reward = 2
        self.step_penalty = -4
        self.soft_step_limit = 500

    def initialize_particles(self):
        return np.random.randint(0, self.volume, size=(self.num_particles, 3)).astype(np.float32)

    def update_particle_positions(self):
        d_particles = cuda.to_device(self.particles)
        d_forces = cuda.device_array_like(self.particles)

        threads_per_block = 256
        blocks = (self.num_particles + threads_per_block - 1) // threads_per_block

        calculate_repulsion[blocks, threads_per_block](d_particles, d_forces, self.volume, self.repulsion_strength, self.min_distance)

        forces = d_forces.copy_to_host()
        self.particles += forces
        np.mod(self.particles, self.volume, out=self.particles)

        self.particles = np.round(self.particles).astype(np.float32)
        self.handle_collisions()

    def handle_collisions(self):
        unique_positions, indices, counts = np.unique(self.particles, axis=0, return_index=True, return_counts=True)
        if len(unique_positions) < len(self.particles):
            collision_mask = counts > 1
            collision_positions = unique_positions[collision_mask]
            for pos in collision_positions:
                collided_indices = np.where((self.particles == pos).all(axis=1))[0]
                for idx in collided_indices[1:]:
                    new_pos = self.nudge_particle(pos)
                    self.particles[idx] = new_pos

    def nudge_particle(self, position):
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        random.shuffle(directions)
        for dx, dy, dz in directions:
            new_pos = np.mod(position + [dx, dy, dz], self.volume)
            if not np.any(np.all(self.particles == new_pos, axis=1)):
                return new_pos
        return position  # If all adjacent positions are occupied, return the original position

    def calculate_entropy(self):
        try:
            kde = gaussian_kde(self.particles.T)
            entropy = -np.sum(kde.pdf(self.particles.T) * np.log(kde.pdf(self.particles.T)))
        except np.linalg.LinAlgError:
            # Fallback to a simpler entropy calculation
            hist, _ = np.histogramdd(self.particles, bins=10, range=[(0, self.volume)]*3)
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob[prob > 0] * np.log(prob[prob > 0]))
        return entropy

    def compute(self, encoded_individual, ga_instance, verbose=False):
        decoded_individual = ga_instance.decode_organism(encoded_individual)
        fitness_score = 0.0
        x, y, z = 0, 0, 0
        sack = []
        step_count = 0

        particle_positions = set(map(tuple, self.particles))

        for gene in decoded_individual:
            if gene == 'DR':
                if sack:
                    dropped_particle = sack.pop(0)
                    current_pos = (x % self.volume, y % self.volume, z % self.volume)
                    if current_pos not in particle_positions:
                        self.particles = np.vstack((self.particles, current_pos))
                        particle_positions.add(current_pos)
                        fitness_score += 0.001  # Small reward for dropping
            elif gene in self.directions:
                dx, dy, dz = self.directions[gene]
                x, y, z = (x + dx) % self.volume, (y + dy) % self.volume, (z + dz) % self.volume

                if step_count < self.soft_step_limit:
                    fitness_score += self.step_reward
                else:
                    fitness_score += self.step_penalty

                current_pos = (x, y, z)
                if current_pos in particle_positions:
                    particle_index = np.where((self.particles == current_pos).all(axis=1))[0][0]
                    sack.append(self.particles[particle_index])
                    self.particles = np.delete(self.particles, particle_index, axis=0)
                    particle_positions.remove(current_pos)

                step_count += 1

        entropy = self.calculate_entropy()
        fitness_score += 1 / (entropy + 1e-10)

        self.update_particle_positions()

        self.update_best(encoded_individual, fitness_score)

        return fitness_score

# Usage example:
def update_best_organism(genome, fitness):
    # Implementation of updating the best organism
    pass

# simulator = IDontKnow(volume=10, num_particles=50, update_best_func=update_best_organism)