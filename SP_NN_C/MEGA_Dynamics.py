import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class MEGAParams:
    """Parameters controlling MEGA's mechanisms"""
    # Basic mutation rates
    point_mutation_rate: float = 0.01
    insertion_rate: float = 0.01
    deletion_rate: float = 0.01
    swap_rate: float = 0.01

    # Structural mutation rates
    capture_rate: float = 0.001
    open_rate: float = 0.0005
    delimiter_insert_rate: float = 0.001
    delimiter_delete_rate: float = 0.001

    # Gene selection probabilities
    base_gene_prob: float = 0.98
    capture_gene_prob: float = 0.02

    # Population parameters
    population_size: int = 400
    max_sequence_length: int = 50
    min_segment_length: int = 2
    max_segment_length: int = 10


class MEGASystem:
    def __init__(self, base_genes: List[str], params: MEGAParams = None):
        self.base_genes = base_genes
        self.params = params or MEGAParams()

        # Initialize state
        self.metagenes = {}  # Dictionary mapping metagene IDs to their content
        self.metagene_usage = {}  # Track usage frequency
        self.next_metagene_id = 0  # Counter for generating unique metagene IDs

        # Special genes
        self.start_gene = 'Start'
        self.end_gene = 'End'

    def select_gene(self) -> str:
        """Gene selection mechanism for mutations and insertions"""
        if random.random() < self.params.base_gene_prob or not self.metagenes:
            # Select base gene
            return random.choice(self.base_genes)
        else:
            # Select metagene with preference for recent ones
            metagene_ids = list(self.metagenes.keys())
            weights = [self.params.capture_gene_prob ** (len(metagene_ids) - i)
                       for i in range(len(metagene_ids))]
            total = sum(weights)
            weights = [w / total for w in weights]  # Normalize
            return f"META_{random.choices(metagene_ids, weights=weights)[0]}"

    def mutate_sequence(self, sequence: List[str]) -> List[str]:
        """Apply mutation operators based on rates and context"""
        i = 0
        while i < len(sequence):
            rates = self.params

            # Determine context
            in_delimited = self.in_delimited_segment(sequence, i)

            # Apply mutations based on probabilities
            if random.random() < rates.point_mutation_rate:
                sequence[i] = self.select_gene()

            elif random.random() < rates.insertion_rate:
                sequence.insert(i, self.select_gene())

            elif random.random() < rates.deletion_rate and len(sequence) > 1:
                sequence.pop(i)
                continue  # Skip increment as we removed current position

            elif random.random() < rates.swap_rate and i < len(sequence) - 1:
                if self.can_swap(sequence, i, i + 1):
                    sequence[i], sequence[i + 1] = sequence[i + 1], sequence[i]

            # Structural mutations
            if not in_delimited:
                if random.random() < rates.delimiter_insert_rate:
                    sequence = self.insert_delimiter_pair(sequence, i)

            if random.random() < rates.delimiter_delete_rate:
                if sequence[i] in [self.start_gene, self.end_gene]:
                    sequence.pop(i)
                    continue

            # Try capture if in delimited segment
            if in_delimited and random.random() < rates.capture_rate:
                new_seq = self.attempt_capture(sequence, i)
                if new_seq is not None:
                    sequence = new_seq
                    continue

            # Try open if current gene is a metagene
            if random.random() < rates.open_rate and self.is_metagene(sequence[i]):
                sequence = self.open_metagene(sequence, i, in_delimited)
                continue

            i += 1

        return sequence

    def is_metagene(self, gene: str) -> bool:
        """Check if a gene is a metagene"""
        if isinstance(gene, str) and gene.startswith("META_"):
            meta_id = int(gene.split("_")[1])
            return meta_id in self.metagenes
        return False

    def get_metagene_content(self, gene: str) -> Optional[List[str]]:
        """Get the content of a metagene"""
        if self.is_metagene(gene):
            meta_id = int(gene.split("_")[1])
            return self.metagenes.get(meta_id)
        return None

    def in_delimited_segment(self, sequence: List[str], pos: int) -> bool:
        """Check if position is within a delimited segment"""
        depth = 0
        for i in range(pos):
            if sequence[i] == self.start_gene:
                depth += 1
            elif sequence[i] == self.end_gene:
                depth -= 1
        return depth > 0

    def can_swap(self, sequence: List[str], i: int, j: int) -> bool:
        """Check if two positions can be swapped"""
        if not (0 <= i < len(sequence) and 0 <= j < len(sequence)):
            return False
        if sequence[i] in [self.start_gene, self.end_gene] and \
                sequence[j] in [self.start_gene, self.end_gene]:
            return False
        return True

    def insert_delimiter_pair(self, sequence: List[str], pos: int) -> List[str]:
        """Insert a pair of delimiters around position"""
        sequence.insert(pos, self.start_gene)
        end_pos = min(pos + 2, len(sequence))
        sequence.insert(end_pos, self.end_gene)
        return sequence

    def attempt_capture(self, sequence: List[str], pos: int) -> Optional[List[str]]:
        """Try to capture a delimited segment into a metagene"""
        segment = self.find_segment(sequence, pos)
        if segment is None:
            return None

        start_pos, end_pos, content = segment

        # Create new metagene
        content_tuple = tuple(content)
        meta_id = None

        # Check if this content already exists as a metagene
        for mid, mcontent in self.metagenes.items():
            if tuple(mcontent) == content_tuple:
                meta_id = mid
                break

        # If not found, create new metagene
        if meta_id is None:
            meta_id = self.next_metagene_id
            self.next_metagene_id += 1
            self.metagenes[meta_id] = content
            self.metagene_usage[meta_id] = 0

        # Create metagene reference and update usage
        metagene_ref = f"META_{meta_id}"
        self.metagene_usage[meta_id] += 1

        return sequence[:start_pos] + [metagene_ref] + sequence[end_pos + 1:]

    def open_metagene(self, sequence: List[str], pos: int, in_delimited: bool) -> List[str]:
        """Open a metagene, with context-dependent delimiter handling"""
        if not self.is_metagene(sequence[pos]):
            return sequence

        content = self.get_metagene_content(sequence[pos])
        if content is None:
            return sequence

        expanded = content.copy()

        # Add delimiters if in undelimited region
        if not in_delimited:
            expanded = [self.start_gene] + expanded + [self.end_gene]

        return sequence[:pos] + expanded + sequence[pos + 1:]

    def find_segment(self, sequence: List[str], pos: int) -> Optional[Tuple[int, int, List[str]]]:
        """Find delimited segment containing position"""
        start_pos = None
        for i in range(pos, -1, -1):
            if sequence[i] == self.start_gene:
                start_pos = i
                break

        if start_pos is None:
            return None

        for i in range(start_pos + 1, len(sequence)):
            if sequence[i] == self.end_gene:
                content = sequence[start_pos + 1:i]
                if len(content) >= self.params.min_segment_length:
                    return (start_pos, i, content)
                break

        return None

    def evolve_population(self, generations: int,
                          initial_sequences: Optional[List[List[str]]] = None):
        """Evolve a population over multiple generations"""
        population = initial_sequences if initial_sequences is not None else self.initialize_population()

        for gen in range(generations):
            new_population = []
            for sequence in population:
                mutated = self.mutate_sequence(sequence.copy())
                new_population.append(mutated)

            population = new_population

            # Print statistics
            avg_length = sum(len(s) for s in population) / len(population)
            num_metagenes = len(self.metagenes)
            print(f"Generation {gen}: Avg Length = {avg_length:.1f}, Metagenes = {num_metagenes}")

        return population

    def initialize_population(self) -> List[List[str]]:
        """Create initial random population"""
        population = []
        for _ in range(self.params.population_size):
            length = random.randint(2, self.params.max_sequence_length)
            sequence = [self.select_gene() for _ in range(length)]
            population.append(sequence)
        return population


# Example usage
if __name__ == "__main__":
    base_genes = ['A', 'B', 'C', 'D']
    params = MEGAParams(
        point_mutation_rate=0.02,
        capture_rate=0.002,
        base_gene_prob=0.95
    )

    system = MEGASystem(base_genes, params)
    final_pop = system.evolve_population(generations=50)