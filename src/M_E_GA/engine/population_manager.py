import random
import concurrent.futures
from functools import partial
# Removed: import logging
from concurrent.futures import ThreadPoolExecutor

# Import LoggingManager
from ...networkCommon.logging_manager import LoggingManager, VERBOSE_LEVEL_NUM # Corrected import path

def evaluate_individual_fitness(individual, fitness_function, ga_instance):
    return fitness_function(individual, ga_instance)

class PopulationManager:
    def __init__(self, ga_instance):
        self.ga = ga_instance

    def initialize_population(self):
        population = []
        for _ in range(int(self.ga.population_size)):
            individual_length = random.randint(2, self.ga.max_individual_length)
            organism = self.ga.encoding_manager.generate_random_organism(
                functional_length=individual_length,
                include_specials=self.ga.delimiters,
                probability=0.10,
                verbose=False
            )
            population.append(organism)
        return population

    def evaluate_population_fitness(self, population):
        if self.ga.before_fitness_evaluation:
            self.ga.before_fitness_evaluation(self.ga)

        # Log that we are starting fitness evaluation.
        # Removed: import logging
        LoggingManager.debug(f"Starting fitness evaluation for {len(population)} individuals. " # Updated
                    f"parallel_processing={self.ga.parallel_processing}", name="MEGA.PopulationManager")

        # Choose evaluation strategy based on configuration.
        if self.ga.parallel_processing:
            # Choose between thread or process executor.
            if hasattr(self.ga, 'use_threads') and self.ga.use_threads:
                executor_class = ThreadPoolExecutor
                kwargs = {}  # No chunksize for threads.
            else:
                executor_class = concurrent.futures.ProcessPoolExecutor
                kwargs = {'chunksize': 10}  # Increase chunksize to reduce overhead.

            with executor_class() as executor:
                if self.ga.fitness_evaluator is not None:
                    func = partial(evaluate_individual_fitness,
                                fitness_function=self.ga.fitness_evaluator.evaluate,
                                ga_instance=self.ga)
                else:
                    func = partial(evaluate_individual_fitness,
                                fitness_function=self.ga.fitness_function,
                                ga_instance=self.ga)
                fitness_scores = list(executor.map(func, population, **kwargs))
        else:
            if self.ga.fitness_evaluator is not None:
                fitness_scores = self.ga.fitness_evaluator.evaluate(population, self.ga)
            else:
                fitness_scores = [self.ga.fitness_function(ind, self.ga) for ind in population]

        if self.ga.after_population_selection:
            self.ga.after_population_selection(self.ga)

        LoggingManager.debug(f"Finished fitness evaluation. Calculated scores: {fitness_scores}", name="MEGA.PopulationManager") # Updated

        return fitness_scores

    def select_and_generate_new_population(self, population, fitness_scores, generation):
        # ... (existing selection, crossover, and mutation code) ...
        sorted_population = sorted(zip(population, fitness_scores),
                                   key=lambda x: x[1], reverse=True)
        num_elites = int(self.ga.elitism_ratio * self.ga.population_size)
        elites = [individual for (individual, _) in sorted_population[:num_elites]]
        new_population = elites[:]
        selected_parents = [individual for (individual, _) in sorted_population[:self.ga.num_parents]]
        shift = 0
        while len(new_population) < self.ga.population_size:
            for i in range(0, len(selected_parents) - 1, 2):
                parent1_index = (i + shift) % len(selected_parents)
                parent2_index = (i + 1 + shift) % len(selected_parents)
                parent1 = selected_parents[parent1_index]
                parent2 = selected_parents[parent2_index]
                if self.ga.crossover_manager.is_fully_delimited(parent1) or \
                   self.ga.crossover_manager.is_fully_delimited(parent2):
                    new_population.extend([parent1, parent2][:self.ga.population_size - len(new_population)])
                    continue
                if random.random() < self.ga.crossover_prob:
                    non_del_indices = self.ga.crossover_manager.get_non_delimiter_indices(parent1, parent2)
                    offspring1, offspring2 = self.ga.crossover_manager.crossover(
                        parent1, parent2, non_del_indices, generation
                    )
                else:
                    offspring1, offspring2 = parent1[:], parent2[:]
                self.ga.logging_manager.log_new_organism(offspring1)
                self.ga.logging_manager.log_new_organism(offspring2)
                offspring1 = self.ga.mutation_manager.mutate_organism(offspring1, generation)
                offspring2 = self.ga.mutation_manager.mutate_organism(offspring2, generation)
                new_population.extend([offspring1, offspring2][:self.ga.population_size - len(new_population)])
            shift += 1
        return new_population
