# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:09:43 2024

@author: Matt Andrews

@file: M_E_GA_Base.py

The central coordinating class for the Genetic Algorithm engine.

"""
import datetime
import json
import os
import random
# Removed: import logging
from collections import OrderedDict # Added for loading checkpoint

# Import LoggingManager first
from ..networkCommon.logging_manager import LoggingManager, VERBOSE_LEVEL_NUM # Corrected import path

from .GA_Logger import GA_Logger
from .M_E_Engine import EncodingManager
from .engine.crossover_manager import CrossoverManager
# Removed: from .engine.logging_manager import LoggingManager as EngineLoggingManager
from .engine.mutation_manager import MutationManager
from .engine.population_manager import PopulationManager


class M_E_GA_Base:
    """
    Base class for the genetic algorithm engine. It orchestrates:
     - GA configuration & initialization
     - Population generation and evolutionary loop
     - Delegation to manager classes for specific responsibilities
     - Summarizing final logs

    This class now defers most logging details to the LoggingManager.
    """

    def __init__(
            self,
            genes,
            fitness_function,
            mutation_prob=0.01,
            delimited_mutation_prob=0.01,
            delimit_delete_prob=0.01,
            open_mutation_prob=0.0001,
            metagene_mutation_prob=0.00001,
            delimiter_insert_prob=0.00001,
            crossover_prob=0.50,
            elitism_ratio=0.06,
            base_gene_prob=0.98,
            max_individual_length=6,
            population_size=400,
            num_parents=80,
            max_generations=1000,
            delimiters=True,
            delimiter_space=3,
            logging=True, # Keep this flag to enable/disable GA_Logger
            generation_logging=True, # Keep for GA_Logger control
            mutation_logging=False, # Keep for GA_Logger control
            crossover_logging=False, # Keep for GA_Logger control
            individual_logging=False, # Keep for GA_Logger control
            experiment_name=None,
            encodings=None,
            seed=None,
            before_fitness_evaluation=None,
            after_population_selection=None,
            before_generation_finalize=None,
            metagene_prob=0.0,
            fitness_evaluator=None,
            lru_cache_size=100,
            parallel_processing=False,
            use_threads=False,
            checkpoint_filename=None,
            resume=False,
            initial_population=None,
            initial_metagene_population=None,
            **kwargs
    ):
        """
        Initialize the genetic algorithm with configuration parameters.

        :param genes: A list of available base gene strings.
        :param fitness_function: A callable for evaluating individual fitness.
        :param mutation_prob: Base probability for point mutations.
        :param delimited_mutation_prob: Mutation probability for genes inside delimiters.
        :param delimit_delete_prob: Probability of deleting delimiter pairs.
        :param open_mutation_prob: Probability of opening meta-genes.
        :param metagene_mutation_prob: Probability for capturing a segment as a metagene.
        :param delimiter_insert_prob: Probability of inserting a delimiter pair.
        :param crossover_prob: Probability of performing a crossover event.
        :param elitism_ratio: Fraction of population that is carried over as elites.
        :param base_gene_prob: Probability of choosing a base gene vs. a meta gene.
        :param max_individual_length: Maximum length for an individual's encoding.
        :param population_size: The number of individuals in the population.
        :param num_parents: The number of parents used in reproduction.
        :param max_generations: How many generations to run the GA.
        :param delimiters: Whether to include Start/End delimiters in random organisms.
        :param delimiter_space: The spacing for random insertion of delimiters.
        :param logging: Enable or disable all logging via GA_Logger.
        :param generation_logging: If True, logs generation summaries via GA_Logger.
        :param mutation_logging: If True, logs each mutation event in detail via GA_Logger.
        :param crossover_logging: If True, logs each crossover event in detail via GA_Logger.
        :param individual_logging: If True, logs each individual's fitness each generation via GA_Logger.
        :param experiment_name: Name of the experiment (used for logging filenames).
        :param encodings: Optionally supply a pre-built dictionary of encodings.
        :param seed: A random seed for reproducibility.
        :param before_fitness_evaluation: A callable invoked before the population is evaluated.
        :param after_population_selection: A callable invoked after population selection.
        :param before_generation_finalize: A callable invoked before the generation finalizes.
        :param metagene_prob: Additional weighting factor used for meta-gene selection.
        :param fitness_evaluator: An object that handles population-level fitness evaluation.
        :param lru_cache_size: The size of the LRU cache for metagene usage.
        :param parallel_processing: Flag to enable parallel processing for fitness evaluation. Defaults to False.
        :param checkpoint_filename: Filename to store GA state between generations.
        :param resume: If True, attempt to load state from the checkpoint file.
        :param initial_population: Optionally supply an initial population (list of encoded organisms).
        :param initial_metagene_population: Optionally supply initial metagene state as a dict with keys:
            "meta_genes", "meta_gene_stack", "metagene_usage", "deletion_basket", "unused_encodings", and optionally "gene_counter_ref".
        :param kwargs: Additional arguments that might be used in extended setups.
        """
        # Get operational logger for M_E_GA_Base itself
        self.op_logger = LoggingManager.get_logger("MEGA.Base")

        self.genes = genes
        self.fitness_function = fitness_function
        self.fitness_evaluator = fitness_evaluator

        # Logging flags (control GA_Logger behavior)
        self.logging = logging
        self.generation_logging = generation_logging
        self.mutation_logging = mutation_logging
        self.crossover_logging = crossover_logging
        self.individual_logging = individual_logging
        self.experiment_name = experiment_name

        self.before_fitness_evaluation = before_fitness_evaluation
        self.after_population_selection = after_population_selection
        self.before_generation_finalize = before_generation_finalize

        # GA config parameters
        self.mutation_prob = mutation_prob
        self.delimited_mutation_prob = delimited_mutation_prob
        self.delimit_delete_prob = delimit_delete_prob
        self.open_mutation_prob = open_mutation_prob
        self.metagene_mutation_prob = metagene_mutation_prob
        self.delimiter_insert_prob = delimiter_insert_prob
        self.crossover_prob = crossover_prob
        self.elitism_ratio = elitism_ratio
        self.base_gene_prob = base_gene_prob
        self.metagene_prob = metagene_prob
        self.max_individual_length = max_individual_length
        self.population_size = population_size
        self.num_parents = num_parents
        self.max_generations = max_generations
        self.delimiters = delimiters
        self.delimiter_space = delimiter_space
        self.seed = seed

        # New state variables for resuming and checkpointing
        self.population = []
        self.current_generation = 0
        self.fitness_scores = []
        self.lru_cache_size = lru_cache_size
        self.parallel_processing = parallel_processing
        self.use_threads = use_threads
        self.checkpoint_filename = checkpoint_filename if checkpoint_filename is not None else "ga_checkpoint.json"

        # Seed the RNG if provided
        if seed is not None:
            random.seed(seed)

        # Setup GA_Logger (for event logging) if logging is on
        if self.logging:
            if self.experiment_name is None:
                self.experiment_name = "UnnamedExperiment"
            self.ga_event_logger = GA_Logger(self.experiment_name) # Renamed attribute
        else:
            self.ga_event_logger = None

        # Create an EncodingManager, integrate encodings if provided
        # Pass a dedicated logger instance
        self.encoding_manager = EncodingManager(
            lru_cache_size=self.lru_cache_size,
            logger=LoggingManager.get_logger("MEGA.EncodingManager") # Pass specific logger
        )
        if encodings:
            self.encoding_manager.integrate_uploaded_encodings(encodings, self.genes)
        else:
            for gene in self.genes:
                self.encoding_manager.add_gene(gene, verbose=True)

        # Instantiate manager classes
        self.population_manager = PopulationManager(self)
        # Pass self, they can access self.ga_event_logger if needed, or get their own logger
        self.mutation_manager = MutationManager(self)
        self.crossover_manager = CrossoverManager(self)

        # Remove instantiation of local LoggingManager
        # self.logging_manager = EngineLoggingManager(...)

        # Initialize initial population and/or metagene population if provided (and not resuming)
        if not resume:
            if initial_population is not None:
                self.population = initial_population

            if initial_metagene_population is not None and isinstance(initial_metagene_population, dict):
                self.encoding_manager.meta_genes = initial_metagene_population.get("meta_genes", [])
                self.encoding_manager.meta_gene_stack = initial_metagene_population.get("meta_gene_stack", [])
                self.encoding_manager.meta_manager.metagene_usage = OrderedDict(initial_metagene_population.get("metagene_usage", {}))
                self.encoding_manager.meta_manager.deletion_basket = initial_metagene_population.get("deletion_basket", {})
                self.encoding_manager.unused_encodings = initial_metagene_population.get("unused_encodings", [])
                self.encoding_manager.gene_counter_ref = initial_metagene_population.get("gene_counter_ref", self.encoding_manager.gene_counter_ref)

        # If resuming, try to load the checkpoint state which will override any initial population definitions.
        if resume:
            self.load_checkpoint()

    def decode_organism(self, encoded_organism, format=False):
        """
        Decode an encoded organism into its gene representation.

        :param encoded_organism: The list/tuple of codons (hash keys).
        :param format: If True, remove 'Start'/'End' from the result.
        :return: A list of decoded genes, possibly excluding delimiters if format=True.
        """
        encoded_organism = tuple(encoded_organism)
        decoded_genes = self.encoding_manager.decode(encoded_organism, verbose=False)
        if format:
            decoded_genes = [g for g in decoded_genes if g not in ['Start', 'End']]
        return decoded_genes

    def encode_string(self, genetic_string):
        """
        Encode a sequence of gene strings into their numeric codon representation.

        :param genetic_string: A list of gene symbols to encode.
        :return: A list of integer codons.
        """
        encoded_sequence = []
        for gene in genetic_string:
            if gene in self.encoding_manager.reverse_encodings:
                encoded_sequence.append(self.encoding_manager.reverse_encodings[gene])
            else:
                # Add gene if missing
                self.encoding_manager.add_gene(gene)
                encoded_sequence.append(self.encoding_manager.reverse_encodings[gene])
        return encoded_sequence

    def initialize_population(self):
        """
        Public method to initialize the population using the population manager.
        Useful for advanced usage if you want to manually do an 'init' step.

        :return: A newly generated population (list of organism encodings).
        """
        self.population = self.population_manager.initialize_population()
        return self.population

    def append_generation_log(self, generation, fitness_scores, population):
        """
        Append a summary of the current generation's fitness data to a single compiled log file.
        The log entry includes generation number, timestamp, average/median/best/worst fitness,
        and a small sample of the population.
        """
        # This method seems redundant if GA_Logger is used correctly.
        # GA_Logger should handle appending to the file.
        # Let's log the event via GA_Logger instead.
        if self.ga_event_logger and self.generation_logging:
            generation_summary = {
                "generation": generation,
                "average_fitness": sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
                "median_fitness": sorted(fitness_scores)[len(fitness_scores)//2] if fitness_scores else 0,
                "best_fitness": max(fitness_scores) if fitness_scores else 0,
                "worst_fitness": min(fitness_scores) if fitness_scores else 0,
                "population_sample": population[:5] # Keep sample small
            }
            self.ga_event_logger.log_event("generation_summary", generation_summary)


    def run_algorithm(self):
        # 1. Initialize population if empty.
        if not self.population:
            self.population = self.population_manager.initialize_population()

        try:
            for generation in range(self.current_generation, self.max_generations):
                self.current_generation = generation
                # Start generation-level logging (if GA_Logger is used)
                if self.ga_event_logger:
                     self.ga_event_logger.log_event("generation_start", {"generation": generation})
                self.encoding_manager.start_new_generation()

                # Evaluate fitness.
                self.fitness_scores = self.population_manager.evaluate_population_fitness(self.population)

                # Compute summary statistics.
                if self.fitness_scores:
                    avg_fit = sum(self.fitness_scores) / len(self.fitness_scores)
                    med_fit = sorted(self.fitness_scores)[len(self.fitness_scores)//2]
                    best_fit = max(self.fitness_scores)
                    worst_fit = min(self.fitness_scores)
                else:
                    avg_fit = med_fit = best_fit = worst_fit = 0

                # Log generation summary event via GA_Logger.
                self.append_generation_log(generation, self.fitness_scores, self.population) # Uses GA_Logger now

                # Generate new population.
                self.population = self.population_manager.select_and_generate_new_population(
                    self.population, self.fitness_scores, generation
                )

                # Optional user callback.
                if self.before_generation_finalize:
                    self.before_generation_finalize(self)

                # Optionally log individual fitness details via GA_Logger.
                if self.ga_event_logger and self.individual_logging:
                     for i, (ind, score) in enumerate(zip(self.population, self.fitness_scores)):
                          self.ga_event_logger.log_event("individual_fitness", {
                               "generation": generation,
                               "individual_index": i,
                               "fitness": score,
                               "organism": ind # Log encoded organism
                          })

                # Save compiled logger events to a unified log file.
                if self.ga_event_logger:
                    self.ga_event_logger.save()
                # Save checkpoint for recovery.
                self.save_checkpoint()

        except KeyboardInterrupt:
            self.op_logger.info("Experiment interrupted. Saving checkpoint and logs.") # Use op_logger
            self.save_checkpoint()
            if self.ga_event_logger:
                self.ga_event_logger.save()
            raise

        # End-of-experiment final logging.
        # print(self.encoding_manager.encodings) # Keep this print? Or log it? Log as debug.
        self.op_logger.debug(f"Final Encodings: {self.encoding_manager.encodings}")
        if self.logging: # Check the original flag
            final_log = {
                # "initial_configuration": { ... },  # Reconstruct if needed
                "final_population": self.population,
                "final_fitness_scores": self.fitness_scores,
                "genes": self.genes,
                "final_encodings": self.encoding_manager.encodings,
                # "compiled_logs": self.logging_manager.get_logs() # Removed local logging_manager
            }
            log_folder = "logs" # Use unified log folder
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            final_log_filename = os.path.join(log_folder, f"{self.experiment_name}_final_state_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
            try:
                with open(final_log_filename, 'w') as f:
                    json.dump(final_log, f, indent=4)
                self.op_logger.info(f"Final experiment state saved to {final_log_filename}")
            except Exception as e:
                 self.op_logger.error(f"Failed to save final experiment state: {e}", exc_info=True)

            if self.ga_event_logger:
                self.ga_event_logger.save() # Final save of any remaining events

    def save_checkpoint(self):
        """
        Save a minimal checkpoint of the GA state to a file.
        Only data necessary for resuming (e.g. current generation, population, and encoding state)
        is stored. Recalculated data (e.g. logs) are not persisted.
        """
        checkpoint_data = {
            "current_generation": self.current_generation,
            "population": self.population,
            "encoding_manager_state": {
                "encodings": self.encoding_manager.encodings,
                "reverse_encodings": self.encoding_manager.reverse_encodings,
                "meta_genes": self.encoding_manager.meta_genes,
                "meta_gene_stack": self.encoding_manager.meta_gene_stack,
                # Convert the OrderedDict to a list of tuples for JSON serialization
                "metagene_usage": list(self.encoding_manager.meta_manager.metagene_usage.items()),
                "deletion_basket": self.encoding_manager.meta_manager.deletion_basket,
                "unused_encodings": self.encoding_manager.unused_encodings,
                "gene_counter_ref": self.encoding_manager.gene_counter_ref
            }
        }
        try:
            with open(self.checkpoint_filename, "w") as f:
                json.dump(checkpoint_data, f, indent=4)
            self.op_logger.debug(f"Checkpoint saved to {self.checkpoint_filename}")
        except Exception as e:
             self.op_logger.error(f"Failed to save checkpoint to {self.checkpoint_filename}: {e}", exc_info=True)

    def load_checkpoint(self, filename=None):
        """
        Load the GA state from a checkpoint file.
        Updates the current generation, population, and encoding manager state.
        """
        filename = filename if filename else self.checkpoint_filename
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    checkpoint_data = json.load(f)
                self.current_generation = checkpoint_data.get("current_generation", 0)
                self.population = checkpoint_data.get("population", [])
                encoding_state = checkpoint_data.get("encoding_manager_state", {})
                self.encoding_manager.encodings = encoding_state.get("encodings", {})
                self.encoding_manager.reverse_encodings = encoding_state.get("reverse_encodings", {})
                self.encoding_manager.meta_genes = encoding_state.get("meta_genes", [])
                self.encoding_manager.meta_gene_stack = encoding_state.get("meta_gene_stack", [])
                metagene_usage_list = encoding_state.get("metagene_usage", [])
                self.encoding_manager.meta_manager.metagene_usage = OrderedDict(metagene_usage_list)
                self.encoding_manager.meta_manager.deletion_basket = encoding_state.get("deletion_basket", {})
                self.encoding_manager.unused_encodings = encoding_state.get("unused_encodings", [])
                self.encoding_manager.gene_counter_ref = encoding_state.get("gene_counter_ref", [3])
                self.op_logger.info(f"Checkpoint loaded from {filename}. Resuming from generation {self.current_generation}.") # Use op_logger
            except Exception as e:
                 self.op_logger.error(f"Failed to load or parse checkpoint file {filename}: {e}", exc_info=True)
                 # Reset state if loading fails? Or raise error? Resetting for now.
                 self.current_generation = 0
                 self.population = []
                 # Reset encoding manager state? Might be complex.
        else:
            self.op_logger.warning(f"No checkpoint file found at {filename}. Starting from scratch.") # Use op_logger