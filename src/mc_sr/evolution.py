""" evolutionary step for calling mutation example:

def evolutionary_step(equation, llm, use_llm=False):
    # Use LLM if triggered, otherwise classical mutate
    if use_llm:
        return llm_mutate(equation, llm)
    else:
        return equation.mutate()  # classical random mutation
"""
from equation import Equation
from generator import Generator
from mcc import EquationMC, GeneratorMC
from logger import MetricLogger
import random, numpy

class EvolutionEngine:
    def __init__(self, equation_queue, generator_queue, anchor_x, anchor_y, 
                 tau, tau_prime, L_max,
                 n_generations=100, batch_size=10, logger=None):
        for eq in equation_queue:
            if eq is None:
                raise RuntimeError("Equation queue returned None!")
        self.equation_queue = equation_queue
        self.generator_queue = generator_queue
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.tau = tau
        self.tau_prime = tau_prime
        self.L_max = L_max
        self.n_generations = n_generations
        self.batch_size = batch_size
        self.logger = logger
        self.mc_pass_history = []              # Counts of MC passes each gen
        self.coverage_grid_history = []        # Cells filled per gen (optional)
        self.stagnation_window = 30            # How many generations to look back
        self.stagnation_min_passes = 0         # Min passes to avoid stagnation
        self.llm_enabled = False               # Toggle for LLM mutation
        # add any other state you want to track (species, metrics, etc.)

    # def log_generation_metrics(self, passed_eqs, passed_gens):
    #     mc_passes = len(passed_eqs) + len(passed_gens)
    #     self.mc_pass_history.append(mc_passes)

    def select_parents(self, queue):
        """Uniform random parent selection from a single viable queue."""
        # If queue is smaller than batch_size, sample with replacement for infinite reproduction
        if len(queue) == 0:
            raise ValueError("Cannot select parents: queue is empty.")
        return random.choices(queue, k=self.batch_size)
    
    def trim_queue(self, queue):
        """Trim queue to max size by removing oldest entries."""
        while len(queue) > self.L_max:
            queue.pop(0)  # remove oldest

    def update_thresholds(self):
        # ToDo: Update MC thresholds - do we need it?
        pass 

    def check_stagnation(self):
        """
        Returns True if no new individuals have passed MC in the last N generations.
        """
        history = self.mc_pass_history
        window = self.stagnation_window
        min_passes = self.stagnation_min_passes
        # Only check if we've run enough generations
        if len(history) < window:
            return False
        recent = history[-window:]
        return sum(recent) <= min_passes
    
    def run(self):
        """
        Main evolution loop. Handles population management, mutation, evaluation,
        logging, and adaptive MC gating.
        """
        eq_mc = EquationMC(self.tau, self.tau_prime, self.L_max)
        gen_mc = GeneratorMC(self.tau_prime)
        for generation in range(self.n_generations):
            # ===== 1. SELECT PARENTS =====
            eq_parents = self.select_parents(self.equation_queue)
            gen_parents = self.select_parents(self.generator_queue)

            # ===== 2. GENERATE CHILDREN =====
            eq_children = [e.mutate() for e in eq_parents]
            gen_children = [g.mutate() for g in gen_parents]

            # ===== 3. EVALUATE EQUATIONS =====
            passed_eqs = []
            for eq in eq_children:
                # Refit constants (if structure mutated)
                eq.fit_constants(self.anchor_x, self.anchor_y)
                # MC gate: must pass on anchor AND at least one generator
                if eq_mc.is_viable(eq, [self.anchor_x, self.anchor_y], self.generator_queue):
                    self.equation_queue.append(eq)
                    passed_eqs.append(eq)
                    self.trim_queue(self.equation_queue)
        
            # ===== 4. EVALUATE GENERATORS =====
            passed_gens = []
            for gen in gen_children:
                if gen_mc.is_viable(gen, self.equation_queue):
                    self.generator_queue.append(gen)
                    passed_gens.append(gen)
                    self.trim_queue(self.generator_queue)

            # # ===== 5. ADAPT THRESHOLDS =====
            # # Example: update tau to 25th percentile of anchor MSEs from last k successful equations
            # self.tau = self.update_thresholds(self.recent_eq_anchor_mse, 0.25)
            # self.tau_prime = self.update_thresholds(self.recent_eq_generator_mse, 0.25)

            # self.log_generation_metrics(passed_eqs, passed_gens)
            # ===== 7. LOG METRICS AND PASS-FAIL STATS =====
            self.logger.log_generation(
                generation=generation,
                tau=self.tau,
                tau_prime=self.tau_prime,
                eq_passed=len(passed_eqs),
                gen_passed=len(passed_gens),
                eq_queue_size=len(self.equation_queue),
                gen_queue_size=len(self.generator_queue)
            )

            # ===== 8. OPTIONAL: LLM TRIGGERS, COVERAGE GRID =====
            if self.llm_enabled and self.check_stagnation():
                self.llm_mutate_generation()
        
            # if self.use_map_elites:
            #     self.update_coverage_grid(passed_eqs, passed_gens)

            # ===== 9. END Generation =====

        # ===== 10. FINALIZE =====
        return self.equation_queue, self.generator_queue, self.logger.metrics