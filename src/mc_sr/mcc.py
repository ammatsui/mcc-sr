"""Classes for evaluation metrics (minimal criterion) on equations and generators."""

from abc import ABC, abstractmethod
from equation import Equation
from generator import Generator

class MinimalCriterion(ABC):
    @abstractmethod
    def is_viable(self, candidate, *args, **kwargs):
        """Return True if candidate passes MC, False otherwise."""
        pass

class EquationMC(MinimalCriterion):
    def __init__(self, tau, tau_prime, L_max):
        self.tau = tau # mse threshold on anchor data
        self.tau_prime = tau_prime # mse threshold on generated data
        self.L_max = L_max # max tree size

    def is_viable(self, equation, D0, generators):
        # 1. Fit constants on anchor, check threshold 
        """NOTE: maybe fit on each generated dataset when fitting instead?"""
        mse_anchor = equation.fit_constants(D0.x, D0.y)
        if mse_anchor > self.tau:
            return False
        # 2. Check if the equation fits at lease one generated dataset
        for gen in generators:
            xg, yg = gen.sample()
            mse_gen = equation.calculate_mse(xg, yg)
            if mse_gen <= self.tau_prime:
                # 3. Check tree size
                if equation.size() <= self.L_max:
                    return True
        return False

class GeneratorMC(MinimalCriterion):
    def __init__(self, tau_prime):
        self.tau_prime = tau_prime # mse threshold on generated data

    def is_viable(self, generator, equations):
        # Checks if the generated data fits at least one equation
        """NOTE: maybe add fitting constants here as well?"""
        xg, yg = generator.sample()
        for eq in equations:
            mse = eq.calculate_mse(xg, yg)
            if mse <= self.tau_prime:
                return True
        return False