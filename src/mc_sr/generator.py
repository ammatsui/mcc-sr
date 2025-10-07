"""Class for generating sample datasets from the given big dataset."""


import numpy as np
import random
import copy

class Generator:
    def __init__(self, anchor_data, input_range=(-2, 2), px='uniform', n_g=512, sigma_y=0.02, unit_scale=1.0, transform_flags=None, mode="benchmark", f_star=None):
        """
        Parameters
        ----------
        input_range   : tuple, range for x
        px            : str, distribution name ('uniform' or 'normal')
        n_g           : int, number of samples
        sigma_y       : float, output noise standard deviation
        unit_scale    : float, scaling factor for units
        transform_flags: dict or None, augmentation flags/settings
        anchor_data   : tuple (X_anchor, y_anchor) for real-data mode
        mode          : 'benchmark' or 'real'
        f_star        : function(x), true law (for benchmarks mode)
        """
        self.input_range = input_range
        self.px = px
        self.n_g = n_g
        self.sigma_y = sigma_y
        self.unit_scale = unit_scale
        self.transform_flags = transform_flags or {}
        self.anchor_data = anchor_data
        self.mode = mode
        self.f_star = f_star


    @staticmethod
    def random_init(mode='benchmark',
                    input_range_choices=[(-2, 2), (-5, 5), (0, 1), (-1, 3)],
                    px_choices=['uniform', 'normal'],
                    n_g_range=(256, 1024),
                    sigma_y_range=(0.005, 0.05),
                    unit_scale_range=(0.5, 2.0),
                    transform_flag_pool=('aug', 'flip', 'scale'),
                    anchor_data=None,
                    f_star=None):
        """
        Returns a randomly initialized Generator instance.
        
        Parameters
        ----------
            mode: 'benchmark' or 'real'
            input_range_choices: list of tuples for possible input ranges
            px_choices: list of possible distribution names
            n_g_range: tuple (min, max) for sample size
            sigma_y_range: tuple (min, max) for output noise
            unit_scale_range: tuple (min, max) for scaling
            transform_flag_pool: tuple/list of possible transform flags
            anchor_data: if mode=='real'
            f_star: if mode=='benchmark'
        """
        
        input_range = random.choice(input_range_choices)
        px = random.choice(px_choices)
        n_g = random.randint(n_g_range[0], n_g_range[1])
        sigma_y = np.random.uniform(*sigma_y_range)
        unit_scale = np.random.uniform(*unit_scale_range)
        
        # Random transform flags: each with random True/False
        transform_flags = {k: bool(random.getrandbits(1)) for k in transform_flag_pool}
        
        # If benchmark, need a function f_star
        if mode == 'benchmark' and f_star is None:
            # Example: f(x) = sin(x) + x^2
            f_star = lambda x: np.sin(x) + x**2
        
        # If real, need anchor_data (or leave as None)
        # anchor_data should be provided externally if wanted

        gen = Generator(
            input_range=input_range,
            px=px,
            n_g=n_g,
            sigma_y=sigma_y,
            unit_scale=unit_scale,
            transform_flags=transform_flags,
            anchor_data=anchor_data,
            mode=mode,
            f_star=f_star
        )
        return gen

    def sample(self):
        """
        Materialize a dataset variant D_g = {(x_i, y_i)} of length n_g.

        Returns
        -------
        x : numpy array
        y : numpy array
        """
        if self.mode == "benchmark" and self.f_star is not None:
            # 1. Sample x_i according to px (within input_range)
            low, high = self.input_range
            if self.px == 'uniform':
                x = np.random.uniform(low, high, self.n_g)
            elif self.px == 'normal':
                x = np.random.normal((low+high)/2, (high-low)/4, self.n_g)
            else:
                raise ValueError("Unknown px type")
            # 2. Apply unit rescaling if needed
            x = self.unit_scale * x
            # 3. Generate y_i using f_star (true law) + noise
            y = self.f_star(x) + np.random.normal(0, self.sigma_y, self.n_g)
            # 4. Any additional transforms
            # (can implement more via transform_flags)
           
            return x.reshape(-1, 1), y

        elif self.mode == "real" and self.anchor_data is not None:
            X_anchor, y_anchor = self.anchor_data
            # 1. Bootstrap/subsample
            idx = np.random.choice(len(X_anchor), self.n_g, replace=True)
            x = X_anchor[idx]
            y = y_anchor[idx]
            # 2. Unit/scale transform
            x = self.unit_scale * x
            y = self.unit_scale * y
            # 3. Mild label noise
            y = y + np.random.normal(0, self.sigma_y, self.n_g)
            # 4. Any transform_flags logic here
            return x, y

        else:
            raise ValueError("Improper configuration for generator")
        
    def mutate(self):
        """
        Randomly mutate one parameter (range, noise, unit_scale, etc).
        Mutation means altering data-creation parameters:
        - Input range: Expand/narrow or shift sampling interval for x.
        - Noise level: Increase or decrease σ (sigma) for added label noise.
        - Distribution type: Switch between, e.g., uniform and normal.
        - Sample size: Adjust n_g up/down.
        - Unit/scaling: Modify scaling coefficient for x or y.
        - Transform flags: Toggle augmentation options, e.g., enable/disable a specific data transform.
        """
        import random
        mutation_types = ['input_range', 'noise', 'distribution', 'sample_size', 'unit_scale', 'transform_flag']
        mutation = random.choice(mutation_types)

        mutant = copy.deepcopy(self)

        if mutation == 'input_range':
            # Expand, narrow, or shift interval
            low, high = self.input_range
            action = random.choice(['expand', 'narrow', 'shift'])
            delta = (high - low) * 0.1
            if action == 'expand':
                mutant.input_range = (low - delta, high + delta)
            elif action == 'narrow' and (high - low) > 2 * delta:
                mutant.input_range = (low + delta, high - delta)
            elif action == 'shift':
                shift = random.uniform(-delta, delta)
                mutant.input_range = (low + shift, high + shift)

        elif mutation == 'noise':
            # Increase or decrease sigma_y
            factor = random.choice([0.8, 1.2])
            mutant.sigma_y = max(1e-6, self.sigma_y * factor)

        elif mutation == 'distribution':
            # Switch between uniform and normal
            mutant.px = 'normal' if self.px == 'uniform' else 'uniform'

        elif mutation == 'sample_size':
            # Adjust n_g up/down
            change = random.choice([-32, 32])
            mutant.n_g = max(1, self.n_g + change)

        elif mutation == 'unit_scale':
            # Modify scaling coefficient
            factor = random.choice([0.9, 1.1])
            mutant.unit_scale *= factor

        elif mutation == 'transform_flag':
            # Toggle a random transform flag
            if self.transform_flags:
                key = random.choice(list(self.transform_flags.keys()))
                mutant.transform_flags[key] = not self.transform_flags[key]
            else:
                # Add a random flag if none exist
                mutant.transform_flags['aug'] = True

        return mutant

    def to_vector(self):
        """Return generator parameters for clustering/species assignment."""
        return np.array([self.input_range[0], self.input_range[1], self.n_g, self.sigma_y, self.unit_scale])

    def __str__(self):
        return (f"Generator(mode={self.mode}, range={self.input_range}, "
                f"n_g={self.n_g}, sigma_y={self.sigma_y}, unit={self.unit_scale})")