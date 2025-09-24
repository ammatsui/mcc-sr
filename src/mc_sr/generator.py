"""Class for generating sample datasets from the given big dataset."""


import numpy as np

class Generator:
    def __init__(self, input_range=(-2, 2), px='uniform', n_g=512, sigma_y=0.02, unit_scale=1.0, transform_flags=None, anchor_data=None, mode="benchmark", f_star=None):
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
            return x, y

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
        """Randomly mutate one parameter (range, noise, unit_scale, etc).
        Mutation means altering data-creation parameters:

Input range: Expand/narrow or shift sampling interval for x.
Noise level: Increase or decrease σ (sigma) for added label noise.
Distribution type: Switch between, e.g., uniform and normal.
Sample size: Adjust n_g up/down.
Unit/scaling: Modify scaling coefficient for x or y.
Transform flags: Toggle augmentation options, e.g., enable/disable a specific data transform."""
        # Placeholder logic
        pass

    def to_vector(self):
        """Return generator parameters for clustering/species assignment."""
        return np.array([self.input_range[0], self.input_range[1], self.n_g, self.sigma_y, self.unit_scale])

    def __str__(self):
        return (f"Generator(mode={self.mode}, range={self.input_range}, "
                f"n_g={self.n_g}, sigma_y={self.sigma_y}, unit={self.unit_scale})")