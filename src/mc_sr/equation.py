""" Class for symbolic equations with tunable constants and tree structure."""

# import numpy for vectorized operations
import numpy as np
from scipy.optimize import least_squares

class EquationNode:
    """
    A node in the expression tree.
    Each node can be an operator (like '+', '*', 'sin') or a terminal (like 'x' or a constant).
    """
    def __init__(self, value, children=None):
        self.value = value  # operator, variable, or constant
        self.children = children or []  # list of child nodes

class Equation:
    """
    Represents a symbolic equation as an expression tree with tunable constants.
    """
    def __init__(self, root, constants=None):
        """
        root: EquationNode, the root of the tree.
        constants: dict mapping names or indices to values (could be a list, array, or dict).
        """
        self.root = root
        self.constants = constants if constants is not None else {}

    def evaluate(self, x):
        """
        Evaluate the equation for given x values.
        x: float or np.ndarray
        Returns: float or np.ndarray
        """
        # Simple parser for demonstration (supports +, -, *, /, sin, cos, power, x, constants)
        stack = []
        const_idx = 0
        for token in self.tokens:
            if token == 'x':
                stack.append(x)
            elif token.startswith('c') and token[1:].isdigit():
                idx = int(token[1:])
                stack.append(self.constants[idx])
            elif token == '+':
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)
            elif token == '-':
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
            elif token == '*':
                b = stack.pop()
                a = stack.pop()
                stack.append(a * b)
            elif token == '/':
                b = stack.pop()
                a = stack.pop()
                stack.append(a / b)
            elif token == 'sin':
                a = stack.pop()
                stack.append(np.sin(a))
            elif token == 'cos':
                a = stack.pop()
                stack.append(np.cos(a))
            elif token == 'power':
                b = stack.pop()
                a = stack.pop()
                stack.append(np.power(a, b))
            else:
                raise ValueError(f"Unknown token: {token}")
        if len(stack) != 1:
            raise ValueError("Invalid equation structure")
        return stack[0]
    
    def calculate_mse(self, x_data, y_data):
        """Calculate MSE with current constants (no fitting)."""
        y_pred = self.evaluate(x_data)
        mse = np.mean((y_pred - y_data)**2)
        return mse
    
    def fit_constants(self, x, y, method="lsq"):
        """
        Fit or optimize the constants of the equation so that self.evaluate(x_data) matches y_data (min MSE).
        Returns MSE after fitting.
        """
        def residual(c):
            # Predict y using candidate constants
            y_pred = self.evaluate(x, constants=c)
            return y_pred - y
        
        # Initial guess: use current constants or 1s
        initial_c = np.array(
            [self.constants.get(i, 1.0) for i in range(len(self.constants))]
            ) if isinstance(self.constants, dict) else np.array(self.constants)
        if initial_c.size == 0:  # fallback if not set
            initial_c = np.ones(2)
        
        # Run non-linear least squares fit
        result = least_squares(residual, initial_c)
        # Update self.constants to best found
        self.constants = result.x
        # Compute and return MSE
        mse = np.mean(residual(result.x)**2)
        return mse

    def mutate(self):
        """
        Randomly mutate tree structure, operator, terminals or constants.
        Useful for evolutionary search.
        Randomly altering the syntax tree involves:

Insert: Add a new random operator or sub-expression at a node.
Delete: Remove a sub-tree or node.
Substitute: Replace an operator, a terminal, or a constant.
Perturb constants: Slightly change a numeric constant.

        """
        # Placeholder: implement mutation logic
        raise NotImplementedError("Mutation not implemented")
    
    def to_prefix(self):
        """
        Return list of tokens (prefix notation) for the tree, useful for speciation/clustering.
        """
        # Placeholder: implement tree traversal here
        raise NotImplementedError("Prefix conversion not implemented")
    
    def to_infix(self, node):
        """
        Recursively convert tree to infix notation string.
        Supports unary and binary operators.
        """
        # Terminal node: variable or constant
        if not node.children:
            if node.value == "const":
                # Assume constants are indexed by order
                idx = getattr(node, "const_idx", 0)
                return str(self.constants[idx])
            return str(node.value)
        
        # Unary operators (e.g., sin, cos)
        if len(node.children) == 1:
            return f"{node.value}({self._to_infix(node.children[0])})"
        
        # Binary operators (e.g., +, -, *, /, pow)
        if len(node.children) == 2:
            left = self._to_infix(node.children[0])
            right = self._to_infix(node.children[1])
            # For 'pow', print as **
            if node.value == "pow":
                return f"({left})**({right})"
            return f"({left} {node.value} {right})"
        
        # Unknown arity
        return f"{node.value}({', '.join(self._to_infix(child) for child in node.children)})"
    
    def __str__(self):
        """
        Pretty-print the equation in infix notation.
        """
        return f"Equation constants: {self.constants}\nTree:  {self._to_infix(self.root)}"

    # def __str__(self):
    #     """
    #     String representation of the equation (for debugging and logging).
    #     """
    #     # Could print as infix or prefix notation, include constants
    #     return f"Equation constants: {self.constants}\nTree: {self.to_prefix()}"

    # -- additional methods you might need --
    def depth(self):
        """ Return the max depth of the tree. """
        # Placeholder
        raise NotImplementedError("Tree depth not implemented")
    
    def size(self):
        """ Return the number of nodes in the tree. """
        # Placeholder
        raise NotImplementedError("Tree size not implemented")