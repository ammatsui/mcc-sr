""" Class for symbolic equations with tunable constants and tree structure."""

# import numpy for vectorized operations
import numpy as np
from scipy.optimize import least_squares
import random
import copy

operators = ['+', '-', '*', '/', 'sin', 'cos', 'pow']
binary_ops = ['+', '-', '*', '/', 'pow']
unary_ops = ['sin', 'cos']

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

    @staticmethod
    def random_init(max_depth=3,
                    operators=None,
                    p_const=0.5,
                    n_constants_range=(1, 4)):
        """
        Randomly initialize an equation (tree structure and constants).
        
        Parameters:
            max_depth: int, maximum tree depth.
            operators: list of operators to choose from.
            p_const: float, probability to generate a constant node.
            n_constants_range: tuple, min/max constants in the expression.
        Returns:
            Equation instance.
        """
        operators = operators or ['+', '-', '*', '/', 'sin', 'cos', 'pow']
        constants = {}
        const_indices = []

        def build_node(depth):
            if depth >= max_depth or (depth>0 and random.random()<0.3):
                # Terminal node
                if random.random() < p_const:
                    idx = len(const_indices)
                    const_indices.append(idx)
                    return EquationNode('const') #, const_idx=idx)
                else:
                    return EquationNode('x')
            else:
                op = random.choice(operators)
                if op in ['sin', 'cos']:
                    child = build_node(depth+1)
                    return EquationNode(op, [child])
                elif op == 'pow':
                    left = build_node(depth+1)
                    right = build_node(depth+1)
                    return EquationNode(op, [left, right])
                else:
                    # binary
                    left = build_node(depth+1)
                    right = build_node(depth+1)
                    return EquationNode(op, [left, right])

        root = build_node(0)
        # Initialize constants randomly, e.g. uniform [-2,2]
        for idx in const_indices:
            constants[idx] = np.random.uniform(-2,2)
        eq = Equation(root, constants=constants)
        print(eq)
        return eq

    def evaluate(self, x):
        """ToDo: add support for multiple variables"""
        """
        Evaluate the equation for given x values.
        x: float or np.ndarray
        Returns: float or np.ndarray
        """
        
        def _eval(node, const_counter=[0]):
            v = node.value
            #print("Evaluating node:", v)
            if v == 'x':
                return x
            elif v == 'const':
                # Constants numbered by order of appearance during tree construction
                idx = const_counter[0]
                const_counter[0] += 1
                #print("Using constant index:", idx, "value:", self.constants[idx])
                return self.constants[idx]
            elif v in {'+', '-', '*', '/', 'pow'}:
                left = _eval(node.children[0], const_counter)
                right = _eval(node.children[1], const_counter)
                #print(f"Operator: {v}, Left: {left}, Right: {right}")
                if v == '+': return left + right
                if v == '-': return left - right
                if v == '*': return left * right
                if v == '/':
                    # Avoid division by zero
                    # Add small epsilon to denominator if needed
                    # Robust division with per-element epsilon
                    right_safe = np.where(np.abs(right) < 1e-8, 1e-8, right)
                    return left / right_safe
                if v == 'pow':
                    # Avoid complex results for negative bases and non-integer exponents
                    with np.errstate(invalid="ignore"):
                        result = np.power(left, right)
                        # fallback: replace nan/inf with large value
                        result = np.where(np.isfinite(result), result, 1e6)
                    return result
            elif v in {'sin', 'cos'}:
                child = _eval(node.children[0], const_counter)
                #print(f"Applying {v} to {child}")
                if v == 'sin': return np.sin(child)
                if v == 'cos': return np.cos(child)
            else:
                raise ValueError(f"Unknown node value: {v}")

        # Always reset per call
       
        return _eval(self.root, [0])
    
    def calculate_mse(self, x_data, y_data):
        """Calculate MSE with current constants (no fitting)."""
        y_pred = self.evaluate(x_data)
        #print(y_pred)
        mse = np.mean((y_pred - y_data)**2)
        return mse
    
    

    def fit_constants(self, x, y, method="lsq"):
        """
        Fit or optimize the constants of the equation so that self.evaluate(x_data) matches y_data (min MSE).
        Returns MSE after fitting.
        """
        def residual(c):
            # Predict y using candidate constants
            y_pred = self.evaluate(x) #, constants=c)
            res = (y_pred - y).flatten()
            # Mask nans/infs for optimizer
            res = np.where(np.isfinite(res), res, 1e6)
            return res
        
        def count_param_nodes(node):
            count = 1  # Each node has a constant
            for child in node.children:
                count += count_param_nodes(child)
            return count
        
        # Initial guess: use current constants or 1s
        initial_c = np.array(
            [self.constants.get(i, 1.0) for i in range(len(self.constants))]
            ) if isinstance(self.constants, dict) else np.array(self.constants)
        if initial_c.size == 0:  # fallback if not set
            # number of constants = number of nodes in the tree
            num_consts = count_param_nodes(self.root)
            initial_c = np.ones(num_consts)
        
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
        Actions:
        - Insert: Add a new random operator or sub-expression at a node.
        - Delete: Remove a sub-tree or node.
        - Substitute: Replace an operator, a terminal, or a constant.
        - Perturb constants: Slightly change a numeric constant.
        """
        import random
        actions = ['insert', 'delete', 'substitute', 'perturb']
        action = random.choice(actions)

        mutant = copy.deepcopy(self)

        def get_all_nodes(node, parent=None, nodes=None):
            if nodes is None:
                nodes = []
            nodes.append((node, parent))
            for child in node.children:
                get_all_nodes(child, node, nodes)
            return nodes

        nodes = get_all_nodes(mutant.root)

        if action == 'insert':
            # Insert a random operator/subtree at a random node
            op_choices = ['+', '-', '*', '/', 'sin', 'cos', 'pow']
            op = random.choice(op_choices)
            target, _ = random.choice(nodes)
            # For binary ops, add two children; for unary, one
            if op in ['sin', 'cos']:
                new_child = EquationNode(random.choice(['x', 'const']))
                new_node = EquationNode(op, [new_child])
            # binary 
            else:
                left = EquationNode(random.choice(['x', 'const']))
                right = EquationNode(random.choice(['x', 'const']))
                new_node = EquationNode(op, [left, right])
                assert len(new_node.children) == 2
            # Insert as a new child (or replace one child if possible)
            target.children.append(new_node)

        elif action == 'delete':
            # Remove a random node (not root)
            non_root_nodes = [(n, p) for n, p in nodes if p is not None]
            candidates = []
            for n, p in non_root_nodes:
                if p.value in binary_ops:
                    # Deleting a child would leave <2 children: illegal, skip
                    if len(p.children) <= 2:
                        continue
                if p.value in unary_ops:
                    # Deleting a child would leave <1 child: illegal, skip
                    if len(p.children) <= 1:
                        continue
                candidates.append((n, p))
            if candidates:
                node_to_delete, parent = random.choice(candidates)
                parent.children = [c for c in parent.children if c != node_to_delete]
           

        elif action == 'substitute':
            # Replace a random node's value
            node_to_sub, _ = random.choice(nodes)
            # Change operator, variable, or constant
            if node_to_sub.value in unary_ops:
                node_to_sub.value = random.choice(unary_ops)
            elif node_to_sub.value in binary_ops:
                node_to_sub.value = random.choice(binary_ops)
            elif node_to_sub.value == 'x':
                node_to_sub.value = 'const'
            elif node_to_sub.value == 'const':
                node_to_sub.value = 'x'

        elif action == 'perturb':
            # Slightly change a random constant
            if isinstance(mutant.constants, dict) and mutant.constants:
                k = random.choice(list(mutant.constants.keys()))
                mutant.constants[k] += np.random.normal(scale=0.1)
            elif isinstance(mutant.constants, (list, np.ndarray)) and len(mutant.constants) > 0:
                idx = random.randint(0, len(mutant.constants)-1)
                mutant.constants[idx] += np.random.normal(scale=0.1)

        return mutant
    
    def to_prefix(self):
        """
        Return list of tokens (prefix notation) for the tree, useful for speciation/clustering.
        """
        # Placeholder: implement tree traversal here
        raise NotImplementedError("Prefix conversion not implemented")
    
    def _to_infix(self, node):
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