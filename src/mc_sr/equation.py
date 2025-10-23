""" Class for symbolic equations with tunable constants and tree structure."""

# import numpy for vectorized operations
import numpy as np
from scipy.optimize import least_squares
import random
import copy

PENALTY_VALUE = 1e6

operators = ['+', '-', '*', '/', 'sin', 'cos', 'pow']
binary_ops = ['+', '-', '*', '/', 'pow']
unary_ops = ['sin', 'cos']

class EquationNode:
    """
    A node in the expression tree.
    Each node can be an operator (like '+', '*', 'sin') or a terminal (like 'x' or a constant).
    """
    def __init__(self, value, const_idx=None, children=None):
        self.value = value  # operator, variable, or constant
        self.const_idx = const_idx
        self.children = children or []  # list of child nodes
        if self.value in binary_ops:
            assert len(self.children) == 2, f"Binary operator {self.value} must have 2 children."
        elif self.value in unary_ops:
            assert len(self.children) == 1, f"Unary operator {self.value} must have 1 child."

class Equation:
    """
    Represents a symbolic equation as an expression tree with tunable constants.
    """
    def __init__(self, root, n_variables = 1, constants=None):
        """
        root: EquationNode, the root of the tree.
        constants: dict mapping names or indices to values (could be a list, array, or dict).
        """
        self.root = root
        self.n_variables = n_variables # number of input variables
        self.constants = constants if constants is not None else {}

    @staticmethod
    def random_init(n_vars, max_depth=3,
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
        var_nodes = [f'x{i}' for i in range(n_vars)]
        # print("Variable nodes:", var_nodes)

        def build_node(depth):
            if depth >= max_depth or (depth>0 and random.random()<0.3):
                # Terminal node
                if random.random() < p_const:
                    idx = len(const_indices)
                    const_indices.append(idx)
                    return EquationNode('const', const_idx=idx)
                else:
                    var = random.choice(var_nodes)
                    return EquationNode(var)
            else:
                op = random.choice(operators)
                if op in ['sin', 'cos']:
                    child = build_node(depth+1)
                    return EquationNode(op, children = [child])
                elif op == 'pow':
                    left = build_node(depth+1)
                    right = build_node(depth+1)
                    return EquationNode(op, children = [left, right])
                else:
                    # binary
                    left = build_node(depth+1)
                    right = build_node(depth+1)
                    return EquationNode(op, children = [left, right])

        root = build_node(0)
        # Initialize constants randomly, e.g. uniform [-2,2]
        for idx in const_indices:
            constants[idx] = np.random.uniform(-2,2)
        eq = Equation(root, n_variables=n_vars, constants=constants)
        # print(eq)
        assert eq.n_variables == n_vars
        assert eq is not None
        return eq

    def evaluate(self, x):
        """ToDo: add support for multiple variables"""
        """
        Evaluate the equation for given x values.
        x: float or np.ndarray
        Returns: float or np.ndarray
        """
        # print("Data shape:", x.shape)
        # print("Data:", x)
        assert x.ndim == 2 and x.shape[1] == self.n_variables , f"Input x must have shape (n_samples, {self.n_variables}), but received shape {x.shape}"
        n_samples = x.shape[0]
        def _eval(node): #, const_counter=[0]):
            v = node.value
            #print("Evaluating node:", v)
            if v == 'const':
                # Constants numbered by order of appearance during tree construction
                # the value is stored in the node
                idx = node.const_idx
                # const_counter[0] += 1
                #print("Using constant index:", idx, "value:", self.constants[idx])
                assert self.constants[idx] is not None, f"Constant {idx} not set."
                return np.full(n_samples, self.constants[idx])
            elif v.startswith('x'):
                vi = int(v[1:])
                res = x[:, vi]
                assert res.shape == (n_samples,), f"Variable x{vi} output shape {res.shape}"
                return x[:, vi]
            elif v in {'+', '-', '*', '/', 'pow'}:
                left = _eval(node.children[0])#, const_counter)
                right = _eval(node.children[1])#, const_counter)
                #print(f"Operator: {v}, Left: {left}, Right: {right}")
                assert left.shape == (n_samples,), f"left output shape {left.shape} for op '{v}'"
                assert right.shape == (n_samples,), f"right output shape {right.shape} for op '{v}'"
                if v == '+': return left + right
                if v == '-': return left - right
                if v == '*': return left * right
                if v == '/':
                    # Avoid division by zero
                    # Add small epsilon to denominator if needed
                    # Robust division with per-element epsilon
                    right_safe = np.where(np.abs(right) < 1e-6, np.sign(right)*1e-6, right)
                    out = left / right_safe
                    out = np.where(np.isfinite(out), out, PENALTY_VALUE)
                   
                    return out
                if v == 'pow':
                    # Avoid complex results for negative bases and non-integer exponents
                    # with np.errstate(invalid="ignore"):
                    #     result = np.power(left, right)
                    #     # fallback: replace nan/inf with large value
                    #     result = np.where(np.isfinite(result), result, 1e6)
                    # return result
                    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                        # Limit exponents to avoid overflow (clip at reasonable large number)
                        left_clip = np.clip(left, -1e2, 1e2)
                        right_clip = np.clip(right, -10, 10)
                        result = np.power(left_clip, right_clip)
                        result = np.where(np.isfinite(result), result, PENALTY_VALUE)
                    return result
            if v == 'sin': 
                child = _eval(node.children[0])
                assert child.shape == (n_samples,), f"child output shape {child.shape} for op '{v}'"
           
                out = np.sin(child)
                out = np.where(np.isfinite(out), out, PENALTY_VALUE)
                return out
            if v == 'cos': 
                child = _eval(node.children[0])
                assert child.shape == (n_samples,), f"child output shape {child.shape} for op '{v}'"
            
                out = np.cos(child)
                out = np.where(np.isfinite(out), out, PENALTY_VALUE)
                return out
            # elif v in {'sin', 'cos'}:
            #     child = _eval(node.children[0])#, const_counter)
            #     assert child.shape == (n_samples,), f"child output shape {child.shape} for op '{v}'"
            #     #print(f"Applying {v} to {child}")
            #     if v == 'sin': return np.sin(child)
            #     if v == 'cos': return np.cos(child)
            else:
                raise ValueError(f"Unknown node value: {v}")

        # Always reset per call
       
        return _eval(self.root)#, [0])
    
    def calculate_mse(self, x_data, y_data):
        """Calculate MSE with current constants (no fitting)."""
        y_pred = self.evaluate(x_data)
        residuals = y_pred - y_data
        residuals = np.where(np.isfinite(residuals), residuals, 0)  # Replace inf/nan with 0 or some sentinel
        mse = np.mean(residuals ** 2)
        # mse = np.mean((y_pred - y_data)**2)
        return mse
    
    def collect_used_const_indices(self):
        used = set()
        def walk(node):
            if node.value == 'const':
                used.add(node.const_idx)
            for child in node.children:
                walk(child)
        walk(self.root)
        return used

    def fit_constants(self, x, y, method="lsq", n_restarts=5, bounds=(-10,10)):
        """
        Fit or optimize the constants of the equation so that self.evaluate(x_data) matches y_data (min MSE).
        Returns MSE after fitting.
        """
        def residual(c):
            # Predict y using candidate constants
            used_indices = self.collect_used_const_indices()
            self.constants = {idx: c[i] for i, idx in enumerate(used_indices)}
            y_pred = self.evaluate(x) #, constants=c)
            
            y_pred = np.asarray(y_pred).flatten()
            y1 = np.asarray(y).flatten()
            if y_pred.shape != y1.shape:
                raise ValueError(f"Shape mismatch: y_pred {y_pred.shape}, y {y.shape}")
            
            res = (y_pred - y1).flatten()
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
        # print("Initial constants for fitting:", self.constants)
        # # Run non-linear least squares fit
        # result = least_squares(residual, initial_c)
        # # Update self.constants to best found
        # self.constants = self.constants = {k:v for k, v in enumerate(result.x)}
        # # Compute and return MSE
        # mse = np.mean(residual(result.x)**2)
        # return mse

        # add restarts to optimise constant fitting
        best_mse = np.inf
        best_constants = initial_c
        for restart in range(n_restarts):
            initial_guess = np.random.uniform(bounds[0], bounds[1], size=initial_c.size)
            assert initial_guess.shape == initial_c.shape
            try:
                result = least_squares(residual, initial_guess) #, initial_guess bounds=bounds)
                mse = np.mean(residual(result.x)**2)
                if mse < best_mse:
                    best_mse = mse
                    best_constants = result.x
                    # print(f"Restart {restart}: Found better MSE {best_mse}")
                    # print(f"Constants: {best_constants}")
                    assert best_constants.shape == initial_c.shape
            except Exception as e:
                # Optionally log exceptions
                pass
        assert best_constants is not None
        res_const = {}
        for k in (self.collect_used_const_indices()):
            for v in best_constants:
                res_const[k] = v
        # res_const = {k: v for k, v in (self.collect_used_const_indices(), best_constants)}
        assert len(self.constants) == len(res_const), f"with {len(self.constants)} vs {len(res_const)} constants after fitting"
        self.constants = res_const
        return best_mse
    

    
    def random_terminal(self):
        if random.random() < 0.5:
            # Find new unique const_idx
            used = set()
            def collect(node):
                if node.value == 'const':
                    used.add(node.const_idx)
                for child in node.children:
                    collect(child)
            collect(self.root)
            max_idx = max(used) if used else -1
            new_idx = max_idx + 1
            self.constants[new_idx] = np.random.uniform(-2,2)
            return EquationNode('const', const_idx=new_idx)
        else:
            var_idx = random.randint(0, self.n_variables - 1)
            return EquationNode(f'x{var_idx}')
    
    def remove_unused_constants(self):
        used = set()
        def collect(node):
            if node.value == 'const':
                used.add(node.const_idx)
            for child in node.children:
                collect(child)
        collect(self.root)
        # Keep only the constants in use
        self.constants = {k: v for k, v in self.constants.items() if k in used}

     

    def mutate(self, action=None, node=None, value=None):
        """
        Randomly mutate tree structure, operator, terminals or constants.
        Actions:
        - Insert: Add a new random operator or sub-expression at a node.
        - Delete: Remove a sub-tree or node.
        - Substitute: Replace an operator, a terminal, or a constant.
        - Perturb constants: Slightly change a numeric constant.
        """
        import random
        actions = ['insert', 'delete', 'substitute', 'perturb', 'grow']
        if action is None:
            action = random.choice(actions)

        terminals = ['const'] + [f'x{i}' for i in range(self.n_variables)]

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
            if value is None:
                op_choices = ['+', '-', '*', '/', 'sin', 'cos', 'pow']
                op = random.choice(op_choices)
            else:
                op = value
            # Choose a random target node to insert at
            if node is None:
                target, _ = random.choice(nodes)
            else:
                target = node
            # For binary ops, add two children; for unary, one
            if op in ['sin', 'cos']:
                new_child = mutant.random_terminal()
                new_node = EquationNode(op, children = [new_child])
            # binary 
            else:
                left = mutant.random_terminal()
                right = mutant.random_terminal()
                new_node = EquationNode(op, children = [left, right])
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
                if node is None:
                    node_to_delete, parent = random.choice(candidates)
                else:
                    node_to_delete = node
                    parent = next((p for n, p in candidates if n == node), None)
                parent.children = [c for c in parent.children if c != node_to_delete]
            mutant.remove_unused_constants()

        elif action == 'substitute':
            # Replace a random node's value
            if node is None:
                node_to_sub, _ = random.choice(nodes)
            else:
                node_to_sub = node
            # Change operator, variable, or constant
            if node_to_sub.value in unary_ops:
                node_to_sub.value = random.choice(unary_ops)
            elif node_to_sub.value in binary_ops:
                node_to_sub.value = random.choice(binary_ops)

            elif node_to_sub.value.startswith('x') or node_to_sub.value == 'const':
                terminal_nodes = [(n, p) for n, p in nodes if not n.children]
                if terminal_nodes:
                    node_to_sub, _ = random.choice(terminal_nodes)
                    old_val = node_to_sub.value
                    new_val = random.choice(terminals)
                    while new_val == old_val:
                        new_val = random.choice(terminals)
                    # CASE 1: Substituting FROM variable TO const
                    if old_val.startswith('x') and new_val == 'const':
                        # Find unused const_idx
                        used_const = set()
                        def collect(node):
                            if node.value == 'const' and node.const_idx is not None:
                                used_const.add(node.const_idx)
                            for child in node.children:
                                collect(child)
                        collect(mutant.root)
                        max_idx = max(used_const) if used_const else -1
                        new_idx = max_idx + 1
                        node_to_sub.value = 'const'
                        node_to_sub.const_idx = new_idx
                        mutant.constants[new_idx] = np.random.uniform(-2,2)
                    # CASE 2: Substituting FROM const TO variable
                    elif old_val == 'const' and new_val.startswith('x'):
                        # Remove const_idx reference
                        if node_to_sub.const_idx is not None:
                            # (The constant will be cleaned up in remove_unused_constants())
                            node_to_sub.const_idx = None
                        node_to_sub.value = new_val
                    # CASE 3: Variable-to-variable or const-to-const
                    else:
                        node_to_sub.value = new_val
                        # If it's a new 'const', assign index as above
                        if new_val == 'const':
                            if node_to_sub.const_idx is None:
                                used_const = set()
                                def collect(node):
                                    if node.value == 'const' and node.const_idx is not None:
                                        used_const.add(node.const_idx)
                                    for child in node.children:
                                        collect(child)
                                collect(mutant.root)
                                max_idx = max(used_const) if used_const else -1
                                new_idx = max_idx + 1
                                node_to_sub.const_idx = new_idx
                                mutant.constants[new_idx] = np.random.uniform(-2,2)
                        # If it's a variable, drop old const_idx
                        elif new_val.startswith("x"):
                            node_to_sub.const_idx = None
                # After substitution, remove any no-longer-used constants
                mutant.remove_unused_constants()

        elif action == 'perturb':
            # Slightly change a random constant
            if isinstance(mutant.constants, dict) and mutant.constants:
                k = random.choice(list(mutant.constants.keys()))
                mutant.constants[k] += np.random.normal(scale=0.1)
            elif isinstance(mutant.constants, (list, np.ndarray)) and len(mutant.constants) > 0:
                idx = random.randint(0, len(mutant.constants)-1)
                mutant.constants[idx] += np.random.normal(scale=0.1)

        elif action == 'grow':
            # replace random node with a random subtree of random length
            if node is None:
                node, parent = random.choice(nodes)  
            
            new_subtree = Equation.random_init(
                n_vars=self.n_variables, max_depth=np.random.randint(0,10)
            )

            # insert new_subtree in place of node
            if parent is None:
                mutant.root = new_subtree.root
            else:
                for i, child in enumerate(parent.children):
                    if child == node:
                        parent.children[i] = new_subtree.root
                        break
            # fix constants sync
            def reindex_subtree_constants(subtree_root, new_constants, mutant):
            # subtree_root: root node of the newly attached subtree
            # new_constants: dict from new_subtree.constants (old_idx -> value)
            # mutant: equation instance receiving the subtree

                def traverse(node):
                    # Do a recursive traversal
                    if node.value == 'const':
                        # Assign new globally unique index
                        max_idx = max(mutant.constants.keys()) if mutant.constants else -1
                        new_idx = max_idx + 1
                        old_idx = node.const_idx  # index from new_subtree
                        node.const_idx = new_idx  # update the node to use new index
                        mutant.constants[new_idx] = new_constants.get(old_idx, np.random.uniform(-2,2))
                    for child in node.children:
                        traverse(child)

                traverse(subtree_root)

            # After attaching new_subtree.root in mutant...
            reindex_subtree_constants(new_subtree.root, new_subtree.constants, mutant)


            
        
        assert mutant is not None
        assert mutant.n_variables == self.n_variables
        # self.remove_unused_constants()
        mutant.remove_unused_constants()
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
                idx = node.const_idx
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
        def count_nodes(node):
            total = 1  # Count this node
            for child in node.children:
                total += count_nodes(child)
            return total
        return count_nodes(self.root)
    
    
    
def crossover(eq1, eq2):
    """
    Perform subtree crossover between two Equation instances, ensuring operator arity is preserved.
    Returns two offspring (deepcopies).
    """
    import copy
    offspring1 = copy.deepcopy(eq1)
    offspring2 = copy.deepcopy(eq2)

    # Helper: get all (node, parent, child_index) triples
    def get_all_nodes(node, parent=None, nodes=None):
        if nodes is None:
            nodes = []
        if parent is not None:
            idx = parent.children.index(node)
        else:
            idx = None
        nodes.append((node, parent, idx))
        for child in node.children:
            get_all_nodes(child, node, nodes)
        return nodes

    nodes1 = get_all_nodes(offspring1.root)
    nodes2 = get_all_nodes(offspring2.root)

    # Only consider non-root nodes for crossover
    nonroot_nodes1 = [(n, p, idx) for n, p, idx in nodes1 if p is not None]
    nonroot_nodes2 = [(n, p, idx) for n, p, idx in nodes2 if p is not None]

    # Filter candidates so swapped subtrees have compatible arity for their parent location
    candidates = []
    for n1, p1, idx1 in nonroot_nodes1:
        for n2, p2, idx2 in nonroot_nodes2:
            # Parent op expects a child of a specific arity; child at swap point must match
            if len(n1.children) == len(n2.children):  # same arity
                candidates.append(((n1, p1, idx1), (n2, p2, idx2)))

    if candidates:
        # Pick a random compatible pair
        (n1, p1, idx1), (n2, p2, idx2) = random.choice(candidates)
        # Swap the subtrees
        p1.children[idx1], p2.children[idx2] = n2, n1

        # traverse trees to sync new constants
        crossed_nodes1 = get_all_nodes(n2)
        crossed_nodes2 = get_all_nodes(n1)
        for node, _, _ in crossed_nodes1:
            # for constants in offspring1, assign constant value from eq2 and vice versa
            # also update the const_idx 
            if node.value == 'const':
                idx = node.const_idx
                # Assign a new constant value
                max_idx = max(offspring1.constants.keys()) if offspring1.constants else -1
                new_idx = max_idx + 1
                node.const_idx = new_idx
                offspring1.constants[new_idx] = eq2.constants[idx]

        for node, _, _ in crossed_nodes2:
            if node.value == 'const':
                idx = node.const_idx
                max_idx = max(offspring2.constants.keys()) if offspring2.constants else -1
                new_idx = max_idx + 1
                node.const_idx = new_idx
                offspring2.constants[new_idx] = eq1.constants[idx]

        offspring1.remove_unused_constants()
        offspring2.remove_unused_constants()
    # else: No compatible crossover points found; return deep copies unchanged

    return offspring1, offspring2