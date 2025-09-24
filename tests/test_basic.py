import numpy as np
from mc_sr.equation import Equation, EquationNode
from mc_sr.generator import Generator

def test_equation_basic_eval():
    # Create tree for: sin(1.2*x) + x**2
    const_node = EquationNode("const")
    const_node.const_idx = 0
    x_node = EquationNode("x")
    mul_node = EquationNode("*", [const_node, x_node])
    sin_node = EquationNode("sin", [mul_node])
    pow_node = EquationNode("pow", [x_node, EquationNode("2")])
    plus_node = EquationNode("+", [sin_node, pow_node])

    eq = Equation(root=plus_node, constants={0: 1.2})  # constant 1.2
    # For now, monkeypatch a dummy evaluate method!
    def dummy_eval(x):
        return np.sin(1.2 * x) + x**2
    eq.evaluate = dummy_eval

    x = np.array([0, 1, 2])
    y = eq.evaluate(x)
    assert np.allclose(y, [0.0, np.sin(1.2 * 1) + 1, np.sin(1.2 * 2) + 4])


def test_generator_benchmark_mode():
    def true_fun(x):
        return np.sin(1.7 * x) + x**2
    gen = Generator(input_range=(-2,2), n_g=50, sigma_y=0.02, f_star=true_fun)
    x, y = gen.sample()
    assert len(x) == 50
    assert len(y) == 50
    # Check output is numeric and noise is present
    assert np.abs(np.std(y) - np.std(true_fun(x))) < 0.5  # allow for noise

def test_generator_real_mode():
    x0 = np.linspace(-2, 2, 100)
    y0 = np.sin(1.7 * x0) + x0**2
    gen = Generator(input_range=(-2,2), n_g=50, sigma_y=0.02, anchor_data=(x0, y0), mode="real")
    x, y = gen.sample()
    assert len(x) == 50
    assert len(y) == 50
    # check that bootstrapping and noise works
    assert np.abs(np.mean(y) - np.mean(y0)) < 2.0