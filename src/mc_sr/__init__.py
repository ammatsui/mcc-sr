from equation import Equation
from generator import Generator
from evolution import EvolutionEngine
from logger import MetricLogger
import numpy as np
import json
import random



def load_dataset(path):
    # Loads anchor dataset D0 from SRBench .csv or .json
    # Returns X shape [n, d], y shape [n,]
    data = np.loadtxt(path, delimiter=' ', skiprows=0)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# if __name__ == "__main__":
#     D0= load_dataset('./dataset/I.6.2.txt') # user supplies filename
#     print("Anchor dataset loaded.")
#     print("X shape:", D0[0].shape)
#     equation = Equation.random_init(D0[0].shape[1])
#     print("Random equation initialized:")
#     print(equation)

# print("Before fitting:", equation)
# mse_before = equation.calculate_mse(D0[0], D0[1])
# print("MSE before:", mse_before)

# mse_after = equation.fit_constants(D0[0], D0[1])
# print("After fitting:", equation)
# print("MSE after:", mse_after)

#     y = equation.evaluate(D0[0])
#     print("Equation evaluated on anchor data.")
#     print(y)


#     equation.fit_constants(D0[0], D0[1])
#     print("Constants fitted.")
#     print(equation)

#     print("Mutated equation:")
#     mutant = equation.mutate()
#     print(mutant)
#     print("Mutant evaluated on anchor data.")
#     print(mutant.evaluate(D0[0]))
    
#     generator = Generator.random_init(anchor_data=D0)
#     print("Random generator initialized:")
#     print(generator)

#     xg, yg = generator.sample()
#     print("Generator sampled data:")
#     print(xg)
#     print(yg)

#     print("Mutated generator:")
#     mutantg = generator.mutate()
#     print(mutantg)
#     print("Mutant sampled.")
#     xg, yg = mutantg.sample()
#     print("Generator sampled data:")
#     print(xg)
#     print(yg)



if __name__ == "__main__":
    # Load anchor dataset D0
    print("Loading anchor dataset...")
    D0 = load_dataset('./dataset/I.6.2.txt') # user supplies filename

    # Initial seeds (example)
    # --- Create queues ---
    equation_queue = [Equation.random_init(D0[0].shape[1]) for _ in range(5)]
   # gen = Generator(anchor_data = D0)
    generator_queue = [Generator.random_init(anchor_data = D0) for _ in range(5)]

    # Params
    params = {
        'tau_init': 1e-2, 
        'taup_init': 1e-2,
        'Lmax': 8,
        'n_generations': 100,
        'delta_eq': 0.2,
        'delta_gen': 0.25,
        'k_generators': 3,
        'tau_quantile': 0.25,
        'llm_trigger_gens': 30
    }

     # --- Logger ---
    logger = MetricLogger()

    # --- Evolution settings ---
    tau = 0.05
    tau_prime = 0.07
    L_max = 40

    engine = EvolutionEngine(
        equation_queue=equation_queue,
        generator_queue=generator_queue,
        anchor_x=D0[0],
        anchor_y=D0[1],
        tau=tau,
        tau_prime=tau_prime,
        L_max=L_max,
        n_generations=5,
        batch_size=2,
        logger=logger,
        llm_enabled=False
    )

    eqs, gens = engine.run()

    """Comparison!!!"""

#     from pysr import PySRRegressor

#     model = PySRRegressor(
#     niterations=40,                  # Main compute knob (higher = better models)
#     populations=5,
#     model_selection="best",          # Option: "best" or "accuracy"
#     unary_operators=["sin", "cos"],  # Operator set
#     binary_operators=["+", "-", "*", "/"],
# )

#     model.fit(D0[0], D0[1])

#     print("PySR discovered equations:")

#     print(model.get_best())
# Prints the best discovered equation

# print(model.equations_)  
# DataFrame with all tried equations, scores, complexities, etc.

# score = model.score(D0[0], D0[1])
# print("MSE:", score)
    
    # # --- Save results to files ---
    # with open("results_equations.txt", "w") as f_eq:
    #     for eq in eqs:
    #         f_eq.write(str(eq) + "\n")
            
    # with open("results_generators.txt", "w") as f_gen:
    #     for gen in gens:
    #         f_gen.write(str(gen) + "\n")
    
    # with open("results_metrics.json", "w") as f_metrics:
    #     json.dump(metrics, f_metrics, indent=2)

    # print("Results saved to results_equations.txt, results_generators.txt, and results_metrics.json")


print("Done.")
# __all__ = [
#     "Equation",
#     "Generator",
#     "EvolutionEngine",
#     "run_evolution_demo",
# ]