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
    data = np.genfromtxt(path, delimiter='\t', skip_header=1)
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

#     print("Before fitting:", equation)
#     mse_before = equation.calculate_mse(D0[0], D0[1])
#     print("MSE before:", mse_before)

#     mse_after = equation.fit_constants(D0[0], D0[1])
#     print("After fitting:", equation)
#     print("MSE after:", mse_after)

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

#     equation.evaluate(xg)
#     print("Equation evaluated on generated data.")
#     print(equation.evaluate(xg))
#     print("Mutated generator:")
#     mutantg = generator.mutate()
#     print(mutantg)
#     print("Mutant sampled.")
#     xg, yg = mutantg.sample()
#     print("Generator sampled data:")
#     print(xg)
#     print(yg)
#     print("Equation evaluated on mutated generated data.")
#     print(equation.evaluate(xg))



if __name__ == "__main__":
    # Load anchor dataset D0
    print("Loading anchor dataset...")
    D0 = load_dataset('C:/Users/matsu/Downloads/Hill_Valley_with_noise.tsv.gz') # user supplies filename
    D0 = (D0[0], D0[1])  #
    print("Anchor dataset loaded.")

    assert D0[0].shape[0] ==D0[1].shape[0] 


    n_pop = 10 #25
    
    import os
    if os.path.exists("log.txt"):
            os.remove("log.txt")

    if os.path.exists("generation_log.csv"):
            os.remove("generation_log.csv")

    # Initial seeds (example)
    # --- Create queues ---
    equation_queue = [Equation.random_init(D0[0].shape[1]) for _ in range(n_pop)]
   # gen = Generator(anchor_data = D0)
    generator_queue = [Generator.random_init(anchor_data = D0, mode = 'real') for _ in range(n_pop)]
    print("Created initial equation and generator queues.")
    # Params
    params = {
        'tau_init': 1e-2, 
        'taup_init': 1e-2,
        'Lmax': 8,
        'n_generations': 50,
        'delta_eq': 0.2,
        'delta_gen': 0.25,
        'k_generators': 3,
        'tau_quantile': 0.25,
        'llm_trigger_gens': 30
    }

     # --- Logger ---
    logger = MetricLogger()

    # --- Evolution settings ---
    tau = 0.3 #0.25 0.05
    tau_prime = 0.35 #0.3 0.07
    L_max = 40

    engine = EvolutionEngine(
        equation_queue=equation_queue,
        generator_queue=generator_queue,
        anchor_x=D0[0],
        anchor_y=D0[1],
        tau=tau,
        tau_prime=tau_prime,
        L_max=L_max,
        n_generations=50,
        batch_size=n_pop,
        logger=logger,
        llm_enabled=False
    )

    eqs, gens = engine.run()

    from plotting import plot_metrics

    log_file = "generation_log.csv"
    plot_metrics(log_file)

    

#     """Comparison!!!"""
#     import os
#     # os.environ["JULIA_EXE"] = r"C:\\Users\\matsu\\AppData\\Local\\Programs\\Julia-1.10.10\\bin\\julia.exe"
#     # print("Environment PATH:")
#     # print(os.environ.get("PATH"))
#     # print(os.environ.get("JULIA_EXE"))
#     # print("Importing PySR...")


    from pysr import PySRRegressor


    model = PySRRegressor(
    niterations=50,                  # Main compute knob (higher = better models)
    populations=n_pop,
    model_selection="best",          # Option: "best" or "accuracy"
    unary_operators=["sin", "cos"],  # Operator set
    binary_operators=["+", "-", "*", "/"],
    maxsize=40,                     # Max tree size
)

    model.fit(D0[0], D0[1])

    print("PySR discovered equations:")

    print(model.get_best())

  
    print(model.equations_)  

    score = model.score(D0[0], D0[1])
    print("MSE:", score)
    
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