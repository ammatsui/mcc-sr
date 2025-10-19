class MetricLogger:
    def __init__(self, log_file="log.txt"):
        self.records = []
        self.log_file = log_file

    def log_generation(self, generation, tau, tau_prime, eq_passed, gen_passed, 
                       eq_queue_size, gen_queue_size, equation_population, mse, generators_population):
        log_entry = dict(
            generation=generation,
            tau=tau,
            tau_prime=tau_prime,
            eq_passed=eq_passed,
            gen_passed=gen_passed,
            eq_queue_size=eq_queue_size,
            gen_queue_size=gen_queue_size,
            equation_population = equation_population,
            mse = mse,
            generators_population = generators_population
        )
        self.records.append(log_entry)
        if self.log_file:
            # Optionally write to file/CSV immediately
            # import json
            # with open(self.log_file, "a") as f:
            #     f.write(json.dumps(log_entry) + '\n')
            with open(self.log_file, "a") as f:  # 'a' mode appends
                f.write(f"Gen {generation}: {eq_passed} equations and {gen_passed} generators passed MC.\n")
                f.write(f"  Queue sizes - Equations: {eq_queue_size}, Generators: {gen_queue_size}\n")
                f.write(f"Equations: {[str(eq) for eq in equation_population]}\n")
                f.write(f"MSE: {mse}\n")
                f.write(f"Generators: {[str(gen) for gen in generators_population]}\n")
                

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)