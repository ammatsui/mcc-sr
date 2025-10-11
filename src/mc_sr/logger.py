class MetricLogger:
    def __init__(self, log_file="log.json"):
        self.records = []
        self.log_file = log_file

    def log_generation(self, generation, tau, tau_prime, eq_passed, gen_passed, 
                       eq_queue_size, gen_queue_size):#, eqs, gens):
        log_entry = dict(
            generation=generation,
            tau=tau,
            tau_prime=tau_prime,
            eq_passed=eq_passed,
            gen_passed=gen_passed,
            eq_queue_size=eq_queue_size,
            gen_queue_size=gen_queue_size,
            # equation_population = eqs,
            # generators_population = gens
        )
        self.records.append(log_entry)
        if self.log_file:
            # Optionally write to file/CSV immediately
            import json
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + '\n')

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)