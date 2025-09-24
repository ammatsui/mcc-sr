""" evolutionary step for calling mutation example:

def evolutionary_step(equation, llm, use_llm=False):
    # Use LLM if triggered, otherwise classical mutate
    if use_llm:
        return llm_mutate(equation, llm)
    else:
        return equation.mutate()  # classical random mutation
"""