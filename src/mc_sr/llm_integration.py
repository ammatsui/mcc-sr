import openai  # or anthropic, or other LLM API

class LLMMutator:
    def __init__(self, api_key, model="gpt-4"):
        openai.api_key = api_key
        self.model = model

    def propose_mutation(self, equation_tree_str, context=None):
        """Send the equation (as string/tree) to the LLM, get mutation proposal."""
        prompt = self.build_prompt(equation_tree_str, context)
        response = self.query_llm(prompt)
        mutation = self.parse_mutation_response(response)
        return mutation
    
    def build_prompt(self, eq_str, context=None):
        # Make this descriptive for your task
        prompt = (
            "Here is an equation tree:\n" + eq_str +
            "\nSuggest a single symbolic mutation as a JSON dictionary (action, target_node, new_value):"
        )
        if context:
            prompt += f"\nContext: here is the information from the past evolution history: {context}"
        return prompt
    
    def query_llm(self, prompt):
        # Call out to the LLM (OpenAI example, modify for your backend)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.8
        )
        return response["choices"][0]["message"]["content"]

    def parse_mutation_response(self, llm_output):
        # Convert the response from string/JSON to Python dict
        import json
        try:
            return json.loads(llm_output)
        except Exception as e:
            raise RuntimeError(f"Bad LLM output: {llm_output}") from e
        
    def apply_eq_mutation(self, equation, mutation):    
        """Apply the mutation to the equation tree, return new equation."""
        # This depends on your Equation class structure
        # new_eq = equation.copy()  # Assume a copy method
        action = mutation.get("action")
        target_node = mutation.get("target_node")
        new_value = mutation.get("new_value")
        
        new_eq = equation.mutate(action = action, target_node=target_node, new_value=new_value)
        print(f"\nApplied LLM mutation: {mutation} to equation {equation} -> {new_eq}\n")
        return new_eq