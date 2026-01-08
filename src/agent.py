import os
import json
from openai import OpenAI # type: ignore
from dotenv import load_dotenv
from logic_engine import NeuroSymbolicSolver

load_dotenv()  # Load environment variables from .env file

# Initialize the env
client = OpenAI() # Ensure OPENAI_API_KEY is in your env
solver = NeuroSymbolicSolver()

# The "System 2" Prompt
# We force the LLM to be a "Compiler", not a Chatbot.
SYSTEM_PROMPT = """
You are the LUMIN Query Compiler for NASA PDS.
Your goal is to translate Natural Language into a Logic S-Expression.

### ONTOLOGY (Valid Concepts):
{ontology_keys}

### OPERATIONS:
1. INTERSECT(A, B) -> Returns overlap.
2. UNION(A, B) -> Returns combination.
3. DIFFERENCE(Base, Subtract) -> Removes constraints.

### RULES:
- Output ONLY the tuple plan. No markdown, no explanation.
- Use Concept Names EXACTLY as listed in the Ontology.
- If the user asks for a specific time/value not in ontology, ignore it for this MVP (or map to closest concept).

### EXAMPLES:
User: "Southern summer images"
Output: "Southern Summer"

User: "Southern summer but not polar regions"
Output: ('DIFFERENCE', 'Southern Summer', 'Polar Regions')

User: "Midnight observations during the dust storm season"
Output: ('INTERSECT', 'Midnight', 'Dust Storm Season')
"""

def run_agent(user_query):
    # 1. Dynamic Prompting: Inject available concepts from your JSON
    # This ensures the LLM knows the data source exist. 
    ontology_keys = ", ".join(solver.ontology.keys())
    prompt = SYSTEM_PROMPT.format(ontology_keys=ontology_keys)

    print(f"--- Processing: '{user_query}' ---")

    try:
        # 2. The "Reasoning" Step (LLM Generation)
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0, # Deterministic logic
        )
        
        raw_plan = response.choices[0].message.content.strip()
        print(f"🔹 LLM Plan:   {raw_plan}")

        # 3. The "Grounding" Step (Logic Engine Execution)
        # We use eval() safely here because we trust the verified prompt output,
        # but we'll probably write a parser later on down the road.
        parsed_plan = eval(raw_plan) 
        
        result = solver.execute_plan(parsed_plan)
        
        # 4. The Result
        print(f"Grounded:   {json.dumps(result, indent=2)}")
        return result

    except Exception as e:
        print(f"Error:      {e}")
        return None

if __name__ == "__main__":
    # The "Tier 3" Test Case
    run_agent("I need thermal data taken at midnight")