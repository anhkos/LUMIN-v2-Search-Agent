from src.logic_engine import NeuroSymbolicSolver
import json

def run():
    solver = NeuroSymbolicSolver()
    
    print("--- LUMIN v2 Logic Core Test ---\n")

    # Test 1: Standard Scalar (Southern Summer)
    print("Test 1: Reasoning about 'Southern Summer'")
    res1 = solver.execute_plan("Southern Summer")
    print(json.dumps(res1, indent=2))

    # Test 2: The "Cyclic" Edge Case (Midnight)
    print("\nTest 2: Implicit Cyclic Logic (Midnight)")
    res2 = solver.execute_plan("Midnight")
    print(json.dumps(res2, indent=2))

    # Test 3: Complex Intersection
    # Let's say we define "Observation Window" as 00:30 to 02:00
    # Does it overlap with Midnight (23:00 - 01:00)?
    # Result should be 00:30 - 01:00.
    print("\nTest 3: Intersecting 'Midnight' with '00:30-02:00'")
    
    obs_window = {
        "type": "scalar_range", 
        "field": "local_true_solar_time", 
        "min": 0.5, "max": 2.0 
    }
    # Manually injecting for test
    solver.ontology["Obs_Window"] = obs_window
    
    plan = ('INTERSECT', 'Midnight', 'Obs_Window')
    res3 = solver.execute_plan(plan)
    print("Result (Should be 0.5 to 1.0):")
    print(json.dumps(res3, indent=2))

if __name__ == "__main__":
    run()