import json

class NeuroSymbolicSolver:
    def __init__(self, ontology_path="data/ontology.json"):
        # Allow loading from dict (for testing) or file
        if isinstance(ontology_path, dict):
            self.ontology = ontology_path
        else:
            with open(ontology_path, 'r') as f:
                self.ontology = json.load(f)

    def execute_plan(self, plan):
        """
        Recursively executes a Logical S-Expression.
        Example Plan: ('DIFFERENCE', 'Southern Summer', 'Polar Regions')
        """
        # Base Case: Concept Lookup
        if isinstance(plan, str):
            return self._lookup(plan)

        # Recursive Case: Unpack Operation
        op, arg1, arg2 = plan
        val1 = self.execute_plan(arg1)
        val2 = self.execute_plan(arg2)

        if op == 'INTERSECT':
            return self._intersect(val1, val2)
        elif op == 'UNION':
            return self._union(val1, val2)
        elif op == 'DIFFERENCE':
            return self._difference(val1, val2)
        
        raise ValueError(f"Unknown Op: {op}")

    def _lookup(self, concept):
        if concept not in self.ontology:
            raise ValueError(f"Concept '{concept}' not found in Ontology.")
        return self.ontology[concept]

    def _intersect(self, c1, c2):
        # 1. Check structural compatibility
        if c1.get('field') != c2.get('field'):
             raise ValueError(f"Field Mismatch: {c1.get('field')} vs {c2.get('field')}")

        # 2. Dispatch based on Abstract Data Type
        data_type = c1.get('type')
        
        if data_type == 'scalar_range':
            return self._intersect_scalar(c1, c2)
        elif data_type == 'categorical':
            return self._intersect_set(c1, c2)
        elif data_type == 'cyclic_range' or c2.get('type') == 'cyclic_range':
            return self._intersect_cyclic(c1, c2)
        
        raise ValueError(f"Unknown logic type: {data_type}")

    # --- GENERIC MATH KERNELS ---

    def _intersect_scalar(self, c1, c2):
        return {
            "type": "scalar_range",
            "field": c1['field'],
            "min": max(c1['min'], c2['min']),
            "max": min(c1['max'], c2['max']),
            "unit": c1.get('unit')
        }

    def _intersect_set(self, c1, c2):
        set1, set2 = set(c1['values']), set(c2['values'])
        return {
            "type": "categorical",
            "field": c1['field'],
            "values": list(set1.intersection(set2))
        }

    def _intersect_cyclic(self, c1, c2):
        """
        Handles logic where min > max (wrapping around a cycle).
        Logic: Unroll the cyclic range into linear ranges.
        """
        def unroll(c):
            if c.get('type') != 'cyclic_range':
                return [c]
            limit = c.get('range_max', 24.0)
            return [
                {"min": c['min'], "max": limit, "field": c['field']},
                {"min": 0, "max": c['max'], "field": c['field']}
            ]

        ranges1 = unroll(c1)
        ranges2 = unroll(c2)

        results = []
        for r1 in ranges1:
            for r2 in ranges2:
                # Use scalar logic on unrolled linear segments
                res_min = max(r1['min'], r2['min'])
                res_max = min(r1['max'], r2['max'])
                if res_min <= res_max:
                    results.append({"min": res_min, "max": res_max})
        
        if not results: return None
        return {"type": "multi_range", "field": c1['field'], "ranges": results}
    
    def _union(self, c1, c2):
        return [c1, c2] # Simplified for MVP
        
    def _difference(self, base, subtract):
        if base.get('field') != subtract.get('field'): return base
        return {"type": "difference_result", "base": base, "removed": subtract}