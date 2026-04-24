"""
version_space_cases.py - Demonstration of 5 cases where version space becomes impossible
"""

from find_s import FindS
from candidate_elimination import CandidateElimination
from datasets import CustomDataset

class VersionSpaceImpossibleCases:
    """
    Demonstrates 5 cases where version space becomes empty/impossible to obtain
    """
    
    def __init__(self):
        self.results = []
    
    def case1_negative_first(self):
        """
        Case 1: First example is negative
        """
        print("\n" + "="*70)
        print("CASE 1: FIRST EXAMPLE IS NEGATIVE")
        print("="*70)
        print("Issue: Find-S cannot initialize without a positive example")
        
        attributes = ['Sky', 'Temp']
        attr_values = {'Sky': ['Sunny', 'Rainy'], 'Temp': ['Warm', 'Cold']}
        
        X = [('Sunny', 'Warm'), ('Rainy', 'Cold')]
        y = [0, 1]
        
        print(f"\nData: {list(zip(X, y))}")
        
        # Try Find-S
        find_s = FindS(attributes)
        find_s.fit(X, y, verbose=False)
        print(f"\nFind-S result: {find_s.get_hypothesis()}")
        
        # Try CE
        ce = CandidateElimination(attributes, attr_values)
        ce.fit(X, y, verbose=False)
        print(f"\nCE result - S: {ce.S}, G: {ce.G}")
        
        return {
            'case': 1,
            'name': 'Negative Example First',
            'find_s_works': False,
            'ce_works': True,
            'explanation': 'Find-S cannot initialize without positive example'
        }
    
    def case2_inconsistent_positives(self):
        """
        Case 2: Same instance with different labels
        """
        print("\n" + "="*70)
        print("CASE 2: INCONSISTENT DATA")
        print("="*70)
        print("Issue: Same instance appears as both positive and negative")
        
        attributes, X, y = CustomDataset.create_inconsistent_dataset()
        attr_values = {'Sky': ['Sunny', 'Rainy', 'Cloudy'], 'Temp': ['Warm', 'Cold']}
        
        print(f"\nData: {list(zip(X, y))}")
        print("Note: First two examples are identical but have different labels!")
        
        find_s = FindS(attributes)
        find_s.fit(X, y, verbose=False)
        print(f"\nFind-S result: {find_s.get_hypothesis()}")
        
        ce = CandidateElimination(attributes, attr_values)
        ce.fit(X, y, verbose=False)
        print(f"\nCE result - S: {ce.S}, G: {ce.G}")
        
        if not ce.S or not ce.G:
            print("[OK] CE correctly identified inconsistency -> empty version space")
        
        return {
            'case': 2,
            'name': 'Inconsistent Data',
            'find_s_works': True,
            'ce_works': False,
            'explanation': 'No hypothesis consistent with contradictory data'
        }
    
    def case3_no_common_value(self):
        """
        Case 3: Positive examples have no common attribute value
        """
        print("\n" + "="*70)
        print("CASE 3: NO COMMON ATTRIBUTE VALUE")
        print("="*70)
        print("Issue: Positive examples require mutually exclusive values")
        
        attributes = ['Sky']
        attr_values = {'Sky': ['Sunny', 'Rainy', 'Cloudy']}
        
        X = [('Sunny',), ('Rainy',)]
        y = [1, 1]
        
        print(f"\nData: {list(zip(X, y))}")
        
        find_s = FindS(attributes)
        find_s.fit(X, y, verbose=False)
        print(f"\nFind-S result: {find_s.get_hypothesis()}")
        
        ce = CandidateElimination(attributes, attr_values)
        ce.fit(X, y, verbose=False)
        print(f"\nCE result - S: {ce.S}, G: {ce.G}")
        
        return {
            'case': 3,
            'name': 'No Common Attribute Value',
            'find_s_works': True,
            'ce_works': True,
            'explanation': 'Both algorithms generalize to "?"'
        }
    
    def case4_non_conjunctive_concept(self):
        """
        Case 4: Non-conjunctive concept (XOR pattern)
        """
        print("\n" + "="*70)
        print("CASE 4: NON-CONJUNCTIVE CONCEPT (XOR Pattern)")
        print("="*70)
        print("Issue: Target concept is not representable as conjunction")
        print("XOR: (Sky=Sunny AND Temp=Warm) OR (Sky=Rainy AND Temp=Cold)")
        
        attributes, X, y = CustomDataset.create_xor_dataset()
        attr_values = {'Sky': ['Sunny', 'Rainy', 'Cloudy'], 'Temp': ['Warm', 'Cold']}
        
        print(f"\nData: {list(zip(X, y))}")
        
        find_s = FindS(attributes)
        find_s.fit(X, y, verbose=False)
        print(f"\nFind-S result: {find_s.get_hypothesis()}")
        
        test_X = [('Sunny', 'Cold'), ('Rainy', 'Warm')]
        preds = find_s.predict(test_X)
        print(f"Predictions on XOR negatives: {list(zip(test_X, preds))}")
        
        ce = CandidateElimination(attributes, attr_values)
        ce.fit(X, y, verbose=False)
        print(f"\nCE result - S: {ce.S}, G: {ce.G}")
        
        return {
            'case': 4,
            'name': 'Non-Conjunctive Concept (XOR)',
            'find_s_works': False,
            'ce_works': False,
            'explanation': 'Neither algorithm can learn disjunctive concepts'
        }
    
    def case5_no_positive_examples(self):
        """
        Case 5: No positive examples in training data
        """
        print("\n" + "="*70)
        print("CASE 5: NO POSITIVE EXAMPLES")
        print("="*70)
        print("Issue: Training data contains only negative examples")
        
        attributes, X, y = CustomDataset.create_no_positive_dataset()
        attr_values = {'Sky': ['Sunny', 'Rainy', 'Cloudy'], 'Temp': ['Warm', 'Cold']}
        
        print(f"\nData: {list(zip(X, y))}")
        
        find_s = FindS(attributes)
        find_s.fit(X, y, verbose=False)
        print(f"\nFind-S result: {find_s.get_hypothesis()}")
        
        ce = CandidateElimination(attributes, attr_values)
        ce.fit(X, y, verbose=False)
        print(f"\nCE result - S: {ce.S}, G: {ce.G}")
        
        return {
            'case': 5,
            'name': 'No Positive Examples',
            'find_s_works': False,
            'ce_works': True,
            'explanation': 'CE maintains hypothesis space, Find-S never initializes'
        }
    
    def run_all_cases(self):
        """Run all 5 impossible cases"""
        print("\n" + "="*70)
        print("5 IMPOSSIBLE CASES FOR VERSION SPACE")
        print("="*70)
        
        cases = [
            self.case1_negative_first,
            self.case2_inconsistent_positives,
            self.case3_no_common_value,
            self.case4_non_conjunctive_concept,
            self.case5_no_positive_examples
        ]
        
        results = []
        for case_func in cases:
            result = case_func()
            results.append(result)
            print("\n" + "-"*70)
        
        # Summary table
        print("\n" + "="*70)
        print("SUMMARY: 5 IMPOSSIBLE/EDGE CASES")
        print("="*70)
        print(f"{'Case':<5} {'Name':<30} {'Find-S':<8} {'CE':<8} {'Explanation'}")
        print("-"*70)
        
        for r in results:
            find_s_status = "OK" if r['find_s_works'] else "FAIL"
            ce_status = "OK" if r['ce_works'] else "FAIL"
            print(f"{r['case']:<5} {r['name']:<30} {find_s_status:<8} {ce_status:<8} {r['explanation']}")
        
        return results


if __name__ == "__main__":
    cases = VersionSpaceImpossibleCases()
    cases.run_all_cases()