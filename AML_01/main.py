"""
main.py - Main driver for Find-S and Candidate Elimination demonstration
"""

from datasets import EnjoySportDataset, CustomDataset
from find_s import FindS
from candidate_elimination import CandidateElimination
from version_space_cases import VersionSpaceImpossibleCases
from utils import display_hypothesis, display_version_space, compare_algorithms


def main():
    """Main function to run the complete demonstration"""
    
    print("\n" + "="*70)
    print("FIND-S & CANDIDATE ELIMINATION ALGORITHMS")
    print("COMPLETE IMPLEMENTATION & ANALYSIS")
    print("="*70)
    
    # ============================================================
    # PART 1: Load Dataset
    # ============================================================
    print("\n" + "="*70)
    print("PART 1: DATASET LOADING")
    print("="*70)
    
    dataset = EnjoySportDataset()
    X, y = dataset.get_data()
    attributes = dataset.get_attributes()
    attr_values = dataset.get_attribute_values()
    
    print(f"\nDataset: {dataset.get_description()['name']}")
    print(f"Source: {dataset.get_description()['source']}")
    print(f"Instances: {dataset.get_description()['instances']}")
    print(f"Positive: {dataset.get_description()['positive_count']}")
    print(f"Negative: {dataset.get_description()['negative_count']}")
    
    print("\nTraining Data:")
    for i, (ex, label) in enumerate(zip(X, y)):
        print(f"  {i+1}. {ex} -> {'Enjoy' if label==1 else 'Not Enjoy'}")
    
    # ============================================================
    # PART 2: Find-S Algorithm
    # ============================================================
    find_s = FindS(attributes)
    find_s.fit(X, y, verbose=True)
    
    print("\n" + "-"*70)
    print("Find-S Hypothesis Interpretation:")
    print(f"  {display_hypothesis(find_s.get_hypothesis(), attributes)}")
    
    # ============================================================
    # PART 3: Candidate Elimination Algorithm
    # ============================================================
    ce = CandidateElimination(attributes, attr_values)
    ce.fit(X, y, verbose=True)
    
    print("\n" + "-"*70)
    print("Candidate Elimination Interpretation:")
    display_version_space(ce, attributes)
    
    # ============================================================
    # PART 4: Testing on New Examples
    # ============================================================
    print("\n" + "="*70)
    print("PART 4: TESTING ON NEW EXAMPLES")
    print("="*70)
    
    test_X = [
        ('Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'),
        ('Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'),
        ('Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'),
        ('Cloudy', 'Warm', 'Normal', 'Weak', 'Cool', 'Same'),
    ]
    test_y = [1, 1, 0, None]
    
    compare_algorithms(find_s, ce, test_X, test_y, attributes)
    
    # ============================================================
    # PART 5: Five Impossible Cases
    # ============================================================
    print("\n" + "="*70)
    print("PART 5: FIVE IMPOSSIBLE/EDGE CASES")
    print("="*70)
    print("\nRunning demonstrations of cases where version space fails...")
    
    cases = VersionSpaceImpossibleCases()
    case_results = cases.run_all_cases()
    
    # ============================================================
    # PART 6: Final Summary
    # ============================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY & CONCLUSIONS")
    print("="*70)
    
    print("\nAlgorithm Comparison:")
    print("  +---------------------+--------------+-------------------------+")
    print("  | Feature             | Find-S       | Candidate Elimination   |")
    print("  +---------------------+--------------+-------------------------+")
    print("  | Output              | 1 hypothesis | Complete version space  |")
    print("  | Handles Negatives   | No           | Yes                     |")
    print("  | Noise Tolerance     | None         | None (needs pruning)    |")
    print("  | Time Complexity     | O(PxA)       | O(NxAx|S||G|)           |")
    print("  | Space Complexity    | O(A)         | O(|S|xA + |G|xA)        |")
    print("  | Best for            | Quick approx | Exact concept learning  |")
    print("  +---------------------+--------------+-------------------------+")
    
    print("\nKey Findings:")
    print("  1. Find-S is efficient but incomplete - only finds specific boundary")
    print("  2. CE maintains complete version space but at higher computational cost")
    print("  3. Both fail on non-conjunctive concepts (XOR, disjunctions)")
    print("  4. Inconsistent data leads to empty version space in CE")
    print("  5. Find-S cannot initialize without positive examples")
    
    print("\n[SUCCESS] Implementation Complete!")
    print("   - All algorithms implemented from scratch")
    print("   - Step-by-step tracing available")
    print("   - 5 edge cases demonstrated")
    print("   - Ready for extension to real-world datasets")
    
    return find_s, ce, case_results


if __name__ == "__main__":
    find_s, ce, results = main()