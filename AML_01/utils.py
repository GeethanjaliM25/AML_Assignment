"""
utils.py - Utility functions for visualization and analysis
"""

def display_hypothesis(hypothesis, attributes):
    """
    Display hypothesis in readable format
    """
    if not hypothesis:
        return "No hypothesis learned"
    
    constraints = []
    for attr, val in zip(attributes, hypothesis):
        if val == '?':
            constraints.append(f"{attr}=any")
        elif val == '∅':
            constraints.append(f"{attr}=none")
        else:
            constraints.append(f"{attr}={val}")
    
    return " ∧ ".join(constraints)


def display_version_space(ce_model, attributes):
    """
    Display version space in readable format
    """
    boundaries = ce_model.get_boundaries()
    version_space = ce_model.get_version_space()
    
    print("\nVersion Space Analysis:")
    print(f"  Specific Boundary (S): {len(boundaries['S'])} hypothesis(es)")
    for s in boundaries['S']:
        print(f"    → {display_hypothesis(s, attributes)}")
    
    print(f"\n  General Boundary (G): {len(boundaries['G'])} hypothesis(es)")
    for g in boundaries['G']:
        print(f"    → {display_hypothesis(g, attributes)}")
    
    print(f"\n  Version Space Size: {len(version_space)} hypothesis pairs")
    
    if len(version_space) <= 10:
        for vs in version_space:
            print(f"    {vs['range']}")
    elif version_space:
        print(f"    (showing first 10 of {len(version_space)})")
        for vs in version_space[:10]:
            print(f"    {vs['range']}")


def compare_algorithms(find_s_model, ce_model, X_test, y_test=None, attributes=None):
    """
    Compare Find-S and Candidate Elimination predictions
    """
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON")
    print("="*70)
    
    find_s_pred = find_s_model.predict(X_test)
    ce_pred = ce_model.predict(X_test)
    
    print(f"\n{'Example':<50} {'Find-S':<10} {'CE':<10} {'Actual':<10}")
    print("-"*80)
    
    for i, ex in enumerate(X_test):
        ex_str = str(ex)[:48]
        # FIX: Handle None values in actual labels
        actual = y_test[i] if y_test and y_test[i] is not None else "?"
        actual_str = str(actual) if actual != "?" else "?"
        print(f"{ex_str:<50} {find_s_pred[i]:<10} {ce_pred[i]:<10} {actual_str:<10}")
    
    if y_test and all(y is not None for y in y_test):
        find_s_acc = sum(1 for p, a in zip(find_s_pred, y_test) if p == a) / len(y_test)
        ce_acc = sum(1 for p, a in zip(ce_pred, y_test) if p == a) / len(y_test)
        
        print("\n" + "-"*80)
        print(f"Find-S Accuracy: {find_s_acc:.2%}")
        print(f"CE Accuracy: {ce_acc:.2%}")


def calculate_hypothesis_complexity(hypothesis):
    """
    Calculate complexity of a hypothesis
    """
    if not hypothesis:
        return {'specific_count': 0, 'general_count': 0, 'complexity': 0}
    
    specific_count = sum(1 for v in hypothesis if v not in ['?', '∅'])
    general_count = sum(1 for v in hypothesis if v == '?')
    none_count = sum(1 for v in hypothesis if v == '∅')
    
    complexity = specific_count / len(hypothesis) if len(hypothesis) > 0 else 0
    
    return {
        'specific_values': specific_count,
        'wildcards': general_count,
        'no_values': none_count,
        'complexity': complexity
    }