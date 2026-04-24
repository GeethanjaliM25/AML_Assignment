"""
candidate_elimination.py - Implementation of Candidate Elimination Algorithm
"""

class CandidateElimination:
    """
    Candidate Elimination Algorithm
    Maintains version space using S (specific) and G (general) boundaries
    """
    
    def __init__(self, attributes, attribute_values):
        """
        Initialize Candidate Elimination algorithm
        
        Parameters:
        -----------
        attributes : list
            List of attribute names
        attribute_values : dict
            Dictionary mapping attribute names to possible values
        """
        self.attributes = attributes
        self.attr_values = attribute_values
        self.num_attrs = len(attributes)
        
        # Initialize S: Most specific hypothesis (all '∅')
        self.S = [tuple(['∅'] * self.num_attrs)]
        
        # Initialize G: Most general hypothesis (all '?')
        self.G = [tuple(['?'] * self.num_attrs)]
        
        self.trace = []  # Store evolution of S and G
    
    def _is_consistent(self, hypothesis, example, label):
        """
        Check if hypothesis is consistent with example
        """
        # Check if hypothesis matches example
        matches = True
        for i, val in enumerate(hypothesis):
            if val != '?' and val != '∅' and val != example[i]:
                matches = False
                break
        
        # For positive: must match, for negative: must not match
        return matches == (label == 1)
    
    def _is_more_general(self, h1, h2):
        """
        Check if hypothesis h1 is more general than or equal to h2
        """
        for i in range(len(h1)):
            # If h1 has ∅, it's not more general than anything (except ∅)
            if h1[i] == '∅':
                if h2[i] != '∅':
                    return False
                else:
                    continue
            # If h1 has specific value and h2 has ∅, h1 is more general
            if h2[i] == '∅':
                continue
            # If h1 is ? and h2 is anything, h1 is more general
            if h1[i] == '?':
                continue
            # Both are specific values
            if h1[i] != h2[i]:
                return False
        return True
    
    def _generalize_S(self, example):
        """
        Generalize S boundary for positive example
        """
        new_S = []
        
        for h in self.S:
            if self._is_consistent(h, example, 1):
                new_S.append(h)
            else:
                # Generalize hypothesis
                new_h = list(h)
                for i in range(len(h)):
                    if h[i] == '∅':
                        new_h[i] = example[i]
                    elif h[i] != example[i]:
                        new_h[i] = '?'
                new_S.append(tuple(new_h))
        
        # Remove duplicates
        self.S = list(set(new_S))
        
        # Remove hypotheses that are more specific than others
        to_remove = set()
        for i in range(len(self.S)):
            for j in range(len(self.S)):
                if i != j and self._is_more_general(self.S[j], self.S[i]):
                    to_remove.add(i)
        self.S = [h for idx, h in enumerate(self.S) if idx not in to_remove]
    
    def _specialize_G(self, example):
        """
        Specialize G boundary for negative example
        """
        new_G = []
        
        for h in self.G:
            if not self._is_consistent(h, example, 0):
                new_G.append(h)
            else:
                # Create specializations
                for i in range(len(h)):
                    if h[i] == '?':
                        for val in self.attr_values[self.attributes[i]]:
                            if val != example[i]:
                                new_h = list(h)
                                new_h[i] = val
                                new_G.append(tuple(new_h))
        
        # Remove duplicates
        new_G = list(set(new_G))
        
        # Remove hypotheses that are more general than others
        to_remove = set()
        for i in range(len(new_G)):
            for j in range(len(new_G)):
                if i != j and self._is_more_general(new_G[j], new_G[i]):
                    to_remove.add(i)
        self.G = [h for idx, h in enumerate(new_G) if idx not in to_remove]
        
        # Keep only hypotheses more general than some S
        valid_G = []
        for g in self.G:
            if any(self._is_more_general(g, s) for s in self.S):
                valid_G.append(g)
        self.G = valid_G if valid_G else [tuple(['?'] * self.num_attrs)]
    
    def fit(self, X, y, verbose=True):
        """
        Train Candidate Elimination algorithm on training data
        """
        if verbose:
            print("\n" + "="*70)
            print("CANDIDATE ELIMINATION ALGORITHM TRACE")
            print("="*70)
            print("\nInitial S (Specific):", self.S)
            print("Initial G (General):", self.G)
        
        for idx, (example, label) in enumerate(zip(X, y)):
            if verbose:
                print(f"\n--- Example {idx+1}: {example} | Label: {'Positive' if label==1 else 'Negative'} ---")
            
            if label == 1:  # Positive example
                self._generalize_S(example)
                # Remove from G any inconsistent hypotheses
                self.G = [h for h in self.G if self._is_consistent(h, example, 1)]
            else:  # Negative example
                self._specialize_G(example)
                # Remove from S any inconsistent hypotheses
                self.S = [h for h in self.S if self._is_consistent(h, example, 0)]
            
            # If S becomes empty or G becomes empty, version space is empty
            if not self.S or not self.G:
                if verbose:
                    print(f"  ⚠️ Version space became empty!")
                self.S = []
                self.G = []
                break
            
            if verbose:
                print(f"  S: {self.S}")
                print(f"  G: {self.G}")
            
            # Store trace
            self.trace.append({
                'example': idx + 1,
                'S': self.S.copy(),
                'G': self.G.copy()
            })
        
        if verbose and self.S and self.G:
            print("\n" + "="*70)
            print("FINAL BOUNDARIES:")
            print(f"  S (Specific): {self.S}")
            print(f"  G (General): {self.G}")
            print(f"  Version Space Size: {len(self.get_version_space())}")
            print("="*70)
        elif verbose:
            print("\n" + "="*70)
            print("⚠️ VERSION SPACE IS EMPTY - No hypothesis consistent with all examples")
            print("="*70)
        
        return self
    
    def get_version_space(self):
        """
        Generate all hypotheses in version space
        """
        if not self.S or not self.G:
            return []
        
        version_space = []
        for s in self.S:
            for g in self.G:
                if self._is_more_general(g, s):
                    version_space.append({
                        'specific': s,
                        'general': g,
                        'range': f"{s} ≤ h ≤ {g}"
                    })
        return version_space
    
    def get_boundaries(self):
        """Return S and G boundaries"""
        return {'S': self.S, 'G': self.G}
    
    def predict(self, X, method='conservative'):
        """
        Predict labels for new examples
        """
        if not self.S or not self.G:
            return [0] * len(X)
        
        predictions = []
        version_space = self.get_version_space()
        
        if not version_space:
            return [0] * len(X)
        
        for example in X:
            consistent_count = 0
            
            for vs in version_space:
                s = vs['specific']
                g = vs['general']
                
                # Check if example lies between s and g
                passes_s = True
                passes_g = True
                
                for i in range(len(example)):
                    if s[i] != '∅' and s[i] != '?' and s[i] != example[i]:
                        passes_s = False
                        break
                
                for i in range(len(example)):
                    if g[i] != '?' and g[i] != example[i]:
                        passes_g = False
                        break
                
                if passes_s and passes_g:
                    consistent_count += 1
            
            if method == 'majority':
                pred = 1 if consistent_count > len(version_space) / 2 else 0
            else:  # conservative - predict positive only if ALL agree
                pred = 1 if consistent_count == len(version_space) else 0
            
            predictions.append(pred)
        
        return predictions
    
    def __repr__(self):
        return f"CandidateElimination(S={self.S}, G={self.G})"