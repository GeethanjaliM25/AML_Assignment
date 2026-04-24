"""
find_s.py - Implementation of Find-S Algorithm
"""

class FindS:
    """
    Find-S Algorithm for Concept Learning
    Finds the most specific hypothesis consistent with positive examples
    """
    
    def __init__(self, attributes):
        """
        Initialize Find-S algorithm
        
        Parameters:
        -----------
        attributes : list
            List of attribute names
        """
        self.attributes = attributes
        self.hypothesis = None
        self.initialized = False
        self.trace = []  # Store trace for visualization
    
    def _initialize_hypothesis(self, example):
        """
        Initialize hypothesis with first positive example
        
        Parameters:
        -----------
        example : tuple
            First positive example
        """
        self.hypothesis = list(example)
        self.initialized = True
        self.trace.append({
            'step': 'Initialization',
            'hypothesis': tuple(self.hypothesis),
            'description': f'Initialized with first positive example: {example}'
        })
    
    def _generalize(self, hypothesis, example):
        """
        Generalize hypothesis to include new positive example
        
        Parameters:
        -----------
        hypothesis : list
            Current hypothesis
        example : tuple
            New positive example
            
        Returns:
        --------
        list : Generalized hypothesis
        """
        new_hypothesis = list(hypothesis)
        
        for i in range(len(hypothesis)):
            # If attribute values differ, generalize to '?'
            if hypothesis[i] != example[i] and hypothesis[i] != '?':
                new_hypothesis[i] = '?'
        
        return new_hypothesis
    
    def fit(self, X, y, verbose=True):
        """
        Train Find-S algorithm on training data
        
        Parameters:
        -----------
        X : list
            List of training examples
        y : list
            List of labels (1 = positive, 0 = negative)
        verbose : bool
            Whether to print trace information
            
        Returns:
        --------
        self : FindS
            Trained model
        """
        if verbose:
            print("\n" + "="*70)
            print("FIND-S ALGORITHM TRACE")
            print("="*70)
            print("\nInitial hypothesis: Most specific (all '∅')")
        
        for idx, (example, label) in enumerate(zip(X, y)):
            if verbose:
                print(f"\n--- Example {idx+1}: {example} | Label: {'Positive' if label==1 else 'Negative'} ---")
            
            if label == 1:  # Positive example
                if not self.initialized:
                    self._initialize_hypothesis(example)
                else:
                    old_hyp = tuple(self.hypothesis)
                    self.hypothesis = self._generalize(self.hypothesis, example)
                    self.trace.append({
                        'step': f'Example {idx+1}',
                        'hypothesis': tuple(self.hypothesis),
                        'description': f'Generalized from {old_hyp} to {tuple(self.hypothesis)}'
                    })
                
                if verbose:
                    print(f"  Current hypothesis: {tuple(self.hypothesis)}")
            
            else:  # Negative example
                if verbose:
                    print(f"  Negative example ignored (Find-S only uses positives)")
                    print(f"  Current hypothesis unchanged: {tuple(self.hypothesis)}")
        
        if verbose:
            print("\n" + "="*70)
            print(f"FINAL HYPOTHESIS: {tuple(self.hypothesis)}")
            print("="*70)
        
        return self
    
    def predict(self, X):
        """
        Predict labels for new examples
        
        Parameters:
        -----------
        X : list
            List of examples to predict
            
        Returns:
        --------
        list : Predicted labels (1 if consistent with hypothesis, else 0)
        """
        if not self.initialized:
            return [0] * len(X)
        
        predictions = []
        for example in X:
            consistent = True
            for i, val in enumerate(self.hypothesis):
                if val != '?' and val != example[i]:
                    consistent = False
                    break
            predictions.append(1 if consistent else 0)
        
        return predictions
    
    def get_hypothesis(self):
        """Return the learned hypothesis"""
        return tuple(self.hypothesis) if self.hypothesis else None
    
    def get_trace(self):
        """Return training trace"""
        return self.trace
    
    def __repr__(self):
        return f"FindS(hypothesis={self.get_hypothesis()})"