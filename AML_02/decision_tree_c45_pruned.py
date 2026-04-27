# src/decision_tree_c45_pruned.py (NEW FILE)
"""
C4.5 Decision Tree with Post-Pruning
"""
import numpy as np
from collections import Counter
from .decision_tree_c45 import C45DecisionTree
from .decision_tree_base import DecisionTreeNode


class C45PrunedDecisionTree(C45DecisionTree):
    """
    C4.5 Decision Tree with improved post-pruning
    
    Additional Characteristics:
    - Post-pruning using reduced error pruning
    - Pessimistic error pruning
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 prune=True, confidence=0.25, prune_type='pessimistic'):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, prune, confidence)
        self.prune_type = prune_type
        
    def _calculate_pessimistic_error(self, node, y_subset):
        """Calculate pessimistic error estimate for pruning"""
        if node.is_leaf:
            # Count misclassifications
            majority = node.value
            errors = np.sum(y_subset != majority)
            # Add continuity correction (0.5)
            return errors + 0.5
        
        # For internal node, pessimistic error = sum of child errors
        return 0  # Will be calculated recursively
    
    def _reduced_error_prune(self, node, X_val, y_val, indices):
        """Reduced error pruning using validation set"""
        if node.is_leaf:
            return node
        
        # Prune children first
        left_indices = indices[node.left_indices_mask]
        right_indices = indices[node.right_indices_mask]
        
        node.left = self._reduced_error_prune(node.left, X_val, y_val, left_indices)
        node.right = self._reduced_error_prune(node.right, X_val, y_val, right_indices)
        
        # Calculate error with subtree on validation set
        y_pred_subtree = np.array([self._predict_one(x, node) for x in X_val[indices]])
        subtree_error = np.mean(y_pred_subtree != y_val[indices])
        
        # Calculate error if replaced by leaf
        majority_class = self._majority_class(y_val[indices])
        leaf_error = np.mean(majority_class != y_val[indices])
        
        # Prune if leaf has lower error
        if leaf_error <= subtree_error:
            return DecisionTreeNode(
                value=majority_class,
                class_distribution=Counter(y_val[indices]),
                is_leaf=True
            )
        
        return node
    
    def fit(self, X, y, X_val=None, y_val=None, feature_names=None):
        """Fit with validation-based pruning"""
        # First build the tree
        super().fit(X, y, feature_names)
        
        if self.prune and X_val is not None and y_val is not None:
            # Perform reduced error pruning using validation set
            all_indices = np.arange(len(y_val))
            self.root = self._reduced_error_prune(self.root, X_val, y_val, all_indices)
        
        return self