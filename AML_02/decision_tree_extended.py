# src/decision_tree_extended.py (NEW FILE - Extended variants)
"""
Extended Decision Tree Variants
- Regression Tree (for continuous targets)
- Adaptive Tree (adjusts splitting criterion)
"""
import numpy as np
from collections import Counter
from .decision_tree_base import BaseDecisionTree, DecisionTreeNode


class RegressionDecisionTree(BaseDecisionTree):
    """
    Regression Decision Tree for continuous target variables
    
    Characteristics:
    - Uses variance reduction for splitting
    - Predicts mean value at leaf nodes
    - For demonstration purposes
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        
    def _variance(self, y):
        """Calculate variance of target"""
        if len(y) == 0:
            return 0
        return np.var(y)
    
    def _impurity(self, y):
        """For regression, use variance"""
        return self._variance(y)
    
    def _variance_reduction(self, y, y_left, y_right):
        """Calculate variance reduction"""
        parent_var = self._variance(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        child_var = (n_left / n) * self._variance(y_left) + (n_right / n) * self._variance(y_right)
        return parent_var - child_var
    
    def _information_gain(self, y, y_left, y_right):
        """Override to use variance reduction"""
        return self._variance_reduction(y, y_left, y_right)
    
    def _best_split(self, X, y):
        """Find best split for regression"""
        n_samples, n_features = X.shape
        best_split = None
        best_reduction = -1
        
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            
            if len(unique_values) < 2:
                continue
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                reduction = self._variance_reduction(y, y_left, y_right)
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gain': reduction,
                        'left_indices': left_mask,
                        'right_indices': right_mask
                    }
        
        return best_split
    
    def _build_tree(self, X, y, depth=0, indices=None):
        """Build regression tree"""
        if indices is None:
            indices = np.arange(len(y))
        
        X_subset = X[indices]
        y_subset = y[indices]
        
        n_samples = len(y_subset)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            return DecisionTreeNode(
                value=np.mean(y_subset),
                is_leaf=True
            )
        
        best_split = self._best_split(X_subset, y_subset)
        
        if best_split is None or best_split['gain'] <= 0:
            return DecisionTreeNode(
                value=np.mean(y_subset),
                is_leaf=True
            )
        
        node = DecisionTreeNode(
            feature_idx=best_split['feature_idx'],
            threshold=best_split['threshold']
        )
        
        left_indices = indices[best_split['left_indices']]
        right_indices = indices[best_split['right_indices']]
        
        node.left = self._build_tree(X, y, depth + 1, left_indices)
        node.right = self._build_tree(X, y, depth + 1, right_indices)
        
        return node
    
    def predict(self, X):
        """Predict using mean values at leaves"""
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)