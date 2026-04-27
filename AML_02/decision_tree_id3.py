# src/decision_tree_id3.py (OPTIMIZED - with max_features limit)
"""
ID3 (Iterative Dichotomiser 3) Decision Tree
Optimized for large datasets with many categorical features
"""
import numpy as np
from collections import Counter
from .decision_tree_base import BaseDecisionTree, DecisionTreeNode


class ID3DecisionTree(BaseDecisionTree):
    """
    ID3 Decision Tree implementation - Optimized version
    
    Characteristics:
    - Uses Information Gain for splitting
    - Handles categorical features only
    - Limits features considered at each split for performance
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.feature_possible_values = None
        self.max_features = max_features  # Limit features for performance
        
    def _impurity(self, y):
        """ID3 uses entropy for impurity"""
        return self._entropy(y)
    
    def _select_features(self, n_features):
        """Select subset of features to consider"""
        if self.max_features is None:
            return range(n_features)
        
        n_selected = min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            n_selected = int(self.max_features * n_features)
        
        return np.random.choice(n_features, n_selected, replace=False)
    
    def _find_best_split_categorical(self, X, y, feature_idx):
        """Find best split for categorical feature (optimized)"""
        unique_values = np.unique(X[:, feature_idx])
        
        # Skip if too many categories (beyond a threshold)
        if len(unique_values) > 20:
            return None
            
        if len(unique_values) < 2:
            return None
            
        best_gain = -1
        best_split_value = None
        best_left_indices = None
        best_right_indices = None
        
        # For each possible value, create binary split (value vs not value)
        for value in unique_values:
            left_mask = X[:, feature_idx] == value
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
                
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            gain = self._information_gain(y, y_left, y_right)
            
            if gain > best_gain:
                best_gain = gain
                best_split_value = value
                best_left_indices = left_mask
                best_right_indices = right_mask
        
        if best_gain <= 0:
            return None
            
        return {
            'feature_idx': feature_idx,
            'threshold': best_split_value,
            'gain': best_gain,
            'left_indices': best_left_indices,
            'right_indices': best_right_indices,
            'is_categorical': True
        }
    
    def _best_split(self, X, y):
        """Find the best split across all features"""
        n_samples, n_features = X.shape
        best_split = None
        
        # Select features to consider
        features_to_consider = self._select_features(n_features)
        
        for feature_idx in features_to_consider:
            split = self._find_best_split_categorical(X, y, feature_idx)
            if split is not None:
                if best_split is None or split['gain'] > best_split['gain']:
                    best_split = split
                    
                    # Early stopping if perfect gain
                    if best_split['gain'] >= 1.0:
                        break
        
        return best_split
    
    def _build_tree(self, X, y, depth=0, indices=None):
        """Override to handle categorical features differently"""
        if indices is None:
            indices = np.arange(len(y))
        
        X_subset = X[indices]
        y_subset = y[indices]
        
        n_samples = len(y_subset)
        n_classes = len(np.unique(y_subset))
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            return DecisionTreeNode(
                value=self._majority_class(y_subset),
                class_distribution=Counter(y_subset),
                is_leaf=True
            )
        
        # Find best split
        best_split = self._best_split(X_subset, y_subset)
        
        if best_split is None or best_split['gain'] <= 0:
            return DecisionTreeNode(
                value=self._majority_class(y_subset),
                class_distribution=Counter(y_subset),
                is_leaf=True
            )
        
        # Create internal node
        node = DecisionTreeNode(
            feature_idx=best_split['feature_idx'],
            threshold=best_split['threshold'],
            class_distribution=Counter(y_subset)
        )
        
        # Recursively build children
        left_indices = indices[best_split['left_indices']]
        right_indices = indices[best_split['right_indices']]
        
        node.left = self._build_tree(X, y, depth + 1, left_indices)
        node.right = self._build_tree(X, y, depth + 1, right_indices)
        
        return node