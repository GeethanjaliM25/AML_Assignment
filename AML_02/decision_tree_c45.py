# src/decision_tree_c45.py (FIXED - add Counter import)
"""
C4.5 Decision Tree
Uses Gain Ratio as splitting criterion
Handles both categorical and numerical features
Includes pruning capability
"""
import numpy as np
from collections import Counter
from .decision_tree_base import BaseDecisionTree, DecisionTreeNode


class C45DecisionTree(BaseDecisionTree):
    """
    C4.5 Decision Tree implementation
    
    Characteristics:
    - Uses Gain Ratio for splitting
    - Handles both categorical and numerical features
    - Handles missing values
    - Post-pruning capability
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 prune=False, confidence=0.25):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.prune = prune
        self.confidence = confidence
        self.feature_possible_values = None
        
    def _impurity(self, y):
        """C4.5 uses entropy"""
        return self._entropy(y)
    
    def _find_best_split(self, X, y, feature_idx):
        """Find best split for a specific feature"""
        unique_values = np.unique(X[:, feature_idx])
        
        if len(unique_values) < 2:
            return None
        
        best_gain_ratio = -1
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        
        # For numerical features, try all possible split points
        if len(unique_values) > 10:  # Treat as numerical
            sorted_values = np.sort(unique_values)
            # Try midpoints between consecutive values
            for i in range(len(sorted_values) - 1):
                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                    
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                gain_ratio = self._gain_ratio(y, y_left, y_right)
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask
        else:  # Categorical feature
            for value in unique_values:
                left_mask = X[:, feature_idx] == value
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                    
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                gain_ratio = self._gain_ratio(y, y_left, y_right)
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_threshold = value
                    best_left_mask = left_mask
                    best_right_mask = right_mask
        
        if best_gain_ratio <= 0:
            return None
            
        return {
            'feature_idx': feature_idx,
            'threshold': best_threshold,
            'gain_ratio': best_gain_ratio,
            'left_indices': best_left_mask,
            'right_indices': best_right_mask,
            'is_categorical': len(unique_values) <= 10
        }
    
    def _best_split(self, X, y):
        """Find the best split across all features"""
        n_samples, n_features = X.shape
        best_split = None
        
        for feature_idx in range(n_features):
            split = self._find_best_split(X, y, feature_idx)
            if split is not None:
                if best_split is None or split['gain_ratio'] > best_split['gain_ratio']:
                    best_split = split
        
        return best_split
    
    def _calculate_error_rate(self, node, X, y, indices):
        """Calculate error rate for pruning"""
        if node.is_leaf:
            predictions = np.full(len(indices), node.value)
            return np.mean(predictions != y[indices])
        
        left_indices = indices[node.left_indices_mask]
        right_indices = indices[node.right_indices_mask]
        
        left_error = self._calculate_error_rate(node.left, X, y, left_indices)
        right_error = self._calculate_error_rate(node.right, X, y, right_indices)
        
        weighted_error = (len(left_indices) * left_error + len(right_indices) * right_error) / len(indices)
        return weighted_error
    
    def _prune_node(self, node, X, y, indices):
        """Prune a node if beneficial"""
        if node.is_leaf:
            return node
        
        # Recursively prune children
        left_indices = indices[node.left_indices_mask]
        right_indices = indices[node.right_indices_mask]
        
        node.left = self._prune_node(node.left, X, y, left_indices)
        node.right = self._prune_node(node.right, X, y, right_indices)
        
        # Calculate error with subtree
        subtree_error = self._calculate_error_rate(node, X, y, indices)
        
        # Calculate error if replaced by leaf
        majority_class = self._majority_class(y[indices])
        leaf_predictions = np.full(len(indices), majority_class)
        leaf_error = np.mean(leaf_predictions != y[indices])
        
        # Prune if leaf has lower or equal error (with confidence interval)
        if leaf_error <= subtree_error + self.confidence * np.sqrt(subtree_error * (1 - subtree_error) / len(indices)):
            return DecisionTreeNode(
                value=majority_class,
                class_distribution=Counter(y[indices]),
                is_leaf=True
            )
        
        return node
    
    def _build_tree(self, X, y, depth=0, indices=None):
        """Build tree with pruning support"""
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
        
        if best_split is None or best_split['gain_ratio'] <= 0:
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
        
        # Store masks for pruning
        node.left_indices_mask = best_split['left_indices']
        node.right_indices_mask = best_split['right_indices']
        
        # Recursively build children
        left_indices = indices[best_split['left_indices']]
        right_indices = indices[best_split['right_indices']]
        
        node.left = self._build_tree(X, y, depth + 1, left_indices)
        node.right = self._build_tree(X, y, depth + 1, right_indices)
        
        return node
    
    def fit(self, X, y, feature_names=None):
        """Fit with optional pruning"""
        super().fit(X, y, feature_names)
        
        if self.prune:
            # Perform pruning on the full dataset
            all_indices = np.arange(len(y))
            self.root = self._prune_node(self.root, X, y, all_indices)
        
        return self