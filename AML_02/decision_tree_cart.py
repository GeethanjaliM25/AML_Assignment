# src/decision_tree_cart.py (FIXED - add Counter import)
"""
CART (Classification and Regression Trees) Decision Tree
Uses Gini Impurity for classification
Handles binary splits only
"""
import numpy as np
from collections import Counter
from .decision_tree_base import BaseDecisionTree, DecisionTreeNode


class CARTDecisionTree(BaseDecisionTree):
    """
    CART Decision Tree implementation
    
    Characteristics:
    - Uses Gini Impurity for classification
    - Binary splits only
    - Handles both categorical and numerical features
    - Supports pruning via cost-complexity
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, ccp_alpha=0.0):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha  # Cost-complexity pruning parameter
        
    def _impurity(self, y):
        """CART uses Gini impurity"""
        return self._gini(y)
    
    def _find_best_split_numerical(self, X, y, feature_idx):
        """Find best split for numerical feature"""
        sorted_indices = np.argsort(X[:, feature_idx])
        sorted_X = X[sorted_indices, feature_idx]
        sorted_y = y[sorted_indices]
        
        best_gain = -1
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        
        # Try splits between different values
        for i in range(len(sorted_X) - 1):
            if sorted_X[i] == sorted_X[i + 1]:
                continue
                
            threshold = (sorted_X[i] + sorted_X[i + 1]) / 2
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
                
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            gain = self._information_gain(y, y_left, y_right)
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                best_left_mask = left_mask
                best_right_mask = right_mask
        
        return best_gain, best_threshold, best_left_mask, best_right_mask
    
    def _find_best_split_categorical(self, X, y, feature_idx):
        """Find best binary split for categorical feature"""
        unique_values = np.unique(X[:, feature_idx])
        
        if len(unique_values) < 2:
            return None, None, None, None
        
        best_gain = -1
        best_left_values = None
        best_left_mask = None
        best_right_mask = None
        
        # For binary split, we need to find the best subset of categories
        # This is combinatorial, so we try all possible subsets (simplified)
        # For efficiency, we'll try each value as its own split
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
                best_left_values = [value]
                best_left_mask = left_mask
                best_right_mask = right_mask
        
        return best_gain, best_left_values, best_left_mask, best_right_mask
    
    def _select_features(self, n_features):
        """Select random subset of features for split (for Random Forest compatibility)"""
        if self.max_features is None:
            return range(n_features)
        
        n_selected = min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            n_selected = int(self.max_features * n_features)
        
        return np.random.choice(n_features, n_selected, replace=False)
    
    def _best_split(self, X, y):
        """Find the best split across all features"""
        n_samples, n_features = X.shape
        best_split = None
        
        # Select features to consider
        features_to_consider = self._select_features(n_features)
        
        for feature_idx in features_to_consider:
            # Check if feature is numerical or categorical
            unique_values = len(np.unique(X[:, feature_idx]))
            
            if unique_values > 10:  # Treat as numerical
                gain, threshold, left_mask, right_mask = self._find_best_split_numerical(X, y, feature_idx)
                if gain > 0:
                    if best_split is None or gain > best_split['gain']:
                        best_split = {
                            'feature_idx': feature_idx,
                            'threshold': threshold,
                            'gain': gain,
                            'left_indices': left_mask,
                            'right_indices': right_mask,
                            'is_categorical': False
                        }
            else:  # Categorical
                gain, left_values, left_mask, right_mask = self._find_best_split_categorical(X, y, feature_idx)
                if gain is not None and gain > 0:
                    if best_split is None or gain > best_split['gain']:
                        best_split = {
                            'feature_idx': feature_idx,
                            'threshold': left_values,  # Store the set of values for left branch
                            'gain': gain,
                            'left_indices': left_mask,
                            'right_indices': right_mask,
                            'is_categorical': True
                        }
        
        return best_split
    
    def _predict_one(self, x, node):
        """Predict with handling for categorical splits"""
        if node.is_leaf:
            return node.value
        
        if node.is_categorical:
            # For categorical, check if x's value is in the left set
            if x[node.feature_idx] in node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
        else:
            # For numerical, use <= comparison
            if x[node.feature_idx] <= node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
    
    def _build_tree(self, X, y, depth=0, indices=None):
        """Build tree with cost-complexity pruning support"""
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
            node = DecisionTreeNode(
                value=self._majority_class(y_subset),
                class_distribution=Counter(y_subset),
                is_leaf=True
            )
            node.n_samples = n_samples
            node.impurity = self._impurity(y_subset)
            return node
        
        # Find best split
        best_split = self._best_split(X_subset, y_subset)
        
        if best_split is None or best_split['gain'] <= self.ccp_alpha:
            node = DecisionTreeNode(
                value=self._majority_class(y_subset),
                class_distribution=Counter(y_subset),
                is_leaf=True
            )
            node.n_samples = n_samples
            node.impurity = self._impurity(y_subset)
            return node
        
        # Create internal node
        node = DecisionTreeNode(
            feature_idx=best_split['feature_idx'],
            threshold=best_split['threshold'],
            class_distribution=Counter(y_subset)
        )
        node.is_categorical = best_split.get('is_categorical', False)
        node.n_samples = n_samples
        node.impurity = self._impurity(y_subset)
        
        # Recursively build children
        left_indices = indices[best_split['left_indices']]
        right_indices = indices[best_split['right_indices']]
        
        node.left = self._build_tree(X, y, depth + 1, left_indices)
        node.right = self._build_tree(X, y, depth + 1, right_indices)
        
        return node