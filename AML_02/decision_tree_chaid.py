# src/decision_tree_chaid.py (COMPLETE - Add to your existing file)
"""
CHAID (Chi-squared Automatic Interaction Detector) Decision Tree
Uses Chi-square test for splitting categorical features
"""
import numpy as np
from collections import Counter
from scipy.stats import chi2_contingency
from .decision_tree_base import BaseDecisionTree, DecisionTreeNode


class CHAIDDecisionTree(BaseDecisionTree):
    """
    CHAID Decision Tree implementation
    
    Characteristics:
    - Uses Chi-square test for splitting
    - Handles categorical features only
    - Can merge non-significant categories
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 alpha_split=0.05, alpha_merge=0.05, max_categories=10):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.alpha_split = alpha_split
        self.alpha_merge = alpha_merge
        self.max_categories = max_categories
        
    def _impurity(self, y):
        return 0
    
    def _chi_square_test(self, X_feature, y):
        """Perform chi-square test for independence"""
        unique_x = np.unique(X_feature)
        unique_y = np.unique(y)
        
        contingency = np.zeros((len(unique_x), len(unique_y)))
        
        for i, x_val in enumerate(unique_x):
            mask = X_feature == x_val
            for j, y_val in enumerate(unique_y):
                contingency[i, j] = np.sum(mask & (y == y_val))
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        return chi2, p_value
    
    def _best_split(self, X, y):
        """Find the best split using chi-square test"""
        n_samples, n_features = X.shape
        best_split = None
        best_chi2 = -1
        
        for feature_idx in range(n_features):
            X_feature = X[:, feature_idx]
            unique_values = np.unique(X_feature)
            
            if len(unique_values) < 2 or len(unique_values) > self.max_categories:
                continue
            
            # Test if the feature is useful for splitting
            _, p_value = self._chi_square_test(X_feature, y)
            
            if p_value < self.alpha_split:
                chi2, _ = self._chi_square_test(X_feature, y)
                split_values = unique_values
                
                if len(split_values) > 1 and chi2 > best_chi2:
                    best_chi2 = chi2
                    best_split = {
                        'feature_idx': feature_idx,
                        'split_values': split_values,
                        'p_value': p_value,
                        'chi2': chi2
                    }
        
        return best_split
    
    def _build_tree(self, X, y, depth=0, indices=None):
        """Build CHAID tree with multi-way splits"""
        if indices is None:
            indices = np.arange(len(y))
        
        X_subset = X[indices]
        y_subset = y[indices]
        
        n_samples = len(y_subset)
        n_classes = len(np.unique(y_subset))
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            return DecisionTreeNode(
                value=self._majority_class(y_subset),
                class_distribution=Counter(y_subset),
                is_leaf=True
            )
        
        best_split = self._best_split(X_subset, y_subset)
        
        if best_split is None or best_split['p_value'] >= self.alpha_split:
            return DecisionTreeNode(
                value=self._majority_class(y_subset),
                class_distribution=Counter(y_subset),
                is_leaf=True
            )
        
        node = DecisionTreeNode(
            feature_idx=best_split['feature_idx'],
            threshold=best_split['split_values'],
            class_distribution=Counter(y_subset)
        )
        node.is_multiway = True
        node.children = {}
        
        # Create child for each split value
        X_feature = X_subset[:, best_split['feature_idx']]
        
        for split_val in best_split['split_values']:
            child_indices = indices[X_feature == split_val]
            if len(child_indices) > 0:
                node.children[split_val] = self._build_tree(X, y, depth + 1, child_indices)
        
        return node
    
    def _predict_one(self, x, node):
        """Predict with multi-way splits"""
        if node.is_leaf:
            return node.value
        
        if hasattr(node, 'is_multiway') and node.is_multiway:
            x_val = x[node.feature_idx]
            if x_val in node.children:
                return self._predict_one(x, node.children[x_val])
            else:
                return node.class_distribution.most_common(1)[0][0]
        else:
            if x[node.feature_idx] <= node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)