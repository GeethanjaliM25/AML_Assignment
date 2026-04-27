# src/decision_tree_randomized.py
"""
Randomized Decision Tree
Randomly selects features and split points for each node
"""
import numpy as np
from .decision_tree_base import BaseDecisionTree, DecisionTreeNode


class RandomizedDecisionTree(BaseDecisionTree):
    """
    Randomized Decision Tree implementation
    
    Characteristics:
    - Randomly selects subset of features at each node
    - Randomly selects split points
    - Can be used as base estimator for Random Forest
    - Introduces randomness to reduce overfitting
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features='sqrt', random_split=False):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        
        # Configure max_features
        if max_features == 'sqrt':
            self.max_features = 'sqrt'
        elif max_features == 'log2':
            self.max_features = 'log2'
        else:
            self.max_features = max_features
            
        self.random_split = random_split  # If True, randomly selects split point
        self.random_state = None
        
    def set_random_state(self, random_state):
        """Set random state for reproducibility"""
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _impurity(self, y):
        """Can use either entropy or gini"""
        return self._gini(y)
    
    def _get_max_features(self, n_features):
        """Determine number of features to consider at each split"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            return n_features
    
    def _random_numerical_split(self, X, y, feature_idx):
        """Randomly select a split point"""
        values = X[:, feature_idx]
        unique_values = np.unique(values)
        
        if len(unique_values) < 2:
            return None, None, None, None
        
        # Randomly select one of the unique values as split point
        random_idx = np.random.randint(0, len(unique_values) - 1)
        threshold = (unique_values[random_idx] + unique_values[random_idx + 1]) / 2
        
        left_mask = values <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return None, None, None, None
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        gain = self._information_gain(y, y_left, y_right)
        
        return gain, threshold, left_mask, right_mask
    
    def _best_numerical_split(self, X, y, feature_idx):
        """Find best split for numerical feature (traditional)"""
        sorted_indices = np.argsort(X[:, feature_idx])
        sorted_X = X[sorted_indices, feature_idx]
        sorted_y = y[sorted_indices]
        
        best_gain = -1
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        
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
    
    def _best_split(self, X, y):
        """Find best split with random feature selection"""
        n_samples, n_features = X.shape
        n_selected = self._get_max_features(n_features)
        
        # Randomly select features
        selected_features = np.random.choice(n_features, n_selected, replace=False)
        
        best_split = None
        
        for feature_idx in selected_features:
            unique_values = len(np.unique(X[:, feature_idx]))
            
            if self.random_split and unique_values > 10:
                gain, threshold, left_mask, right_mask = self._random_numerical_split(X, y, feature_idx)
            else:
                gain, threshold, left_mask, right_mask = self._best_numerical_split(X, y, feature_idx)
            
            if gain is not None and gain > 0:
                if best_split is None or gain > best_split['gain']:
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gain': gain,
                        'left_indices': left_mask,
                        'right_indices': right_mask
                    }
        
        return best_split