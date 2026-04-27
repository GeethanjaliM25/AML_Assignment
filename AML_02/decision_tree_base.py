# src/decision_tree_base.py (FIXED - add Counter import)
"""
Base Decision Tree Class with common functionality
"""
import numpy as np
from collections import Counter
from abc import ABC, abstractmethod


class DecisionTreeNode:
    """Node in the decision tree"""
    
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, 
                 value=None, class_distribution=None, is_leaf=False):
        self.feature_idx = feature_idx      # Index of feature to split on
        self.threshold = threshold           # Threshold for split (for numerical features)
        self.left = left                     # Left child node
        self.right = right                   # Right child node
        self.value = value                   # Predicted class (for leaf nodes)
        self.class_distribution = class_distribution  # Distribution of classes at this node
        self.is_leaf = is_leaf


class BaseDecisionTree(ABC):
    """Abstract base class for all decision tree implementations"""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes = None
        self.n_features = None
        self.feature_names = None
        
    @abstractmethod
    def _best_split(self, X, y):
        """Find the best split - to be implemented by each tree type"""
        pass
    
    @abstractmethod
    def _impurity(self, y):
        """Calculate impurity measure - to be implemented by each tree type"""
        pass
    
    def _entropy(self, y):
        """Calculate entropy (common utility)"""
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _gini(self, y):
        """Calculate Gini impurity (common utility)"""
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(p ** 2 for p in ps)
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain (common utility)"""
        parent_impurity = self._impurity(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
            
        child_impurity = (n_left / n) * self._impurity(y_left) + (n_right / n) * self._impurity(y_right)
        return parent_impurity - child_impurity
    
    def _gain_ratio(self, y, y_left, y_right):
        """Calculate gain ratio (for C4.5)"""
        info_gain = self._information_gain(y, y_left, y_right)
        
        # Calculate split info
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
            
        split_info = - (n_left/n) * np.log2(n_left/n) - (n_right/n) * np.log2(n_right/n)
        
        if split_info == 0:
            return 0
            
        return info_gain / split_info
    
    def _majority_class(self, y):
        """Return the majority class in y"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _build_tree(self, X, y, depth=0, indices=None):
        """Recursively build the decision tree"""
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
    
    def fit(self, X, y, feature_names=None):
        """Fit the decision tree to the data"""
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.feature_names = feature_names
        
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_one(self, x, node):
        """Predict one sample"""
        if node.is_leaf:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        # To be implemented by child classes if needed
        predictions = self.predict(X)
        proba = np.zeros((len(X), self.n_classes))
        for i, pred in enumerate(predictions):
            proba[i, pred] = 1
        return proba
    
    def _get_tree_info(self, node, depth=0):
        """Get information about the tree structure"""
        if node.is_leaf:
            return {
                'type': 'leaf',
                'depth': depth,
                'value': node.value,
                'class_distribution': dict(node.class_distribution)
            }
        else:
            return {
                'type': 'internal',
                'depth': depth,
                'feature': node.feature_idx,
                'threshold': node.threshold,
                'left': self._get_tree_info(node.left, depth + 1),
                'right': self._get_tree_info(node.right, depth + 1)
            }
    
    def get_tree_structure(self):
        """Return the tree structure as a dictionary"""
        if self.root is None:
            return None
        return self._get_tree_info(self.root)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)