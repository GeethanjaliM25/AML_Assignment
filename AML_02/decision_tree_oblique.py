# src/decision_tree_oblique.py
"""
Oblique Decision Tree
Splits using linear combinations of features (axis-aligned or oblique)
"""
import numpy as np
from collections import Counter
from .decision_tree_base import BaseDecisionTree, DecisionTreeNode


class ObliqueDecisionTree(BaseDecisionTree):
    """
    Oblique Decision Tree implementation
    
    Characteristics:
    - Can use linear combinations of features for splitting (oblique)
    - Axis-aligned splits by default
    - Uses random projections for oblique splits
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 use_oblique=False, oblique_samples=10):
        """
        Parameters:
        -----------
        max_depth : int, default=None
            Maximum depth of the tree
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum number of samples required to be at a leaf node
        use_oblique : bool, default=False
            Whether to use oblique splits (linear combinations) or axis-aligned splits
        oblique_samples : int, default=10
            Number of random projections to try for oblique splits
        """
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.use_oblique = use_oblique
        self.oblique_samples = oblique_samples
        
    def _impurity(self, y):
        """Use Gini impurity for splitting"""
        return self._gini(y)
    
    def _find_oblique_split(self, X, y):
        """
        Find oblique split using random projections
        
        An oblique split uses a linear combination of features:
        w1*x1 + w2*x2 + ... + wn*xn <= threshold
        
        This implementation generates random projection vectors and finds
        the best threshold for each projection.
        """
        n_samples, n_features = X.shape
        
        if n_samples < 2 or n_features < 2:
            return None
        
        best_gain = -1
        best_projection = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        
        # Try multiple random projections
        for _ in range(self.oblique_samples):
            # Generate random projection vector (unit vector)
            projection = np.random.randn(n_features)
            norm = np.linalg.norm(projection)
            if norm > 0:
                projection = projection / norm
            
            # Project data onto the random vector
            X_projected = X @ projection
            
            # Find best threshold on projected data
            unique_values = np.unique(X_projected)
            if len(unique_values) < 2:
                continue
            
            # Try all possible split points
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_mask = X_projected <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples constraint
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_projection = projection
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask
        
        if best_gain <= 0 or best_projection is None:
            return None
        
        return {
            'projection': best_projection,
            'threshold': best_threshold,
            'gain': best_gain,
            'left_indices': best_left_mask,
            'right_indices': best_right_mask,
            'is_oblique': True
        }
    
    def _find_axis_aligned_split(self, X, y):
        """
        Find best axis-aligned split (traditional decision tree split)
        
        This considers each feature individually and finds the best threshold.
        """
        n_samples, n_features = X.shape
        best_split = None
        
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            
            if len(unique_values) < 2:
                continue
            
            # Sort values for efficient threshold search
            sorted_indices = np.argsort(X[:, feature_idx])
            sorted_X = X[sorted_indices, feature_idx]
            sorted_y = y[sorted_indices]
            
            # Try splits between consecutive unique values
            for i in range(len(sorted_X) - 1):
                if sorted_X[i] == sorted_X[i + 1]:
                    continue
                
                threshold = (sorted_X[i] + sorted_X[i + 1]) / 2
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples constraint
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                gain = self._information_gain(y, y_left, y_right)
                
                if best_split is None or gain > best_split['gain']:
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gain': gain,
                        'left_indices': left_mask,
                        'right_indices': right_mask,
                        'is_oblique': False
                    }
        
        return best_split
    
    def _best_split(self, X, y):
        """
        Find the best split (either oblique or axis-aligned)
        
        If use_oblique is True, try oblique splits first.
        Fall back to axis-aligned if oblique doesn't find a good split.
        """
        best_split = None
        
        if self.use_oblique:
            # Try oblique splits
            oblique_split = self._find_oblique_split(X, y)
            if oblique_split is not None:
                best_split = oblique_split
        
        # Always try axis-aligned splits as baseline
        axis_split = self._find_axis_aligned_split(X, y)
        
        # Choose the better split
        if best_split is None and axis_split is not None:
            best_split = axis_split
        elif best_split is not None and axis_split is not None:
            if axis_split['gain'] > best_split['gain']:
                best_split = axis_split
        
        return best_split
    
    def _build_tree(self, X, y, depth=0, indices=None):
        """
        Recursively build the decision tree
        
        Handles both oblique and axis-aligned internal nodes.
        """
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
            class_distribution=Counter(y_subset)
        )
        
        # Store split information
        if best_split.get('is_oblique', False):
            node.is_oblique = True
            node.projection = best_split['projection']
            node.threshold = best_split['threshold']
            node.feature_idx = None  # Not used for oblique splits
        else:
            node.is_oblique = False
            node.feature_idx = best_split['feature_idx']
            node.threshold = best_split['threshold']
        
        # Recursively build children
        left_indices = indices[best_split['left_indices']]
        right_indices = indices[best_split['right_indices']]
        
        node.left = self._build_tree(X, y, depth + 1, left_indices)
        node.right = self._build_tree(X, y, depth + 1, right_indices)
        
        return node
    
    def _predict_one(self, x, node):
        """
        Predict a single sample
        
        Handles both oblique and axis-aligned splits during prediction.
        """
        if node.is_leaf:
            return node.value
        
        if getattr(node, 'is_oblique', False):
            # For oblique split: compute projection and compare
            projection_value = np.dot(x, node.projection)
            if projection_value <= node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
        else:
            # For axis-aligned split: compare single feature
            if x[node.feature_idx] <= node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
    
    def get_oblique_coefficients(self):
        """
        Extract oblique coefficients from the tree
        
        Returns a list of dictionaries containing projection vectors
        and thresholds for each oblique node in the tree.
        """
        coefficients = []
        
        def _collect_coeffs(node):
            if node is None or node.is_leaf:
                return
            if getattr(node, 'is_oblique', False):
                coefficients.append({
                    'projection': node.projection.tolist(),
                    'threshold': node.threshold
                })
            _collect_coeffs(node.left)
            _collect_coeffs(node.right)
        
        _collect_coeffs(self.root)
        return coefficients
    
    def print_tree(self, node=None, depth=0, feature_names=None):
        """
        Print the tree structure for visualization
        
        Parameters:
        -----------
        node : DecisionTreeNode, default=None
            Current node (starts from root)
        depth : int, default=0
            Current depth in the tree
        feature_names : list, default=None
            Names of features for better readability
        """
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.is_leaf:
            print(f"{indent}Leaf: class={node.value}, samples={sum(node.class_distribution.values())}")
            print(f"{indent}Distribution: {dict(node.class_distribution)}")
        else:
            if getattr(node, 'is_oblique', False):
                print(f"{indent}Oblique Node (depth={depth}): projection={node.projection[:3]}..., threshold={node.threshold:.4f}")
            else:
                feat_name = feature_names[node.feature_idx] if feature_names else f"Feature {node.feature_idx}"
                print(f"{indent}Axis-Aligned Node (depth={depth}): {feat_name} <= {node.threshold:.4f}")
            
            print(f"{indent}  Left branch:")
            self.print_tree(node.left, depth + 1, feature_names)
            print(f"{indent}  Right branch:")
            self.print_tree(node.right, depth + 1, feature_names)