# src/main.py (Corrected with proper parameters)
"""
Extended Main Script - Run all 8 Decision Tree Types
"""
import sys
import os
import warnings
import time
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import DataLoader
from src.decision_tree_id3 import ID3DecisionTree
from src.decision_tree_c45 import C45DecisionTree
from src.decision_tree_c45_pruned import C45PrunedDecisionTree
from src.decision_tree_cart import CARTDecisionTree
from src.decision_tree_chaid import CHAIDDecisionTree
from src.decision_tree_randomized import RandomizedDecisionTree
from src.decision_tree_oblique import ObliqueDecisionTree
from src.utils import evaluate_model, compare_models, print_model_results, analyze_dataset


def main():
    print("="*80)
    print("DECISION TREE IMPLEMENTATION - 8 TYPES FROM SCRATCH")
    print("="*80)
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    # Preprocess with sample
    SAMPLE_SIZE = 20000
    print(f"\nUsing {SAMPLE_SIZE} samples for training\n")
    
    X, y, feature_names = data_loader.preprocess(
        target_col='Flight Status', 
        sample_size=SAMPLE_SIZE
    )
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y, test_size=0.2)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {X.shape[1]}\n")
    
    # Initialize all 8 models
    models = {
        '1. ID3 (Info Gain)': ID3DecisionTree(
            max_depth=8, min_samples_split=50, min_samples_leaf=25, max_features=10
        ),
        '2. C4.5 (Gain Ratio)': C45DecisionTree(
            max_depth=8, min_samples_split=50, min_samples_leaf=25, prune=False
        ),
        '3. C4.5 (Pruned)': C45PrunedDecisionTree(
            max_depth=8, min_samples_split=50, min_samples_leaf=25, prune=True
        ),
        '4. CART (Gini)': CARTDecisionTree(
            max_depth=8, min_samples_split=50, min_samples_leaf=25
        ),
        '5. CHAID (Chi-square)': CHAIDDecisionTree(
            max_depth=6, min_samples_split=50, min_samples_leaf=25, max_categories=10
        ),
        '6. Randomized DT': RandomizedDecisionTree(
            max_depth=8, min_samples_split=50, min_samples_leaf=25, max_features='sqrt', random_split=False
        ),
        '7. Oblique DT': ObliqueDecisionTree(
            max_depth=6, min_samples_split=50, min_samples_leaf=25, use_oblique=False
        ),
        '8. Randomized DT (Full Features)': RandomizedDecisionTree(
            max_depth=8, min_samples_split=50, min_samples_leaf=25, max_features=None, random_split=False
        ),
    }
    
    # Evaluate all models
    all_metrics = []
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            metrics, predictions = evaluate_model(model, X_train, y_train, X_test, y_test, name)
            elapsed = time.time() - start_time
            
            all_metrics.append(metrics)
            print_model_results(metrics, name)
            print(f"Total time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare all models
    print("\n" + "="*80)
    print("COMPLETE MODEL COMPARISON - 8 DECISION TREE TYPES")
    print("="*80)
    
    if all_metrics:
        comparison_df = compare_models(all_metrics)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_idx = np.argmax([m['test_accuracy'] for m in all_metrics])
        best = all_metrics[best_idx]
        
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {best['model_name']}")
        print("="*80)
        print(f"Test Accuracy: {best['test_accuracy']:.4f} ({best['test_accuracy']*100:.2f}%)")
        print(f"Test F1 Score: {best['test_f1']:.4f}")
        print(f"Training Time: {best['train_time']:.2f} seconds")
        
        # Save results
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        comparison_df.to_csv(output_dir / 'all_8_models_comparison.csv', index=False)
        print(f"\n✅ Results saved to {output_dir / 'all_8_models_comparison.csv'}")
    else:
        print("\n❌ No models were successfully trained!")
    
    return all_metrics


if __name__ == "__main__":
    results = main()