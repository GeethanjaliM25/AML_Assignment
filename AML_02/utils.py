# src/utils.py
"""
Utility functions for evaluation and visualization
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """Evaluate a model and return metrics"""
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred_train = model.predict(X_train)
    predict_time_train = time.time() - start_time
    
    start_time = time.time()
    y_pred_test = model.predict(X_test)
    predict_time_test = time.time() - start_time
    
    metrics = {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_time': train_time,
        'predict_time_train': predict_time_train,
        'predict_time_test': predict_time_test,
        'train_precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
        'test_precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'train_recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'train_f1': f1_score(y_train, y_pred_train, average='weighted', zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'test_confusion_matrix': confusion_matrix(y_test, y_pred_test)
    }
    
    return metrics, y_pred_test


def compare_models(all_metrics):
    """Compare multiple models and return a comparison DataFrame"""
    comparison_data = []
    
    for metrics in all_metrics:
        comparison_data.append({
            'Model': metrics['model_name'],
            'Train Accuracy': metrics['train_accuracy'],
            'Test Accuracy': metrics['test_accuracy'],
            'Train Precision': metrics['train_precision'],
            'Test Precision': metrics['test_precision'],
            'Train Recall': metrics['train_recall'],
            'Test Recall': metrics['test_recall'],
            'Train F1': metrics['train_f1'],
            'Test F1': metrics['test_f1'],
            'Train Time (s)': metrics['train_time'],
            'Predict Time (s)': metrics['predict_time_test']
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def print_model_results(metrics, model_name):
    """Print detailed results for a model"""
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Train Precision: {metrics['train_precision']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Train Recall: {metrics['train_recall']:.4f}")
    print(f"Test Recall: {metrics['test_recall']:.4f}")
    print(f"Train F1 Score: {metrics['train_f1']:.4f}")
    print(f"Test F1 Score: {metrics['test_f1']:.4f}")
    print(f"Training Time: {metrics['train_time']:.4f} seconds")
    print(f"Prediction Time (Test): {metrics['predict_time_test']:.4f} seconds")
    print(f"\nConfusion Matrix (Test):")
    print(metrics['test_confusion_matrix'])
    
    # Extract feature importance if available
    if hasattr(metrics.get('model', None), 'feature_importances_'):
        print("\nFeature Importances:")
        print(metrics['model'].feature_importances_)


def analyze_dataset(df_info):
    """Print dataset analysis"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"Dataset Shape: {df_info['shape']}")
    print(f"\nColumns ({len(df_info['columns'])}):")
    for col in df_info['columns']:
        print(f"  - {col}: {df_info['dtypes'].get(col, 'unknown')}")
    
    print(f"\nMissing Values:")
    for col, count in df_info['null_counts'].items():
        if count > 0:
            print(f"  - {col}: {count}")
    
    print(f"\nTarget Distribution (Flight Status):")
    for status, count in df_info['target_distribution'].items():
        percentage = count / sum(df_info['target_distribution'].values()) * 100
        print(f"  - {status}: {count} ({percentage:.1f}%)")