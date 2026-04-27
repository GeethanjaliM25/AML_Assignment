# 🌳 Comprehensive Decision Tree Implementation from Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/)

[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![Version](https://img.shields.io/badge/Version-1.0.0-blue)]()
[![Documentation](https://img.shields.io/badge/Documentation-Yes-brightgreen.svg)]()
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange.svg)]()


## 🎯 Overview

This project provides a **complete from-scratch implementation** of 8 different Decision Tree algorithms for classification tasks. Built entirely with NumPy and pandas, no machine learning libraries (like scikit-learn) were used for the core algorithms, demonstrating deep understanding of the underlying mathematics and logic.

**Dataset**: Airline Passenger Dataset (98,619 rows, 15 columns)  
**Task**: Predict Flight Status classification  
**Performance**: Achieved up to 34.48% test accuracy across 8 models

## ✨ Key Features

- ✅ **8 Complete Decision Tree Implementations** from scratch
- ✅ **No ML Libraries** for core algorithms (pure NumPy/pandas)
- ✅ **Comprehensive Evaluation Metrics** (Accuracy, Precision, Recall, F1-Score)
- ✅ **Confusion Matrices** for each model
- ✅ **Automatic Results Comparison** and CSV export
- ✅ **Handles Mixed Data Types** (categorical & numerical)
- ✅ **Memory Efficient** with optional sampling
- ✅ **Professional Code Structure** with modular design

## 🌲 Decision Tree Types Implemented

| # | Algorithm | Splitting Criterion | Key Characteristics | Best Use Case |
|---|---|---|---|---|
| **1** | **ID3** | Information Gain | Basic entropy-based, categorical features | Small datasets with categorical features |
| **2** | **C4.5** | Gain Ratio | Improved ID3, handles continuous values | Medium-sized datasets, mixed data types |
| **3** | **C4.5 (Pruned)** | Gain Ratio + Pruning | Reduces overfitting with post-pruning | Large datasets, prevents overfitting |
| **4** | **CART** | Gini Impurity | Binary splits, classification & regression | General purpose, balanced datasets |
| **5** | **CHAID** | Chi-Square Test | Statistical significance, multi-way splits | Surveys, market research, categorical data |
| **6** | **Randomized DT** | Random Features | Random feature subset at each split | High-dimensional data, feature selection |
| **7** | **Oblique DT** | Oblique Splits | Diagonal decision boundaries | Complex patterns, correlated features |
| **8** | **Randomized DT (Full)** | Random Features (All) | Uses all features with random splits | Baseline comparison, diverse trees |

### Algorithm Complexity Comparison

| Algorithm | Time Complexity | Space Complexity | Overfitting Risk | Interpretability |
|-----------|----------------|------------------|------------------|------------------|
| ID3 | O(n log n) | O(n) | High | Very High |
| C4.5 | O(n²) | O(n²) | Medium | High |
| CART | O(n log n) | O(n) | Medium | High |
| CHAID | O(n²) | O(n²) | Low | Medium |
| Randomized | O(n log n) | O(n) | Low | Medium |
| Oblique | O(n³) | O(n²) | Medium | Low |


