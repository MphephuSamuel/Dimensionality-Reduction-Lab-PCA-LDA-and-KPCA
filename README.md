# Dimensionality Reduction Techniques: PCA, LDA, and KPCA

This notebook explores three dimensionality reduction techniques: Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Kernel Principal Component Analysis (KPCA). It demonstrates their implementation from scratch (for PCA and KPCA) and using scikit-learn, applies them to the Wine dataset and synthetic datasets (half-moon and concentric circles), and compares their performance for classification tasks.

## Table of Contents

1.  [Part 1: Principal Component Analysis (PCA)](#part-1-principal-component-analysis-pca)
    *   [Step 1.1: Load and Preprocess the Wine Dataset](#step-11-load-and-preprocess-the-wine-dataset)
    *   [Step 1.2: Implement PCA from Scratch](#step-12-implement-pca-from-scratch)
    *   [Step 1.3: Visualize Explained Variance](#step-13-visualize-explained-variance)
    *   [Step 1.4: PCA with scikit-learn](#step-14-pca-with-scikit-learn)
    *   [Step 1.5: Classification and Visualization](#step-15-classification-and-visualization)
2.  [Part 2: Linear Discriminant Analysis (LDA)](#part-2-linear-discriminant-analysis-lda)
    *   [Step 2.1: Compute Scatter Matrices](#step-21-compute-scatter-matrices)
    *   [Step 2.2: Eigendecomposition and Projection](#step-22-eigendecomposition-and-projection)
    *   [Step 2.3: LDA with scikit-learn](#step-23-lda-with-scikit-learn)
    *   [Step 2.4: Classification and Visualization](#step-24-classification-and-visualization)
3.  [Part 3: Kernel Principal Component Analysis (KPCA)](#part-3-kernel-principal-component-analysis-kpca)
    *   [Step 3.1: Implement RBF Kernel PCA](#step-31-implement-rbf-kernel-pca)
    *   [Step 3.2: Half-Moon Dataset](#step-32-half-moon-dataset)
    *   [Step 3.3: Concentric Circles Dataset](#step-33-concentric-circles-dataset)
    *   [Step 3.4: KPCA with scikit-learn](#step-34-kpca-with-scikit-learn)
4.  [Analysis Questions](#analysis-questions)
5.  [Limitations](#limitations)
6.  [Classifier Performance Comparison](#classifier-performance-comparison)

## Part 1: Principal Component Analysis (PCA)

PCA is an unsupervised technique that finds principal components (directions of maximum variance) to reduce dimensionality.

### Step 1.1: Load and Preprocess the Wine Dataset

Loads the Wine dataset, splits it into features and labels, and standardizes the features.

### Step 1.2: Implement PCA from Scratch

Demonstrates the manual steps to perform PCA, including computing the covariance matrix, performing eigendecomposition, and creating a projection matrix.

### Step 1.3: Visualize Explained Variance

Visualizes the explained variance ratio of each principal component and the cumulative explained variance to help determine the number of components to retain.

### Step 1.4: PCA with scikit-learn

Shows how to use the `KernelPCA` class from scikit-learn to perform PCA.

### Step 1.5: Classification and Visualization

Applies Logistic Regression to the PCA-transformed data and visualizes the decision regions.

## Part 2: Linear Discriminant Analysis (LDA)

LDA is supervised and maximizes class separability.

### Step 2.1: Compute Scatter Matrices

Calculates the within-class scatter matrix ($S_W$) and the between-class scatter matrix ($S_B$).

### Step 2.2: Eigendecomposition and Projection

Solves the generalized eigenvalue problem for $S_W^{-1}S_B$ to find the linear discriminants and creates a projection matrix.

### Step 2.3: LDA with scikit-learn

Shows how to use the `LinearDiscriminantAnalysis` class from scikit-learn to perform LDA.

### Step 2.4: Classification and Visualization

Applies Logistic Regression to the LDA-transformed data and visualizes the decision regions.

## Part 3: Kernel Principal Component Analysis (KPCA)

KPCA is used for nonlinear data.

### Step 3.1: Implement RBF Kernel PCA

Provides a manual implementation of KPCA using the Radial Basis Function (RBF) kernel.

### Step 3.2: Half-Moon Dataset

Applies the custom RBF kernel PCA to the half-moon dataset and visualizes the transformed data.

### Step 3.3: Concentric Circles Dataset

Applies the custom RBF kernel PCA to the concentric circles dataset and visualizes the transformed data, showing how the first component can separate the classes.

### Step 3.4: KPCA with scikit-learn

Shows how to use the `KernelPCA` class from scikit-learn to perform KPCA.

## Analysis Questions

This section contains questions and answers related to the concepts and results presented in the notebook, including:

*   Explained variance in PCA.
*   Comparison of PCA and LDA for classification.
*   The effect of the gamma parameter in KPCA.

## Limitations

Discusses the limitations of standard PCA, provides an example (Swiss Roll), and explains how KPCA addresses nonlinearity using the kernel trick.

## Classifier Performance Comparison

Compares the accuracy and computation time of a classifier (Logistic Regression) applied to the original, PCA-transformed, and LDA-transformed Wine data. The results are presented in a table, showing that for this dataset, LDA provided the best balance of efficiency and accuracy.
