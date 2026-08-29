# 📰 News Title Classification — SVM

### TF-IDF Feature Engineering + Support Vector Machine Classification + Model Evaluation

This project implements an end-to-end **News Title Classification** pipeline using **TF-IDF features and Support Vector Machines (SVM)**.

The system takes news-title TF-IDF features, combines them with labeled categories, trains multiple SVM variants, evaluates their performance using multiple classification metrics, and generates a formatted Excel report containing the complete experimental results.

---

## 📊 Project Results

The experiment was performed on **4,999 labeled news titles** across six categories:

**Business · Energy · Health · Markets · Politics · Technology**

Four SVM models were trained and compared:

- Linear SVM (`LinearSVC`)
- SVC with Linear Kernel
- SVC with RBF Kernel
- SVC with Polynomial Kernel

The best model was selected using **Macro F1**, since the dataset is imbalanced.

### 🏆 Best Model

**Linear SVM (LinearSVC)**

- Accuracy: **63.8%**
- Weighted F1: **63.6%**
- Macro F1: **56.8%**
- Best Category: **Politics**
- Best Category F1: **0.74**
- Weakest Category: **Health**
- Weakest Category F1: **0.41**

---

## 🖼️ Project Report

### 📋 Project Summary

<p align="center">
  <img src="assets/project_summary.png" width="100%">
</p>

The Project Summary sheet contains the complete overview of the dataset, feature engineering process, category distribution, trained models, headline results, and key findings.

---

### 📊 Model Comparison

<p align="center">
  <img src="assets/model_comparison.png" width="100%">
</p>

The Model Comparison sheet compares all four SVM models using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1
- Macro Precision
- Macro Recall
- Macro F1

The green row represents the selected best model based on **Macro F1**.

---

### 📈 Per-Category Metrics

<p align="center">
  <img src="assets/per_category_metrics.png" width="100%">
</p>

This sheet provides detailed classification performance for every category, including:

- Precision
- Recall
- F1-score
- Support

For the selected LinearSVC model, **Politics achieved the highest F1-score (0.74)**, while **Health achieved the lowest F1-score (0.41)**.

---

### 🔥 Confusion Matrix

<p align="center">
  <img src="assets/confusion_matrix.png" width="100%">
</p>

The confusion matrix shows the actual versus predicted categories for the best-performing model.

The highlighted diagonal represents correctly classified news titles, while the off-diagonal values represent classification errors.

---

# 🔄 End-to-End Pipeline

```text
                News Titles
                     │
                     ▼
          TF-IDF Feature Extraction
                     │
                     ▼
           Sparse Feature Matrix
                     │
                     ▼
             Label Alignment
                     │
                     ▼
            Label Encoding
                     │
                     ▼
          Stratified Train/Test Split
                 80% / 20%
                     │
                     ▼
       ┌────────────────────────────┐
       │        SVM Models           │
       ├────────────────────────────┤
       │ LinearSVC                   │
       │ SVC - Linear                │
       │ SVC - RBF                   │
       │ SVC - Polynomial            │
       └────────────────────────────┘
                     │
                     ▼
             Model Evaluation
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Accuracy    Precision      Recall
                     │
                     ▼
               F1 Score
             Macro + Weighted
                     │
                     ▼
             Best Model Selection
               Highest Macro F1
                     │
                     ▼
              Excel Report
