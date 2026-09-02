
# 📰 News Title Classification using SVM

> An end-to-end Machine Learning / NLP pipeline for classifying news titles into multiple categories using TF-IDF features and Support Vector Machine (SVM) models.

---

## 📌 Project Overview

This project implements an end-to-end **News Title Classification** system using classical Machine Learning techniques.

The pipeline takes pre-computed **TF-IDF text features**, combines them with manually labeled news categories, trains multiple **Support Vector Machine (SVM)** models, evaluates their performance, and generates a complete formatted Excel report containing model comparisons, per-category metrics, and a confusion matrix.

### 🎯 News Categories

The classifier predicts news titles into six categories:

- 💼 Business
- ⚡ Energy
- ❤️ Health
- 📈 Markets
- 🏛️ Politics
- 💻 Technology

---

## 🔄 End-to-End Pipeline

```text
News Classification Labels (.xlsx)
              +
TF-IDF Features (.csv)
              │
              ▼
      Data Cleaning & Alignment
              │
              ▼
      Sparse TF-IDF Feature Matrix
              │
              ▼
       Label Encoding
              │
              ▼
     Stratified Train/Test Split
          80% / 20%
              │
              ▼
      ┌───────────────────────┐
      │     SVM Models        │
      ├───────────────────────┤
      │ LinearSVC              │
      │ SVC - Linear           │
      │ SVC - RBF              │
      │ SVC - Polynomial       │
      └───────────────────────┘
              │
              ▼
       Model Evaluation
              │
              ▼
 Accuracy / Precision / Recall
       Weighted & Macro F1
              │
              ▼
       Best Model Selection 
        Highest Macro F1
              │
              ▼
     📊 Excel Results Report
