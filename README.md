# 🍷 Wine Quality Prediction - Unsupervised + Supervised ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Competition-Kaggle-20BEFF)](https://www.kaggle.com/competitions/wine-ensgti-2026)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/yourusername/wine-quality-prediction)

## 🎯 Overview

This project implements a **hybrid ML architecture** that combines unsupervised clustering with supervised classification to predict wine quality scores (3-9 scale) from physicochemical properties. The pipeline processes **3,898 wine samples** through dimensionality reduction, cluster discovery, and ensemble learning to achieve **61.14% accuracy** on Kaggle's public leaderboard.

The solution transitions from raw chemical measurements to actionable quality predictions, revealing that wine excellence is driven by a complex interplay of **alcohol content, sulphate levels, and acidity balance** — patterns that emerge clearly only after strategic feature engineering.

**Competition:** Wine Quality Prediction - ENSGTI 2026  
**Final Leaderboard:** 61.14% (Public) | 65.0% (Validation)  
**Approach:** Clustering → Feature Engineering → Ensemble Classification

---



- **Core ML:** `scikit-learn`, `xgboost`, `lightgbm`
- **Clustering:** `KMeans`, `DBSCAN`, `HDBSCAN`
- **Dimensionality Reduction:** `PCA`, `t-SNE`, `UMAP`
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Analysis Type:** Hybrid Unsupervised + Supervised Learning

---

## 📈 Pipeline Phases

### 1. Data Exploration & Feature Understanding

Before jumping into modeling, deep exploratory analysis revealed the hidden structure:

* **Correlation Mining:** Discovered that `alcohol` and `density` have a strong negative correlation (-0.50), while `fixed acidity` and `density` move together (0.67).
* **Distribution Patterns:** Used violin plots to identify that most features follow near-normal distributions, validating the use of StandardScaler.
* **Variance Decomposition:** PCA analysis showed that **85% of variance** is captured in just 7 components out of 11 features — signaling redundancy.

**Key Discovery:** Wine quality isn't uniformly distributed — it clusters around scores 5-6 (medium quality), creating a class imbalance challenge.

### 2. Unsupervised Learning - Cluster Discovery

A three-algorithm approach to find natural wine groupings:

* **KMeans (k=3):** Selected via Elbow Method + Silhouette Analysis (score: 0.2138 at k=2, but k=3 chosen for interpretability)
  - **Cluster 0 (1,156 wines):** "High-Alcohol Classics" — High alcohol, moderate acidity
  - **Cluster 1 (1,392 wines):** "Sweet & Stable" — High residual sugar, low alcohol, high sulfur dioxide
  - **Cluster 2 (1,350 wines):** "Crisp & Balanced" — Low fixed acidity, high pH and sulphates

* **DBSCAN (ε=2.25):** Density-based approach with K-distance plot tuning
  - Result: 1 large cluster + 241 noise points
  - **Insight:** Dataset lacks distinct density separations, suggesting overlapping quality zones

* **HDBSCAN (min_cluster_size=20):** Hierarchical density clustering
  - Result: 3 clusters + 3,612 noise points (93% labeled as noise!)
  - **Insight:** Ultra-conservative — only highly confident core regions accepted

**Strategic Choice:** Used KMeans clusters as features despite moderate silhouette score, as they added **predictive signal** in supervised models.

### 3. Dimensionality Reduction - Visualization Strategy

Three complementary techniques to visualize high-dimensional wine chemistry:

* **PCA (Principal Component Analysis):** Preserves global variance structure
  - Revealed: Clusters heavily overlap in PC1-PC2 space
  - Captured: 40% variance in first 2 components

* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Preserves local neighborhoods
  - Revealed: Better visual separation than PCA
  - Trade-off: Distorts global distances

* **UMAP (Uniform Manifold Approximation and Projection):** Balanced approach
  - Revealed: Best of both worlds — local structure + global topology
  - **Winner for visualization:** Clearest cluster boundaries

**Visualization Insight:** Ground truth quality labels show **massive overlap** — even in ideal projections, quality grades aren't cleanly separable, explaining the ~60% accuracy ceiling.

### 4. Feature Engineering - Cluster Intelligence

To bridge unsupervised insights with supervised learning:

* **Cluster Membership:** One-hot encoded cluster assignment (3 new features)
* **Geometric Distance:** Euclidean distance to each of 3 cluster centroids (3 new features)
* **Quality Binning:** Created `quality_binned` (Low: 3-4, Med: 5-6, High: 7-9) for cross-tabulation analysis

**Engineering Outcome:** Transformed 11 chemical features → **17 total features** (11 original + 1 cluster + 3 distances + 3 bins)

**Cross-Tab Analysis:**
- Cluster 2 has **35.1% high-quality wines** (best performer)
- Cluster 1 is dominated by **85.4% medium-quality** (safest bet)
- Cluster 0 balances across all quality tiers

### 5. Supervised Learning - Model Tournament

Four algorithms competed in a speed-optimized validation setup (80/20 split, no CV):

| Model | Validation Accuracy | Training Time | Key Strength |
|-------|-------------------|---------------|--------------|
| **Random Forest** ⭐ | **62.95%** | ~30s | Robust, interpretable |
| LightGBM | 61.92% | ~5s | Fast, gradient-based |
| XGBoost | 61.67% | ~15s | Regularization control |
| Gradient Boosting | 60.64% | ~45s | Baseline boosting |

**Winner:** Random Forest with `n_estimators=100, max_depth=12`

**Label Remapping:** Since quality scores are 3-9, XGBoost required remapping to 0-6 (0-indexed classes). Post-prediction, scores were converted back to 3-9 for submission.

---

## 🏆 Key Results

### Performance Metrics

| Stage | Accuracy | Dataset | Insight |
|-------|----------|---------|---------|
| **Training** | ~70% | 3,118 samples | High in-sample fit |
| **Validation** | 65.0% | 780 samples | Good generalization |
| **Public Test** | 61.14% | Unknown | ~4% overfitting gap ⚠️ |

**Overfitting Analysis:** The 4% drop from validation to test suggests the model memorized some training-specific patterns. Future work: Regularize Random Forest or ensemble multiple models.

### Cluster Performance Analysis

**Quality Distribution per Cluster:**

| Cluster | Low Quality (3-4) | Medium (5-6) | High (7-9) | Interpretation |
|---------|------------------|--------------|------------|----------------|
| 0 | 5.6% | 71.5% | 22.9% | Balanced mid-tier |
| 1 | 4.1% | 85.4% | 10.5% | Medium-quality factory |
| 2 | 4.2% | 60.7% | **35.1%** | Premium wine cluster ⭐ |

**Actionable Insight:** Cluster 2 wines (low acidity, high pH/sulphates) have **3.3× better odds** of being high-quality compared to Cluster 1.

### Feature Importance (Top 5)

From Random Forest's `feature_importances_`:

1. **Alcohol (18.2%)** — Strongest single predictor
2. **Sulphates (12.7%)** — Quality enhancer
3. **Volatile Acidity (11.3%)** — Negative quality driver
4. **Cluster Membership (9.1%)** — Unsupervised signal
5. **Total Sulfur Dioxide (8.8%)** — Preservation indicator

**Domain Validation:** Matches wine science — alcohol boosts body, sulphates add complexity, volatile acidity indicates spoilage.

---

## 📂 Repository Structure
```
wine-quality-prediction/
│
├── wine-dataset_predicted.ipynb    # Complete analysis workbook
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── submission.csv                   # Kaggle predictions (61.14%)
│
├── data/
│   ├── train.csv                   # 3,898 training samples
│   └── test.csv                    # Test set (unlabeled)
│
└── visualizations/
    ├── correlation_matrix.png      # Feature relationships
    ├── elbow_silhouette.png        # Optimal k selection
    ├── pca_clusters.png            # PCA projection comparison
    ├── tsne_clusters.png           # t-SNE projection
    ├── umap_clusters.png           # UMAP projection
    ├── cluster_profiles.png        # Chemical "personality" heatmap
    └── feature_importance.png      # Random Forest importances
```

---

## 🚀 Future Improvements

### Quick Wins (Expected +2-3%)

- ✅ **Ensemble Voting:** Combine RF + XGB + LGB predictions via majority vote
- ✅ **Regularization:** Reduce Random Forest complexity (`max_depth=10`, `min_samples_leaf=5`)
- ✅ **Feature Selection:** Drop least important features to reduce noise

### Advanced Strategies (Expected +3-5%)

- 🔧 **Stacking Ensemble:** Use base model predictions as meta-features
- 🔧 **Ordinal Regression:** Treat quality as ordered categories (3 < 4 < 5...)
- 🔧 **Interaction Features:** `alcohol × sulphates`, `fixed_acidity / volatile_acidity`
- 🔧 **Hyperparameter Optimization:** Bayesian search with Optuna (200+ iterations)

### Experimental Ideas

- 🧪 **Deep Learning:** Feedforward NN with embeddings for cluster membership
- 🧪 **Gaussian Mixture Models:** Soft clustering instead of hard KMeans
- 🧪 **SMOTE Oversampling:** Balance rare quality classes (3, 8, 9)

---

## 💡 Key Learnings & Insights

### Technical Takeaways

1. **Clustering ≠ Always Better:** KMeans features helped, but HDBSCAN's 93% noise rate showed not all clustering adds value
2. **Visualization Reveals Truth:** PCA/t-SNE/UMAP all showed **massive class overlap** — explaining the ~60% accuracy ceiling
3. **Random Forest > Boosting:** On this specific dataset, RF's bagging approach beat gradient methods
4. **Overfitting is Subtle:** 4% validation→test gap appeared even with conservative hyperparameters

### Business/Domain Insights

1. **Alcohol is King:** Single strongest predictor — high-alcohol wines score better
2. **Cluster 2 = Premium Zone:** 35% high-quality rate vs. 10% in Cluster 1
3. **Volatile Acidity = Red Flag:** Strong negative correlation with quality (spoilage indicator)
4. **Subjectivity Ceiling:** Wine quality is partially subjective — perfect prediction may be impossible

### Competition Strategy Lessons

1. **Late submissions still teach:** Even after deadline, Kaggle feedback validates local CV scores
2. **Overfitting detection matters:** Always compare validation vs. public leaderboard
3. **Feature engineering > model complexity:** Cluster features added more value than tuning
4. **Visualize everything:** Spending 30% of time on EDA prevented wasted modeling efforts

---

## 🏃 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/wine-quality-prediction.git
cd wine-quality-prediction

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.0.0
matplotlib==3.7.2
seaborn==0.12.2
hdbscan==0.8.33
umap-learn==0.5.3
```

### Run Analysis
```bash
# Launch Jupyter
jupyter notebook wine-dataset_predicted.ipynb

# Or run programmatically
python train_model.py  # (if you extract code to .py file)
```

### Generate Submission
```python
# In notebook or script
predictions = model.predict(X_test)
submission = pd.DataFrame({
    'id': test_ids,
    'quality': predictions + 3  # Convert 0-6 back to 3-9
})
submission.to_csv('submission.csv', index=False)
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Implement stacking ensemble
- [ ] Add hyperparameter tuning with Optuna
- [ ] Create interactive Plotly visualizations
- [ ] Experiment with neural network architectures
- [ ] Add SHAP values for explainability

**How to contribute:**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📧 Contact

**Prashant Shukla**  
📧 Email: prashantshukla8851@gmail.com  
💼 LinkedIn: [Prashant Shukla](https://www.linkedin.com/in/prashant-shukla-58ba19373)  
🔗 GitHub: [@pr4sh4nt-shukla](https://github.com/pr4sh4nt-shukla)

**Project Link:** [https://github.com/pr4sh4nt-shukla/Wine-Dataset-with-Unsupervised-Learning](https://github.com/pr4sh4nt-shukla/Wine-Dataset-with-Unsupervised-Learning)

---

## 🙏 Acknowledgments

- **Kaggle & ENSGTI 2026** — For hosting the competition and providing the platform
- **UCI Machine Learning Repository** — Original wine quality dataset source
- **scikit-learn Contributors** — For the comprehensive ML ecosystem
- **Open Source Community** — For XGBoost, LightGBM, UMAP, and HDBSCAN libraries

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project insightful, please consider giving it a star!** ⭐

*Wine quality prediction through the lens of data science — where chemistry meets machine learning.* 🍷📊
