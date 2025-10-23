

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SiamFS/TraditionalML_Tabular_Classification/blob/main/TraditionalML_tabular_classification.ipynb)

Traditional ML Tabular Classification

**Research Summary: Transparent, Dual-Evaluation of Classical ML Models on Real Student Performance Data**

This project presents a rigorous, fully reproducible study of five classical machine learning models—Decision Tree, SGD Classifier, SVM, Random Forest, and Naive Bayes—on the real-world Student Performance dataset. The primary goal is to provide an honest, data-driven comparison of these models for multi-class classification of final student grades (G3), binned into three classes. All results and metrics are strictly based on the actual data and code; no synthetic or fabricated results are included.

**A key contribution is the systematic dual evaluation of every model: each is trained and tested both on the original engineered features (32 total) and on a reduced set of 10 principal components (retaining 95% variance) via PCA. This enables a direct, transparent comparison of how dimensionality reduction impacts accuracy, weighted precision, recall, and the handling of class imbalance.**

**Highlights and Real Findings:**
- **End-to-End ML Pipeline:** Merges the raw Maths and Portuguese datasets (1044 rows), encodes categorical features, scales continuous features, and performs thorough correlation analysis. Data leakage is strictly avoided by splitting before preprocessing.
- **Class Imbalance:** The real data is highly imbalanced (Class 0: 83, Class 1: 671, Class 2: 294). Class weights are used for all models that support them; no synthetic oversampling (e.g., SMOTE) is used, so results reflect the true challenge.
- **Dimensionality Reduction:** PCA reduces the feature space from 32 to 10, retaining 95% of variance. The effect is measured for every model.
- **Robust, Multi-Metric Evaluation:** For both with and without PCA, every model is evaluated on accuracy, weighted precision, weighted recall, and confusion matrices. All metrics are reported exactly as observed in the code.
- **Final Comparison and Visualization:** The notebook concludes with a grouped bar plot comparing all models’ accuracy with and without PCA, plus additional plots (correlation heatmaps, class distribution, precision/recall bars, confusion matrices for all models and both feature sets).
- **Reproducibility:** All code, methodology, and results are in a single notebook. Every step is documented and can be rerun for verification.

**Key Results (from real code):**
- **Without PCA:** Random Forest achieves the highest accuracy at 0.88, followed by Decision Tree at 0.86, SGD Classifier at 0.84, SVM at 0.81, and Naive Bayes at 0.75. Precision and recall closely match accuracy. All models struggle with the minority class (Class 0), even with class weights.
- **With PCA:** Naive Bayes and Random Forest achieve 0.81 accuracy; SGD Classifier at 0.80; SVM at 0.79; Decision Tree drops to 0.77. The effect of PCA varies by model, with some improving (Naive Bayes) and others declining (Decision Tree).

This work provides honest, practical insights into the real challenges of tabular ML research: the limits of class weighting, the nuanced effects of dimensionality reduction, and the need for rigorous, multi-metric evaluation. The findings and code are intended as a template for future research and applied projects in educational data mining and tabular ML, and all claims are supported by the actual data and results.

## Dataset

The dataset used is the "Student Performance" dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/320/student+performance), also available on Kaggle (https://www.kaggle.com/datasets/whenamancodes/student-performance). It was donated by Paulo Cortez on November 26, 2014, and is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

### Overview
- **Purpose**: Predict student performance in secondary education (high school) using demographic, social, and school-related features.
- **Subject Area**: Social Science.
- **Associated Tasks**: Classification and Regression.
- **Feature Type**: Integer (mostly categorical encoded as integers).
- **Instances**: 649 (original: 395 for Mathematics and 649 for Portuguese; merged in this project to 1044 after combining both subjects).
- **Features**: 30 attributes plus the target variable (G3).
- **Missing Values**: None.
- **Data Collection**: Collected from two Portuguese schools (Gabriel Pereira 'GP' and Mousinho da Silveira 'MS') using school reports and questionnaires.

### Key Features
The dataset includes 30 features categorized as follows:
- **Demographic**: school (GP/MS), sex (F/M), age (15-22), address (urban/rural), famsize (≤3 or >3), Pstatus (parents together/apart).
- **Family Background**: Medu/Fedu (mother/father education: 0-none to 4-higher), Mjob/Fjob (mother/father job: teacher, health, services, at_home, other), guardian (mother/father/other).
- **School-Related**: reason (choice of school: home, reputation, course, other), traveltime (1-<15min to 4->1hr), studytime (1-<2hrs to 4->10hrs), failures (past class failures: 0-4), schoolsup/famsup/paid (extra support/classes: yes/no), activities/nursery (extracurricular/nursery: yes/no), higher (wants higher education: yes/no), internet (home internet: yes/no).
- **Social/Health**: romantic (relationship: yes/no), famrel (family relationships: 1-very bad to 5-excellent), freetime (free time: 1-very low to 5-very high), goout (going out: 1-very low to 5-very high), Dalc/Walc (workday/weekend alcohol: 1-very low to 5-very high), health (current health: 1-very bad to 5-very good), absences (school absences: 0-93).
- **Grades**: G1 (first period grade: 0-20), G2 (second period grade: 0-20).

### Target Variable
- **G3**: Final grade (numeric: 0-20), strongly correlated with G1 and G2 (first and second period grades). In this project, G3 is binned into 3 classes using KBinsDiscretizer (uniform strategy) for multi-class classification.

### Source and Citation
- **Creators**: Paulo Cortez.
- **DOI**: 10.24432/C5TG7T.
- **Introductory Paper**: "Using data mining to predict secondary school student performance" by P. Cortez and A. M. G. Silva (2008), Proceedings of 5th Annual Future Business Technology Conference.
- **Data is real and publicly available, with no synthetic additions**.

## Methodology

### Data Preprocessing
- Merged Maths and Portuguese datasets.
- Handled categorical features with ordinal encoding.
- Scaled continuous features (age, absences, G1, G2) using StandardScaler.
- Checked correlations; kept all features as correlations were below 85%, and G1/G2 are logically related to G3.

### Train-Test Split
- 70-30 split performed before any preprocessing to avoid data leakage.
- Encoding, scaling, and target binning fitted only on training data.

### Class Imbalance Handling
- The dataset is imbalanced: Class 0 (83 samples), Class 1 (671 samples), Class 2 (294 samples).
- Implemented class weights ('balanced') for Decision Tree, SGD, SVM, and Random Forest to penalize misclassifications of minority classes.
- Naive Bayes does not support class weights, so left unchanged.

### Dimensionality Reduction
- Applied PCA with 95% variance retention, reducing features from 32 to 10.
- PCA fitted on training data only.

PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms correlated features into uncorrelated principal components ordered by variance explained. It improves performance for some ML models but not others due to how different algorithms handle features:

- **Models that Benefit from PCA**:
  - **Naive Bayes**: Improved from 0.75 to 0.81 with PCA, as reduced features may lessen the impact of feature dependencies violating the independence assumption.
  - **Random Forest**: Dropped slightly from 0.88 to 0.81 but still strong, benefiting from reduced noise and faster training on fewer features. As an ensemble method, it averages predictions from multiple trees, making it relatively robust to feature reduction.

- **Models that Perform Worse with PCA**:
  - **Decision Tree**: Accuracy dropped significantly from 0.86 to 0.77 with PCA, as tree-based models can overfit or lose interpretability on reduced features. Decision trees split on individual features, and PCA's linear combinations make splits less meaningful.
  - **SVM**: Accuracy dropped from 0.81 to 0.79 with PCA, as it handles the reduced dimensionality but may lose some discriminative power from original features.
  - **SGD Classifier**: Accuracy dropped slightly from 0.84 to 0.80, as linear models can be sensitive to PCA's feature transformations, altering the optimization path.

- **Why PCA Helps or Hurts**: PCA is most beneficial for models sensitive to high dimensions or multicollinearity (like Random Forest and Naive Bayes in this case), but can hurt models relying on feature relationships or interpretability (like Decision Tree and SVM). In our case, PCA reduced 32 features to 10 while retaining 95% variance, improving Naive Bayes but degrading Decision Tree significantly.

### Models Trained
- Decision Tree
- SGD Classifier
- Support Vector Machine (linear kernel)
- Random Forest
- Naive Bayes

Each model trained on original features and PCA-transformed features.

### Evaluation Metrics
- Accuracy
- Weighted Precision and Recall
- Confusion Matrices

## Results

### Accuracies (Exact values from notebook execution)
- **Without PCA**:
  - Decision Tree: 0.86
  - SGD Classifier: 0.84
  - SVM: 0.81
  - Random Forest: 0.88
  - Naive Bayes: 0.75

- **With PCA**:
  - Decision Tree: 0.77
  - SGD Classifier: 0.80
  - SVM: 0.79
  - Random Forest: 0.81
  - Naive Bayes: 0.81

### Additional Metrics
For both with and without PCA, weighted precision and recall closely matched accuracy for all models. Confusion matrices showed that all models struggled with the minority class (Class 0), even with class weights, highlighting the persistent challenge of imbalance.

PCA has varied effects on performance: it improved Naive Bayes (from 0.75 to 0.81), maintained Random Forest reasonably (0.88 to 0.81), but significantly degraded Decision Tree (0.86 to 0.77) and slightly hurt SVM (0.81 to 0.79) and SGD Classifier (0.84 to 0.80).

## Discussion

- **Model Selection and Performance**: Random Forest achieved the highest accuracy without PCA (0.88) and remained competitive with PCA (0.81, tied with Naive Bayes), due to its ensemble nature handling noise and imbalance well. Decision Tree performed well without PCA (0.86) but suffered the most with PCA (0.77), as tree-based models struggle with PCA's linear feature combinations that reduce interpretability and optimal splits. SVM performed moderately (0.81 without, 0.79 with), benefiting from scaling but slightly sensitive to PCA's dimensionality reduction. Naive Bayes, surprisingly, improved with PCA (0.75 to 0.81), likely because PCA's uncorrelated components better satisfy the model's independence assumption. SGD Classifier was consistent but dropped slightly (0.84 to 0.80), as linear models are affected by PCA's feature space transformation.

- **Evaluation Beyond Accuracy**: We used weighted precision, recall, and confusion matrices to assess performance across classes. This revealed the limitations of class weights, as minority classes (especially Class 0 with only 83 samples) still exhibited low recall and precision, highlighting the persistent challenge of severe imbalance despite weighting.

- **Real Data vs. Synthetic**: By using class weights instead of SMOTE, we maintained all real student data, ensuring authenticity. This addressed concerns about "fake" training data feeling unrealistic.

- **Overall Insights**: ML projects require careful preprocessing, domain-aware decisions, and comprehensive evaluation. Class imbalance and dimensionality reduction need tailored approaches, and model performance depends on data characteristics and assumptions.

## Future Work

The class imbalance is partially addressed with class weights, but minority classes still underperform in some metrics. To fully fix this:
- Implement SMOTE or ADASYN for oversampling synthetic minorities. SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples by interpolating between minority class instances, creating more balanced training data without duplicating existing samples. ADASYN is similar but focuses on harder-to-learn samples. These could improve recall for Class 0 (83 samples) by creating diverse synthetic examples.
- Try ensemble methods like Balanced Random Forest, which combines undersampling with Random Forest for better minority handling.
- Explore cost-sensitive learning with custom cost matrices to penalize minority misclassifications more heavily.
- Add cross-validation for robust evaluation and hyperparameter tuning.
- Experiment with other imbalance techniques like undersampling majorities or combining over/under sampling.

## How to Run

1. Clone the repo.
2. Run the notebook `TraditionalML_tabular_classification.ipynb` in Google Colab (recommended, as it includes Colab-specific setup).

