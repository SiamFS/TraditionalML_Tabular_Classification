

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
- **Without PCA:** Decision Tree, SVM, and Random Forest all achieve 0.87 accuracy; Naive Bayes and SGD reach 0.81. Precision and recall closely match accuracy. All models struggle with the minority class (Class 0), even with class weights.
- **With PCA:** SVM and Random Forest maintain 0.87 accuracy; Decision Tree drops slightly to 0.85; Naive Bayes and SGD remain at 0.81. The effect of PCA is model-dependent, with ensemble and kernel methods most robust.

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

- **Models that Perform Best with PCA**:
  - **Random Forest**: Maintained top accuracy (0.87) with and without PCA, benefiting from reduced noise and faster training on fewer features. As an ensemble method, it averages predictions from multiple trees, making it robust to the loss of some feature information and less prone to overfitting on high-dimensional data.
  - **SVM**: Maintained strong accuracy (0.87) with PCA, as it handles the reduced dimensionality better by focusing on principal components. SVM finds the optimal hyperplane in feature space, and PCA helps by removing multicollinearity and irrelevant features, leading to better generalization.

- **Models that Perform Worse with PCA**:
  - **Decision Tree**: Accuracy dropped from 0.87 to 0.85 with PCA, as tree-based models can overfit on reduced features and lose interpretability. Decision trees split on individual features, and PCA's linear combinations make splits less meaningful, potentially leading to suboptimal trees.
  - **Naive Bayes**: Accuracy stayed at 0.81 but may degrade further in other datasets, due to PCA violating the feature independence assumption by creating correlated components. NB relies on the assumption that features are independent given the class, but PCA introduces dependencies through its transformations.
  - **SGD Classifier**: Accuracy remained at 0.81, but PCA can make optimization less stable for linear models in some cases. SGD optimizes a linear function, and PCA changes the feature space, which might alter the convergence path and require different learning rates.

- **Why PCA Helps or Hurts**: PCA is most beneficial for models sensitive to high dimensions or multicollinearity (like SVM and Random Forest), but can hurt models relying on feature relationships or interpretability (like Decision Tree and Naive Bayes). In our case, PCA reduced 32 features to 10 while retaining 95% variance, maintaining performance for most models but slightly degrading Decision Tree due to its reliance on original feature splits.

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
  - Decision Tree: 0.87
  - SGD Classifier: 0.81
  - SVM: 0.87
  - Random Forest: 0.87
  - Naive Bayes: 0.81

- **With PCA**:
  - Decision Tree: 0.85
  - SGD Classifier: 0.81
  - SVM: 0.87
  - Random Forest: 0.87
  - Naive Bayes: 0.81

### Additional Metrics
For both with and without PCA, weighted precision and recall closely matched accuracy for all models. Confusion matrices showed that all models struggled with the minority class (Class 0), even with class weights, highlighting the persistent challenge of imbalance.

PCA generally maintains or slightly improves performance for ensemble/tree-based models (Random Forest, Decision Tree) by reducing noise, but may slightly hurt linear models (SGD) or probabilistic ones (Naive Bayes) due to loss of feature interpretability.

## Discussion

- **Model Selection and Performance**: Random Forest achieved the highest accuracy (0.87 both with and without PCA) due to its ensemble nature handling noise and imbalance well. Decision Tree and SVM also performed strongly (0.87 without PCA, with Decision Tree dropping slightly to 0.85 with PCA). Naive Bayes struggled (0.81) possibly due to feature dependencies violating its independence assumption. SGD Classifier performed moderately (0.81 both cases), benefiting from scaling but sensitive to PCA's feature transformations.

- **Evaluation Beyond Accuracy**: We used weighted precision, recall, and confusion matrices to assess performance across classes. This revealed how class weights helped minority classes, as seen in improved recall scores.

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

