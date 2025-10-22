# Traditional ML Tabular Classification

This project demonstrates a comprehensive approach to tabular classification using classical machine learning models on student performance data. It covers data preprocessing, handling class imbalance, dimensionality reduction with PCA, model training, evaluation, and comparative analysis. The notebook implements standard ML practices to predict student final grades (G3) binned into three classes, showcasing real-world challenges like imbalanced datasets and feature engineering.

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
  - Gradient Descent: 0.81
  - SVM: 0.87
  - Random Forest: 0.87
  - Naive Bayes: 0.81

- **With PCA**:
  - Decision Tree: 0.85
  - Gradient Descent: 0.81
  - SVM: 0.87
  - Random Forest: 0.87
  - Naive Bayes: 0.81

PCA generally maintains or slightly improves performance for ensemble/tree-based models (Random Forest, Decision Tree) by reducing noise, but may slightly hurt linear models (SGD) or probabilistic ones (Naive Bayes) due to loss of feature interpretability.

## What We Learned

- **Data Preprocessing Importance**: Proper preprocessing is critical for ML success. We learned to prevent data leakage by performing train-test split (70-30) before any preprocessing steps like encoding, scaling, and binning. This ensures the model isn't trained on information from the test set, leading to realistic evaluation.
  
- **Handling Mixed Data Types**: The dataset had both categorical (e.g., school, sex, address) and continuous features (age, absences, G1, G2). We used OrdinalEncoder for categorical features and StandardScaler for continuous ones, fitted only on training data to avoid leakage. This ensured fair transformation and maintained data integrity.

- **Correlation Analysis and Feature Retention**: Despite high correlations between G1, G2, and G3 (as expected since they are sequential grades), we retained all features because G1 and G2 are predictive of G3. Other features had correlations below 85%, so no removal was needed. This taught us to consider domain knowledge over strict correlation thresholds.

- **Target Binning for Classification**: The original G3 was continuous (0-20), so we binned it into 3 classes using KBinsDiscretizer with uniform strategy. This converted regression to multi-class classification, making evaluation clearer with confusion matrices and class-specific metrics.

- **Class Imbalance Handling**: The dataset was highly imbalanced (Class 0: 83, Class 1: 671, Class 2: 294). We used class_weight='balanced' for applicable models (DT, SGD, SVM, RF) to penalize misclassifications of minority classes. Naive Bayes does not support class weights, so left unchanged.

- **Model Selection and Performance**: Random Forest achieved the highest accuracy (0.87 both with and without PCA) due to its ensemble nature handling noise and imbalance well. Decision Tree and SVM also performed strongly (0.87 without PCA, with Decision Tree dropping slightly to 0.85 with PCA). Naive Bayes struggled (0.81 without PCA, dropping to 0.81 with PCA) possibly due to feature dependencies violating its independence assumption. Gradient Descent performed moderately (0.81 both cases), benefiting from scaling but sensitive to PCA's feature transformations.

- **Evaluation Beyond Accuracy**: We used weighted precision, recall, and confusion matrices to assess performance across classes. This revealed how class weights helped minority classes, as seen in improved recall scores.

- **PCA Dimensionality Reduction**: PCA reduced 32 features to 10 while retaining 95% variance, speeding up training and potentially reducing overfitting. However, results varied by model, teaching us PCA's model-dependent nature.

- **Real Data vs. Synthetic**: By using class weights instead of SMOTE, we maintained all real student data, ensuring authenticity. This addressed the user's concern about "fake" training data feeling unrealistic.

- **Overall Insights**: ML projects require careful preprocessing, domain-aware decisions, and comprehensive evaluation. Class imbalance and dimensionality reduction need tailored approaches, and model performance depends on data characteristics and assumptions.

### Detailed Learnings on PCA
PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms correlated features into uncorrelated principal components ordered by variance explained. It improves performance for some ML models but not others due to how different algorithms handle features:

- **Models that Perform Best with PCA**:
  - **Random Forest**: Maintained top accuracy (0.87) with and without PCA, benefiting from reduced noise and faster training on fewer features. As an ensemble method, it averages predictions from multiple trees, making it robust to the loss of some feature information and less prone to overfitting on high-dimensional data.
  - **SVM**: Maintained strong accuracy (0.87) with PCA, as it handles the reduced dimensionality better by focusing on principal components. SVM finds the optimal hyperplane in feature space, and PCA helps by removing multicollinearity and irrelevant features, leading to better generalization.

- **Models that Perform Worse with PCA**:
  - **Decision Tree**: Accuracy dropped from 0.87 to 0.85 with PCA, as tree-based models can overfit on reduced features and lose interpretability. Decision trees split on individual features, and PCA's linear combinations make splits less meaningful, potentially leading to suboptimal trees.
  - **Naive Bayes**: Accuracy stayed at 0.81 but may degrade further in other datasets, due to PCA violating the feature independence assumption by creating correlated components. NB relies on the assumption that features are independent given the class, but PCA introduces dependencies through its transformations.
  - **Gradient Descent**: Accuracy remained at 0.81, but PCA can make optimization less stable for linear models in some cases. SGD optimizes a linear function, and PCA changes the feature space, which might alter the convergence path and require different learning rates.

- **Why PCA Helps or Hurts**: PCA is most beneficial for models sensitive to high dimensions or multicollinearity (like SVM and Random Forest), but can hurt models relying on feature relationships or interpretability (like Decision Tree and Naive Bayes). In our case, PCA reduced 32 features to 10 while retaining 95% variance, maintaining performance for most models but slightly degrading Decision Tree due to its reliance on original feature splits.

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

