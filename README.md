# ICR-age-related-conditions

This code implements a machine learning pipeline for an age-related conditions identification competition. The code is divided into several sections:

Imports:

Libraries: Pandas, NumPy, XGBoost, LightGBM, Scikit-learn libraries for data manipulation, feature engineering, model training, and evaluation.
Functions: Custom functions for competition-specific evaluation metric (log loss), loss weights, best threshold selection, and probability scaling.
Parameters: Define paths, number of folds, and other parameters used throughout the code.
Data Preprocessing:

Load data: Read train and test datasets.
Feature engineering: Preprocess categorical features, merge with Greeks dataset, and handle missing values.
Feature transformation: Standardize numeric features and apply dimensionality reduction techniques (PCA, LDA, etc.) for various models.
Model Training:

Looped training: Train models for each fold using StratifiedKFold with loss weights.
Model options: Train Logistic Regression, Random Forest, Support Vector Machine, KNN, XGBoost, LightGBM, TabPFNClassifier, MLP, Decision Tree, and AdaBoost with Random Forest.
Prediction: Make predictions on the test set for each model.
Evaluation and Blending:

Log loss calculation: Calculate competition-specific log loss metric for each model's predictions.
Threshold selection: Find the optimal threshold for each model's predictions based on the log loss metric.
Blending: Blend predictions from multiple models using weighted average.
Submission Generation:

Create a submission file with class probabilities for the test set using the blended model.
Key Libraries and Functions:

Scikit-learn: Provides various machine learning models and evaluation metrics.
XGBoost and LightGBM: Gradient boosting models often used for leaderboard performance.
TabPFNClassifier: Tabular Pre-trained Feature Network Classifier for tabular data.
Iterative Stratification: Multilabel Stratified KFold for improved fold creation.
Competition-specific functions: Calculate log loss with weighted classes and best threshold selection.
Note: This readme provides a high-level overview of the code. For detailed information, refer to the code comments and the resources listed within the code.

Resources:

Kaggle competition: https://www.kaggle.com/competitions/icr-identify-age-related-conditions
Iterative Stratification: https://github.com/trent-b/iterative-stratification
TabPFNClassifier: https://github.com/automl/TabPFN
This code demonstrates a well-structured approach to machine learning competition participation, including data preprocessing, feature engineering, model training, evaluation, blending, and submission generation. By experimenting with various models and blending techniques, you can optimize your performance and achieve better results.
