# AI Project: Evaluation of Diabetic Classification Models

This repository contains the code and analysis for an Artificial Intelligence course project (LM-66 Cybersecurity and Cloud Platforms) focused on evaluating machine learning models for diabetes classification.

## Project Overview

The project utilizes the "Diabetes Dataset" to classify patients as diabetic or non-diabetic. The primary goal is to:

* Implement and compare Logistic Regression, Decision Tree, and Random Forest models. 
   
* Evaluate performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC. 
   
* Perform hyperparameter tuning using RandomizedSearchCV.
   
* Analyze feature importance to identify key factors influencing diabetes prediction.

The implementation is in Python, leveraging libraries including scikit-learn, Pandas, NumPy, Seaborn, and Matplotlib.This work demonstrates the application of AI techniques to a healthcare problem, relevant to data analysis and predictive modeling.

## Key Objectives

* **Accuracy:** To develop models that accurately classify patients with and without diabetes. 
   
* **Robustness:** To ensure the models perform well under various conditions and with real-world data. 
   
* **Clinical Relevance:** To minimize false negatives to avoid misdiagnosis of diabetic patients. 

## Dataset

The "Diabetes Dataset" from Kaggle was used.It contains diagnostic information for Pima Indian women aged 21 and older. 

### Dataset Characteristics

* 768 instances
   
* 9 columns (8 features, 1 target variable)
   
* Features include: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
* Target variable: Outcome (diabetic or non-diabetic).

## Workflow

1.  **Data Preprocessing:**
   
    * Exploratory data analysis. 
       
    * Outlier removal using the Interquartile Range (IQR) method. 
       
    * Feature engineering, including age categorization. 
    * Data normalization. 
       
2.  **Dataset Splitting:**
   
    * 75% training set, 25% test set.
  Okay, continuing the detailed README:

---

3.  **Model Implementation:**
   
    * Logistic Regression: Implemented with L1 and L2 regularization.
       
    * Decision Tree: Tuned for optimal depth.
       
    * Random Forest: Ensemble of decision trees.
       
4.  **Hyperparameter Tuning:**
   
    * RandomizedSearchCV was used to optimize hyperparameters for each model.
       
5.  **Model Evaluation:**
   
    * Evaluation on the test set.
       
    * Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix, AUC-ROC curve.
       
6.  **Feature Importance:**
   
    * Analysis of feature importance for Random Forest and Logistic Regression to understand the influence of each variable on the prediction.

## Models

### Logistic Regression

* **Description:** A linear model for classification.
   
* **Hyperparameters Tuned:** Regularization type (L1, L2), C (inverse of regularization strength).
   
* **Regularization:** L1 and L2 regularization were applied to prevent overfitting.

### Decision Tree

* **Description:** A tree-like model that makes decisions based on features.
   
* **Hyperparameters Tuned:** Max depth of the tree.
   
* **Depth Control:** Limiting the depth helps to avoid overfitting.

### Random Forest

* **Description:** An ensemble of decision trees that improves generalization.
   
* **Hyperparameters Tuned:** Number of trees in the forest, max depth of trees, min samples split.
   
* **Ensemble Method:** Combining multiple trees reduces variance and improves robustness.

## Results
For more details, you can consult the document “Evaluation of diabetic metrics” found in the Github repo

## Conclusion

This project successfully implemented and evaluated machine learning models for diabetes classification. The results demonstrate the potential of AI in healthcare for early detection and diagnosis support.

## Files

* `main.py`: Main script to run the analysis.
   
* `DecisionTree.py`: Contains the model implementation of Decision Tree.

* `LogisticRegression.py`: Contains the model implementation of Logistic Regression.

* `RandomForest.py`: Contains the model implementation of Random Forest.
   
* `diabates.csv`: File for the dataset.
  

## Dependencies

* Python 3.12
   
* scikit-learn
   
* Pandas
   
* NumPy
   
* Matplotlib
   
* Seaborn

## Author(s)

* Francesco Alfonso Barlotti
   
* Salvatore Colucci
   
* Domenico Falco

## Acknowledgements

* Prof.ssa Loredana Caruccio
   
* Prof.ssa Genoveffa Tortora
   
* University of Salerno-Master's Degree in Computer Science LM-66 Cybersecurity and Cloud Platforms

