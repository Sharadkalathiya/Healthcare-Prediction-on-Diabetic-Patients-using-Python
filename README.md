# Healthcare Prediction on Diabetic Patients Using Python

This project focuses on building a predictive model to determine the likelihood of diabetes in patients based on medical and demographic data. The primary dataset is sourced from the National Institute of Diabetes and Digestive and Kidney Diseases, specifically targeting female patients aged 21 or older and of Pima Indian heritage.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Workflow](#workflow)
4. [Modeling Techniques](#modeling-techniques)
5. [Results and Findings](#results-and-findings)
6. [Future Work](#future-work)
7. [Installation and Usage](#installation-and-usage)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction

The project aims to provide a reliable prediction of diabetes presence based on features such as glucose levels, blood pressure, insulin levels, and BMI. The dataset used exhibits certain challenges, including missing values, outliers, and class imbalance, which have been systematically addressed throughout the project.

## Dataset Description

The dataset includes 768 observations with the following attributes:

- **Predictor Variables:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- **Target Variable:**
  - Outcome (1: Diabetes, 0: No Diabetes)

### Key Observations
- Significant missing values in features like Insulin and Skin Thickness.
- Presence of outliers, particularly in the Insulin column.
- Class imbalance in the target variable, with a higher prevalence of non-diabetic cases.

---

## Workflow

The project workflow consists of the following steps:

1. **Data Exploration:**
   - Summary statistics and visual exploration.
   - Identification and handling of missing values.
2. **Data Cleaning:**
   - Imputation of missing values with feature-specific means.
   - Outlier detection using boxplots and the IQR method.
3. **Exploratory Data Analysis (EDA):**
   - Univariate, bivariate, and multivariate analyses using histograms, scatterplots, violin plots, and heatmaps.
4. **Data Balancing:**
   - Addressing class imbalance using techniques like SMOTE or resampling.
5. **Feature Engineering:**
   - Scaling and standardization using `StandardScaler` and `MinMaxScaler`.
   - Feature selection using statistical methods like `SelectKBest` and `chi2`.
6. **Model Building:**
   - Implementing machine learning algorithms such as:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - Naive Bayes
     - K-Nearest Neighbors (KNN)
   - Hyperparameter tuning using `GridSearchCV`.
7. **Model Evaluation:**
   - Metrics used: Accuracy, Precision, Recall, Confusion Matrix, ROC Curve, and AUC.

---

## Modeling Techniques

The project employed a variety of machine learning models to find the best-performing model for diabetes prediction. Techniques included:

- **Logistic Regression:** A baseline model for binary classification.
- **Decision Trees & Random Forests:** For feature importance analysis and robust predictions.
- **Support Vector Machines (SVM):** For separating data points with a non-linear decision boundary.
- **K-Nearest Neighbors (KNN):** For instance-based learning.

---

## Results and Findings

- **Key Insights:**
  - Glucose and BMI are the most significant predictors of diabetes.
  - Age and pregnancies showed strong relationships, indicating potential indirect effects.
- **Model Performance:**
  - Random Forest provided the highest accuracy and robustness against class imbalance.
  - Logistic Regression and SVM performed well but were sensitive to class imbalance.

---

## Future Work

1. **Incorporate Additional Data:**
   - Include more demographic and health-related features for better predictions.
2. **Deep Learning Models:**
   - Experiment with neural networks for improved performance on large datasets.
3. **Deploy as an Application:**
   - Create a web or mobile application for real-time diabetes risk assessment.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Sharadkalathiya/Healthcare-Prediction-on-Diabetic-Patients-using-Python
   cd Healthcare-Prediction-on-Diabetic-Patients-using-Python
   ```
2. Install the required packages:

3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook
   ```
4. Explore the data, preprocessing steps, and models included in the project.

---

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository, make changes, and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

--- 

For more details, visit the [GitHub Repository](https://github.com/Sharadkalathiya/Healthcare-Prediction-on-Diabetic-Patients-using-Python).
