# Credit Risk Prediction using Random Forest

This project demonstrates how to build a predictive model for classifying credit risk using the Random Forest algorithm. The analysis is conducted in R, and it includes various data preprocessing steps, model training, evaluation, and results documentation.

---

## Project Overview

The goal of this analysis is to predict the credit risk of individuals based on multiple features. The dataset includes both categorical and numerical variables. The Random Forest model is trained using 10-fold cross-validation, and the performance is evaluated using the AUC score from the ROC curve.

---

## Dataset

### Dataset File
- **File Name**: `credit_data.csv`

### Dataset Structure
- **Numeric Variables**: Age, income, credit score, etc.
- **Categorical Variables**: Occupation, credit risk, etc.
- **Target Variable**: `credit_risk` (binary classification: "good" or "bad")

---

## Steps Involved

### 1. Install and Load Necessary Packages
Install and load the required R packages:

\```r
install.packages(c("tidyverse", "caret", "DataExplorer", "corrplot", "randomForest", "DMwR", "pROC"))
library(tidyverse)
library(caret)
library(DataExplorer)
library(corrplot)
library(randomForest)
library(DMwR)
library(pROC)

\```
## 2. Load and Explore the Dataset
- The dataset is loaded using `read.csv()`.
- The dataset structure is explored using `glimpse()` and `summary()`.
- Missing values are visualized using the `plot_missing()` function from the `DataExplorer` package.
- Distribution of numeric variables is visualized with histograms.

---

## 3. Handle Missing Values
- Missing values in the `occupation` column are handled by:
  - Replacing non-standard missing values (e.g., `"NA"`, `"unknown"`) with `NA`.
  - Imputing missing values with the most frequent value.
- Missing values in the `age` column are replaced with the median age.
- Other categorical variables are imputed using their most frequent value.

---

## 4. Encode Categorical Variables
- Categorical variables are encoded into numeric format using the `model.matrix()` function.

---

## 5. Feature Scaling
- Numeric features are scaled using the `scale()` function to standardize the values.

---

## 6. Split the Dataset
- The dataset is split into training (80%) and testing (20%) subsets using `createDataPartition()`.

---

## 7. Handle Class Imbalance
- The class imbalance in the target variable (`credit_risk`) is addressed using **SMOTE** (Synthetic Minority Over-sampling Technique).
  - SMOTE generates synthetic samples for the minority class to balance the distribution of the target variable.

---

## 8. Train the Random Forest Model
- The Random Forest model is trained using 10-fold cross-validation (`trainControl()`) and evaluated using 5-fold tuning (`tuneLength = 5`).
- The model is trained to predict `credit_risk`.

---

## 9. Evaluate Model Performance
- Predictions are made on the test set using the trained model.
- A confusion matrix is generated to assess classification performance.
- **Feature importance** is visualized to identify which features contribute the most to the predictions.
- The **ROC curve** is plotted to visualize model performance, and the **AUC (Area Under the Curve)** score is computed.

---

## 10. Save Results
- **Predictions** are saved to a CSV file: `credit_risk_predictions_rf.csv`.
- **Documentation** detailing the steps, model performance, and execution time is saved to a text file: `model_workflow_documentation.txt`.
- A summary of execution time and AUC score is saved in `model_execution_summary.txt`.

---

## Results
- **AUC Score**: The AUC score is calculated using the ROC curve to measure the predictive performance of the model. A higher AUC value indicates better model performance.
- **Feature Importance**: The most influential features in predicting credit risk are identified.
- **Execution Time**: The total time taken for execution is logged.

---

## Conclusion
This project demonstrates the process of building a credit risk prediction model using Random Forest, including:
- Handling missing data.
- Addressing class imbalance.
- Evaluating model performance using the AUC score.

The generated predictions can be used to assess individuals' credit risk for further decision-making.

