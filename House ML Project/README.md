# House Prices: Advanced Regression Techniques

This repository focuses on **Exploratory Data Analysis (EDA)** of the **House Prices** dataset, followed by **feature engineering**, **model building**, and **evaluation** using **stacked regression models**

## Table of Contents

### Part 1: Exploratory Data Analysis (EDA)
1. EDA Overview
2. Personal Feature Selection and Understanding
3. Exploring the Target Variable: `SalePrice`
    - 3.1 Relationship with Numerical Variables
    - 3.2 Relationship with Categorical Variables
4. Analysis of Other Variables
    - 4.1 Correlation Matrix Heatmap
5. Missing Data
6. Considering the Effect of Outliers
    - 6.1 Univariate Analysis
    - 6.2 Bivariate Analysis
7. Target Assumptions
    - 7.1 Normality
    - 7.2 Homoscedasticity
    - 7.3 Skewed Features
    - 7.4 Temporal Variables
8. Types of Numerical Variables
    - 8.1 Discrete Numerical Values
    - 8.2 Continuous Numerical Values
        - 8.2.1 Logarithmic Transformation of Data
9. Categorical Variables

---

### Part 2: Feature Engineering and Model Building

1. Combining Train and Test Data for Transformations
2. Feature Engineering
3. Outlier Removal
4. Log-Transformation of Target Variable
5. Treating Missing Values
    - 5.1 Numerical Features
    - 5.2 Categorical Features
6. Date and Time Feature Engineering
7. Variable Transformation
    - 7.1 Transforming Numerical Variables that are Categorical
    - 7.2 Label Encoding Categorical Features
    - 7.3 Transforming Skewed Features
8. Feature Selection
    - 8.1 Using Lasso
    - 8.2 Using Tree-Based Models
9. Model Building
10. Stacking Regressor
11. Final Run and Submission

---

### Part 1: Exploratory Data Analysis (EDA)

#### 1. EDA Overview
This section includes a detailed overview of the dataset, including data types, missing values, outliers, and relationships between the target variable and other features.

#### 2. Personal Feature Selection and Understanding
In this section, we select key features that are important for predicting house prices based on domain knowledge and data exploration.

#### 3. Exploring the Target Variable: `SalePrice`
- **3.1 Relationship with Numerical Variables**: Analyzing how the target variable correlates with continuous numerical features.
- **3.2 Relationship with Categorical Variables**: Exploring how `SalePrice` varies with different categorical variables.

#### 4. Analysis of Other Variables
- **4.1 Correlation Matrix Heatmap**: Visualizing the correlations between numerical features using a heatmap to understand relationships and potential multicollinearity.

#### 5. Missing Data
Handling missing data by identifying which features have missing values and exploring potential imputation techniques.

#### 6. Considering the Effect of Outliers
- **6.1 Univariate Analysis**: Analyzing the distribution of individual features to detect outliers.
- **6.2 Bivariate Analysis**: Exploring the relationship between features and identifying any outliers that might affect the model.

#### 7. Target Assumptions
Before applying regression techniques, we check the assumptions about the target variable and its distribution:
- **7.1 Normality**: Checking if the target variable is normally distributed.
- **7.2 Homoscedasticity**: Verifying that residuals show constant variance.
- **7.3 Skewed Features**: Identifying and handling skewed features.
- **7.4 Temporal Variables**: Understanding how time-related features affect house prices.

#### 8. Types of Numerical Variables
- **8.1 Discrete Numerical Values**: Features that take on a finite set of distinct values.
- **8.2 Continuous Numerical Values**: Features that take on a continuous range of values.
    - **8.2.1 Logarithmic Transformation of Data**: Applying log transformations to skewed numerical data.

#### 9. Categorical Variables
Exploring and preprocessing categorical variables for inclusion in the regression models.

---

### Part 2: Feature Engineering and Model Building

#### 1. Combining Train and Test Data for Transformations
Merging the train and test datasets for consistent transformation application.

#### 2. Feature Engineering
Creating new features based on the existing dataset, such as polynomial features, interactions, and domain-specific variables.

#### 3. Outlier Removal
Identifying and removing outliers that may skew the results of the model.

#### 4. Log-Transformation of Target Variable
Log-transforming the target variable to improve model performance.

#### 5. Treating Missing Values
- **5.1 Numerical Features**: Imputing missing values for continuous numerical features.
- **5.2 Categorical Features**: Imputing missing values for categorical features.

#### 6. Date and Time Feature Engineering
Handling features related to dates and times, and extracting useful components like year, month, and day.

#### 7. Variable Transformation
- **7.1 Transforming Numerical Variables that are Categorical**: Changing continuous variables to categorical when needed.
- **7.2 Label Encoding Categorical Features**: Converting categorical variables into numerical format using label encoding.
- **7.3 Transforming Skewed Features**: Applying transformations like square root or logarithmic transformations to address skewed data.

#### 8. Feature Selection
- **8.1 Using Lasso**: Applying Lasso regression for feature selection to eliminate irrelevant features.
- **8.2 Using Tree-Based Models**: Using tree-based models (like Random Forest) to select the most important features.

#### 9. Model Building
Building different regression models and evaluating their performance, such as Linear Regression, Ridge, and Lasso Regression.

#### 10. Stacking Regressor
Implementing stacking models that combine multiple individual models to improve performance.

#### 11. Final Run and Submission
Running the final model and preparing the submission file for Kaggle or other platforms.

---

### Installation and Setup
To use this project, you will need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `statsmodels`

You can install these libraries by running:

```bash
pip install -r requirements.txt
