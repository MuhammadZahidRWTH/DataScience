# Diabetes Classification Supervised Learning

This project performs diabetes classification using multiple supervised learning algorithms. The goal is to predict whether a patient has diabetes based on various diagnostic factors. The project utilizes several classification models, such as Random Forest, Gradient Boosting, Logistic Regression, and others, to evaluate and compare their performance

## <span style="color:blue">Table of Contents</span>

- [Overview](#overview)
- [Classifiers Used](#classifiers-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Evaluation](#model-evaluation)
  - [ROC AUC Score](#roc-auc-score)
  - [Confusion Matrix](#confusion-matrix)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Results Documentation](#results-documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains a diabetes classification project that compares various classifiers using multiple random seeds for reproducibility. The classification models are evaluated on the following metrics:

- ROC AUC Score
- Confusion Matrix
- Classification Report

The following classifiers are used in the project:

- Random Forest
- Gradient Boosting
- AdaBoost
- Logistic Regression
- Support Vector Machine (SVC)

The project leverages **SMOTE** for balancing the dataset and applies polynomial feature engineering. Results are stored in Word documents and visualized through ROC curves and confusion matrices.

## Classifiers Used

### Random Forest
A versatile ensemble classifier based on decision trees that is known for its robustness and high performance.

### Gradient Boosting
A machine learning model that builds an additive model in a forward stage-wise fashion, optimizing for any differentiable loss function.

### AdaBoost
A boosting technique that adjusts the weights of classifiers based on the accuracy of the previous classifiers to improve performance.

### Logistic Regression
A linear classifier used for binary classification tasks, which outputs probabilities using the logistic function.

### Support Vector Classifier (SVC)
A powerful classifier that finds the optimal hyperplane to separate the classes by maximizing the margin between them.

## Data Preprocessing

The project uses the **diabetes.csv** dataset, which contains features that can be used to predict whether a patient has diabetes. The preprocessing steps are as follows:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to balance the dataset by generating synthetic samples.
2. **Polynomial Features**: Interaction-only polynomial features are generated to capture higher-order feature relationships.
3. **Standard Scaling**: The features are scaled to normalize the dataset for model training.

## Model Evaluation

### ROC AUC Score
The **ROC AUC score** is used to evaluate the classifier's ability to distinguish between positive and negative classes. A higher ROC AUC score indicates better performance.

### Confusion Matrix
The **confusion matrix** is used to summarize the performance of the classification model by displaying the True Positives, True Negatives, False Positives, and False Negatives.

### Classification Report
A detailed **classification report** provides precision, recall, f1-score, and support for each class.

## Installation

To run this project locally, follow the steps below:
1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadZahidRWTH/Diabetes-Classification.git


## Installation

To run this project locally, follow the steps below:
1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadZahidRWTH/Diabetes-Classification.git
   ```
2. Navigate to the project folder:
   ```bash
   cd Diabetes-Classification
   ```
3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not provided, you can manually install the necessary packages:
   ```bash
   pip install numpy pandas scikit-learn imbalanced-learn matplotlib docx
   ```
4. Download or prepare the **diabetes.csv** dataset and place it in the root of the project directory.

## Usage

Once the dependencies are installed, and the dataset is placed correctly, you can run the project script. The script will train the models on the dataset and generate performance reports.

Run the script:
```bash
python diabetes_classification.py
```

### Script Behavior:
- For each classifier, the results will be saved in a folder named after the classifier. The results include:
  - Classification report
  - ROC curve image
  - Confusion matrix image
- An overall summary document will be generated at the end with links to individual reports for each classifier and seed.

The models will use three random seeds: 42, 123, and 456, to ensure reproducibility of the results. The output will include detailed analysis for each classifier, which is stored in a separate Word document for each seed and model combination.

## Output

After running the script, the following output will be generated:
1. **Word Documents**: For each classifier and seed, a Word document will be generated with the following sections:
  - **Classification Report**: Precision, recall, f1-score, and support for each class.
  - **ROC AUC Score**: A numeric score showing the performance of the model.
  - **ROC Curve**: A plot showing the trade-off between the true positive rate and the false positive rate.
  - **Confusion Matrix**: A matrix summarizing the number of true positives, false positives, false negatives, and true negatives.
  - **Confusion Matrix Plot**: A visual representation of the confusion matrix.

2. **Overall Results Document**: A document consolidating results from all classifiers and seeds, containing the following:
  - Summary of the ROC AUC Score for each classifier and seed.
  - Links to the individual reports for each classifier.

## Results Documentation

The results for each classifier are saved in separate **Word documents**. Each document includes:
1. **Classification Report**: Precision, recall, F1-score, and support for each class.
2. **ROC AUC Score**: A numeric score showing the performance of the model.
3. **ROC Curve**: A visual plot of the ROC curve.
4. **Confusion Matrix**: A matrix summarizing the number of true positives, false positives, false negatives, and true negatives.
5. **Confusion Matrix Plot**: A visual representation of the confusion matrix.

An overall summary document is also created that consolidates the results from all classifiers and seeds.

## Contributing

If you'd like to contribute to the project, feel free to fork the repository and submit a pull request. Contributions could include:
- Bug fixes
- New features
- Improved documentation
- Model enhancements

### Steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to your forked repository (`git push origin feature-name`).
5. Open a pull request to the `main` branch.

## License

This project is open-source and available under the [MIT License](LICENSE).

