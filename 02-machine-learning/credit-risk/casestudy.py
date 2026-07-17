import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'credit_data.csv'
data = pd.read_csv(file_path)

# Inspect the data
print(data.info())
print(data.describe())

# Step 2: Data Cleaning
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Visualize overall missing values with counts
plt.figure(figsize=(10, 6))
ax = missing_values[missing_values > 0].plot(kind='bar', color='skyblue')
plt.title("Missing Values per Column")
plt.xlabel("Columns")
plt.ylabel("Count of Missing Values")
plt.xticks(rotation=45, ha='right')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.tight_layout()
plt.show()

# Visualize target distribution before cleaning
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='credit_risk', data=data)
plt.title('Credit Risk Distribution (Before Cleaning)')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()

# Handle missing or invalid target values (-1 treated as missing)
data = data[data['credit_risk'] != -1]

# Visualize target distribution after cleaning
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='credit_risk', data=data)
plt.title('Credit Risk Distribution (After Cleaning)')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()

# Fill missing numerical values with the median
data.fillna(data.median(numeric_only=True), inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='credit_risk', data=data)
plt.title('Credit Risk Distribution')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()

# Convert categorical features to numeric for correlation analysis
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Correlation heatmap for numeric features
plt.figure(figsize=(12, 8))
ax = sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 4: Data Preprocessing
# Encoding categorical variables
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Separating features and target
X = data.drop('credit_risk', axis=1)
y = data['credit_risk']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Model Training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Evaluation
predictions = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
labels = sorted(np.unique(y))  # Ensure correct class labels sorted
plt.figure(figsize=(8, 6))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cmd.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# AUC-ROC for multi-class classification
if len(np.unique(y)) > 2: 
    y_pred_proba = clf.predict_proba(X_test)
    y_test_encoded = LabelEncoder().fit_transform(y_test)
    auc_score = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
    print("AUC-ROC Score (Multi-class, OVR):", auc_score)
else:
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print("AUC-ROC Score:", auc)

# Feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
ax = plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45, ha='right')
for i, rect in enumerate(ax):
    plt.text(rect.get_x() + rect.get_width() / 2., rect.get_height(), f'{importances[indices[i]]:.2f}',
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()
