import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

cells = []

# Title and Intro
cells.append(new_markdown_cell("""# Comparative Analysis of Machine Learning Models for Highly Imbalanced Credit Card Fraud Detection

## 1. Project Objective
This project builds and evaluates multiple machine learning models to classify credit card transactions as legitimate or fraudulent. The dataset is highly imbalanced (fraud makes up less than 0.2%).
We will focus on Recall and F1-score to evaluate the models' ability to minimize False Negatives.

## 2. Dataset Description
- **Source**: OpenML / Kaggle (Credit Card Fraud Detection Dataset)
- **Size**: 284,807 transactions, 492 frauds.
- **Features**: `Amount`, and 28 PCA-transformed features (`V1` to `V28`).
- **Target**: `Class` (0 = Legitimate, 1 = Fraudulent)."""))

# Setup
cells.append(new_markdown_cell("""## Imports & Setup"""))
cells.append(new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

# Set visualization style
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')"""))

# Data Acquisition
cells.append(new_markdown_cell("""## Data Acquisition
We fetch the dataset from OpenML. This provides the exact same Kaggle dataset without requiring API keys."""))
cells.append(new_code_cell("""# Fetching the dataset
print("Downloading dataset from OpenML...")
data = fetch_openml('creditcard', version=1, as_frame=True, parser='auto')
df = data.frame

print(f"Dataset shape: {df.shape}")
df.head()"""))

# Convert target to int (OpenML returns it as category or string sometimes)
cells.append(new_code_cell("""# Ensure Class column is integer type
if df['Class'].dtype != 'int64' and df['Class'].dtype != 'int32':
    df['Class'] = df['Class'].astype(int)
"""))

# Phase 1: EDA
cells.append(new_markdown_cell("""## Phase 1: Exploratory Data Analysis (EDA)"""))
cells.append(new_code_cell("""# Class distribution
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"Fraudulent ratio: {class_counts[1] / len(df) * 100:.3f}%")

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0: Legitimate, 1: Fraudulent)")
plt.show()"""))

cells.append(new_code_cell("""# Amount distribution
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

sns.histplot(df['Amount'], bins=50, color='r')
plt.title('Distribution of Transaction Amount')
plt.xlim([0, 2000]) # Zooming in as most amounts are small
plt.show()"""))

cells.append(new_code_cell("""# Check for missing values
print("Missing values in dataset:", df.isnull().sum().max())"""))

# Phase 2: Data Preprocessing
cells.append(new_markdown_cell("""## Phase 2: Data Preprocessing
- **Scaling**: We use `RobustScaler` on `Amount` because it is less prone to outliers than `StandardScaler`.
- **Splitting**: 80/20 stratified split.
- **Addressing Imbalance**: We will apply SMOTE to the training set only."""))
cells.append(new_code_cell("""# Scaling
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

df.drop(['Amount'], axis=1, inplace=True)

# Reordering columns
scaled_amount = df['scaled_amount']
df.drop(['scaled_amount'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)

df.head()"""))

cells.append(new_code_cell("""# Splitting
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)"""))

cells.append(new_code_cell("""# Applying SMOTE to training data
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f"Before SMOTE - Fraud: {sum(y_train==1)}, Legit: {sum(y_train==0)}")
print(f"After SMOTE - Fraud: {sum(y_train_sm==1)}, Legit: {sum(y_train_sm==0)}")"""))

# Phase 3: Model Implementation
cells.append(new_markdown_cell("""## Phase 3: Model Implementation
We train 12 models on the SMOTE-augmented dataset to conduct an extensive comparative analysis:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Linear SVM (using `LinearSVC` instead of standard `SVC` for performance on large datasets)
5. HistGradientBoosting (Fast native LightGBM alternative)
6. Extra Trees
7. AdaBoost
8. Gradient Boosting
9. Gaussian Naive Bayes
10. SGD Classifier (Stochastic Gradient Descent)
11. Ridge Classifier
12. Multi-Layer Perceptron (Neural Network)"""))
cells.append(new_code_cell("""# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), # using 50 trees for speed
    "Support Vector Machine": LinearSVC(random_state=42, dual=False),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "Gaussian NB": GaussianNB(),
    "SGD Classifier": SGDClassifier(random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "MLP (Neural Network)": MLPClassifier(max_iter=300, random_state=42)
}

# Train models
print("Training models... This might take a few minutes.")
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_sm, y_train_sm)
    trained_models[name] = model
print("Training complete!")"""))

# Phase 4: Evaluation & Comparative Analysis
cells.append(new_markdown_cell("""## Phase 4: Evaluation & Comparative Analysis
In addition to standard metrics like Precision, Recall, and F1-score, we are calculating advanced metrics robust to highly imbalanced datasets:
- **MCC (Matthews Correlation Coefficient)**: A balanced measure handling true and false positives and negatives.
- **ROC-AUC**: Evaluates discrimination ability.
- **Average Precision (AP)**: Summarizes the Precision-Recall curve, highly relevant for fraud detection."""))
cells.append(new_code_cell("""# Evaluation dictionary
results = {}

fig, axes = plt.subplots(4, 3, figsize=(16, 20))
axes = axes.flatten()

for i, (name, model) in enumerate(trained_models.items()):
    # Predict on unseen test set
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    
    results[name] = [acc, prec, rec, f1, mcc, roc_auc, ap]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    axes[i].set_title(f"{name} Confusion Matrix")
    axes[i].set_ylabel('True Label')
    axes[i].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()"""))

cells.append(new_code_cell("""# Results Comparison
results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC', 'Avg Precision']).T
print(results_df.sort_values(by='Avg Precision', ascending=False).round(4))"""))

cells.append(new_markdown_cell("""### Visualizing ROC and Precision-Recall Curves"""))
cells.append(new_code_cell("""# Plot ROC and PR Curves for all models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

for name, model in trained_models.items():
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
        
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # PR Curve
    precisions, recalls, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    ax2.plot(recalls, precisions, label=f'{name} (AP = {ap:.3f})')

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
ax1.legend(loc='lower right', fontsize='small')

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc='lower left', fontsize='small')

plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### Feature Importance Visualization
We display the feature importances (or absolute coefficients for linear models) for the top 10 features of each model. Note that some models (like HistGradientBoosting, GaussianNB, and MLP) do not provide native feature importances without expensive permutation techniques."""))
cells.append(new_code_cell("""# Extract feature importances for all models
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.flatten()
feature_names = X.columns

for i, (name, model) in enumerate(trained_models.items()):
    ax = axes[i]
    importances = None
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
        
    if importances is not None:
        indices = np.argsort(importances)[::-1][:10] # Top 10
        ax.bar(range(10), importances[indices], align="center")
        ax.set_xticks(range(10))
        ax.set_xticklabels(feature_names[indices], rotation=45)
        ax.set_title(f"{name}")
        ax.set_xlim([-1, 10])
    else:
        ax.text(0.5, 0.5, "Not Natively Supported", ha='center', va='center', fontsize=12)
        ax.set_title(f"{name}")
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()"""))

cells.append(new_code_cell("""# Detailed Classification Report for each model
for name, model in trained_models.items():
    print(f"\\n{'='*50}")
    print(f"Classification Report for {name}:")
    print(f"{'='*50}\\n")
    print(classification_report(y_test, model.predict(X_test)))"""))

nb['cells'] = cells

with open('credit_card_fraud_detection.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Notebook created successfully!")
