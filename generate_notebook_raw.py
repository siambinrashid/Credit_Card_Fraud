import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Comparative Analysis of Machine Learning Models for Highly Imbalanced Credit Card Fraud Detection\n",
    "\n",
    "## 1. Project Objective\n",
    "This project builds and evaluates multiple machine learning models to classify credit card transactions as legitimate or fraudulent. The dataset is highly imbalanced (fraud makes up less than 0.2%).\n",
    "We will focus on Recall and F1-score to evaluate the models' ability to minimize False Negatives.\n",
    "\n",
    "## 2. Dataset Description\n",
    "- **Source**: OpenML / Kaggle (Credit Card Fraud Detection Dataset)\n",
    "- **Size**: 284,807 transactions, 492 frauds.\n",
    "- **Features**: `Time`, `Amount`, and 28 PCA-transformed features (`V1` to `V28`).\n",
    "- **Target**: `Class` (0 = Legitimate, 1 = Fraudulent)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Imports & Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.datasets import fetch_openml\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import RobustScaler\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import LinearSVC\n",
    "\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "# Set visualization style\n",
    "sns.set_style('whitegrid')\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Acquisition\n",
    "We fetch the dataset from OpenML. This provides the exact same Kaggle dataset without requiring API keys."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fetching the dataset\n",
    "print(\"Downloading dataset from OpenML...\")\n",
    "data = fetch_openml('creditcard', version=1, as_frame=True, parser='auto')\n",
    "df = data.frame\n",
    "\n",
    "print(f\"Dataset shape: {df.shape}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ensure Class column is integer type\n",
    "if df['Class'].dtype != 'int64' and df['Class'].dtype != 'int32':\n",
    "    df['Class'] = df['Class'].astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Phase 1: Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Class distribution\n",
    "class_counts = df['Class'].value_counts()\n",
    "print(class_counts)\n",
    "print(f\"Fraudulent ratio: {class_counts[1] / len(df) * 100:.3f}%\")\n",
    "\n",
    "plt.figure(figsize=(6,4))\n",
    "sns.countplot(x='Class', data=df)\n",
    "plt.title(\"Class Distribution (0: Legitimate, 1: Fraudulent)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Amount distribution\n",
    "fig, ax = plt.subplots(1, 1, figsize=(7, 4))\n",
    "\n",
    "sns.histplot(df['Amount'], bins=50, color='r')\n",
    "plt.title('Distribution of Transaction Amount')\n",
    "plt.xlim([0, 2000]) # Zooming in as most amounts are small\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values\n",
    "print(\"Missing values in dataset:\", df.isnull().sum().max())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Phase 2: Data Preprocessing\n",
    "- **Scaling**: We use `RobustScaler` on `Amount` because it is less prone to outliers than `StandardScaler`.\n",
    "- **Splitting**: 80/20 stratified split.\n",
    "- **Addressing Imbalance**: We will apply SMOTE to the training set only."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scaling\n",
    "scaler = RobustScaler()\n",
    "df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))\n",
    "\n",
    "df.drop(['Amount'], axis=1, inplace=True)\n",
    "\n",
    "# Reordering columns\n",
    "scaled_amount = df['scaled_amount']\n",
    "df.drop(['scaled_amount'], axis=1, inplace=True)\n",
    "df.insert(0, 'scaled_amount', scaled_amount)\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting\n",
    "X = df.drop('Class', axis=1)\n",
    "y = df['Class']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "print(\"Training set shape:\", X_train.shape)\n",
    "print(\"Testing set shape:\", X_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Applying SMOTE to training data\n",
    "sm = SMOTE(random_state=42)\n",
    "X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)\n",
    "\n",
    "print(f\"Before SMOTE - Fraud: {sum(y_train==1)}, Legit: {sum(y_train==0)}\")\n",
    "print(f\"After SMOTE - Fraud: {sum(y_train_sm==1)}, Legit: {sum(y_train_sm==0)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Phase 3: Model Implementation\n",
    "We train 4 models on the SMOTE-augmented dataset:\n",
    "1. Logistic Regression\n",
    "2. Decision Tree\n",
    "3. Random Forest\n",
    "4. Linear SVM (using `LinearSVC` instead of standard `SVC` for performance on large datasets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize models\n",
    "models = {\n",
    "    \"Logistic Regression\": LogisticRegression(max_iter=1000, random_state=42),\n",
    "    \"Decision Tree\": DecisionTreeClassifier(random_state=42),\n",
    "    \"Random Forest\": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), # using 50 trees for speed\n",
    "    \"Support Vector Machine\": LinearSVC(random_state=42, dual=False)\n",
    "}\n",
    "\n",
    "# Train models\n",
    "print(\"Training models... This might take a few minutes.\")\n",
    "trained_models = {}\n",
    "for name, model in models.items():\n",
    "    print(f\"Training {name}...\")\n",
    "    model.fit(X_train_sm, y_train_sm)\n",
    "    trained_models[name] = model\n",
    "print(\"Training complete!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Phase 4: Evaluation & Comparative Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluation dictionary\n",
    "results = {}\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, (name, model) in enumerate(trained_models.items()):\n",
    "    # Predict on unseen test set\n",
    "    y_pred = model.predict(X_test)\n",
    "    \n",
    "    # Metrics\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    prec = precision_score(y_test, y_pred)\n",
    "    rec = recall_score(y_test, y_pred)\n",
    "    f1 = f1_score(y_test, y_pred)\n",
    "    \n",
    "    results[name] = [acc, prec, rec, f1]\n",
    "    \n",
    "    # Confusion Matrix\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], \n",
    "                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])\n",
    "    axes[i].set_title(f\"{name} Confusion Matrix\")\n",
    "    axes[i].set_ylabel('True Label')\n",
    "    axes[i].set_xlabel('Predicted Label')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Results Comparison\n",
    "results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T\n",
    "print(results_df.round(4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detailed Classification Report for the best model (typically Random Forest or Logistic Regression)\n",
    "print(\"Classification Report for Random Forest:\\n\")\n",
    "print(classification_report(y_test, trained_models[\"Random Forest\"].predict(X_test)))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('credit_card_fraud_detection.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!")
