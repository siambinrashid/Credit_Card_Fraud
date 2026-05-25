# Project Report: Comparative Analysis of Machine Learning Models for Highly Imbalanced Credit Card Fraud Detection

## 1. Executive Summary
This project addresses the critical challenge of credit card fraud detection on a highly imbalanced dataset, where fraudulent transactions account for less than 0.2% of the overall data. Four machine learning models were developed and compared: Logistic Regression, Decision Tree, Random Forest, and a Support Vector Machine (LinearSVC). 

To address the severe class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data. The analysis prioritizes **Recall** (minimizing False Negatives) and **F1-Score** (balancing Precision and Recall) over standard Accuracy, which is fundamentally flawed for this dataset.

## 2. Methodology
The dataset was obtained from OpenML (identical to the Kaggle dataset). It contains 284,807 transactions with 492 frauds. The features `V1` to `V28` were already PCA-transformed.

**Key Preprocessing Steps:**
1. **Scaling**: The `Amount` feature was scaled using a `RobustScaler` as it is less prone to extreme outliers compared to standard scaling.
2. **Data Splitting**: A stratified 80/20 train-test split was used to ensure the train and test sets maintained the exact proportion of fraudulent to legitimate transactions.
3. **Addressing Class Imbalance**: The SMOTE algorithm was strictly applied to the **training data only** to oversample the minority class, generating synthetic examples of fraudulent transactions to help the models learn distinguishing patterns effectively.

## 3. Results and Comparative Analysis
The models were evaluated on the unseen 20% testing set. The results are summarized below:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9737 | 0.0569 | **0.9184** | 0.1072 |
| **Decision Tree** | 0.9971 | 0.3423 | 0.7755 | 0.4750 |
| **Random Forest** | **0.9995** | **0.8989** | 0.8163 | **0.8556** |
| **Support Vector Machine** | 0.9771 | 0.0643 | 0.9082 | 0.1201 |

### 3.1 Evaluating Trade-offs
- **Logistic Regression & SVM**: Both models achieved exceptional Recall (>90%), meaning they successfully identified over 90% of the fraudulent transactions (minimizing False Negatives). However, their Precision was abysmally low (~6%). In a real-world scenario, this would result in a massive volume of False Positives, causing significant customer dissatisfaction due to blocked legitimate transactions.
- **Decision Tree**: Provided a moderate balance but struggled to generalize, likely due to overfitting the SMOTE-generated data despite typical regularization.
- **Random Forest**: Demonstrated superior overall performance. It achieved a strong Recall of 81.6% while maintaining an outstanding Precision of 89.8%. This resulted in an F1-Score of 0.8556, vastly outperforming all other models.

## 4. Conclusion & Recommendations
While Logistic Regression caught slightly more frauds, the sheer number of False Positives makes it impractical for deployment without further threshold tuning. **Random Forest is heavily recommended as the best model for real-world deployment**. It reliably identifies over 80% of fraudulent activity without overwhelming the system (or the customers) with False Positives. 

**Future Work:** 
- Fine-tune the probability threshold of the Random Forest to capture slightly more frauds (increase Recall) while keeping Precision at an acceptable level.
- Explore advanced ensemble techniques like XGBoost or LightGBM, which may offer even better precision-recall trade-offs.
