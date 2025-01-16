# Telco Customer Churn Prediction

This repository contains a machine learning pipeline to predict customer churn using the **Telco Customer Churn** dataset from Kaggle. The project leverages **Logistic Regression** with L1 regularization to identify factors influencing customer churn and make predictions.

---

## **Overview**
Customer churn refers to when customers stop doing business with a company. Identifying such customers in advance can help businesses take proactive measures to retain them. This project aims to:

- Preprocess the dataset for machine learning.
- Train a Logistic Regression model with feature scaling and L1 regularization.
- Evaluate the model using metrics like accuracy, confusion matrix, ROC-AUC, and classification report.
- Save the trained model for future use.

---

## **Dataset**
The **Telco Customer Churn** dataset contains information about customers' demographics, services they use, account details, and whether they churned.

### Dataset Features:
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, InternetService, OnlineSecurity, TechSupport, etc.
- **Account Information**: Contract type, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Target**: `Churn` (Yes/No)

### Link to Dataset:
[Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## **Project Steps**

### **1. Data Preprocessing**
- Converted categorical features to numeric using a mapping dictionary.
- Handled missing or blank values in the `TotalCharges` column by replacing them with `0.0`.
- Dropped irrelevant or redundant features (`customerID`, `Dependents`, `tenure`, etc.).
- Shuffled the dataset to reduce bias.

### **2. Train-Test Split**
- Split the dataset into **90% training data** and **10% test data**.

### **3. Model Pipeline**
The project uses a **scikit-learn Pipeline** with two steps:
1. **StandardScaler**: Standardizes features by scaling them to have a mean of 0 and a standard deviation of 1.
2. **Logistic Regression**: Trained with L1 regularization to perform feature selection and improve interpretability.

### **4. Model Evaluation**
Evaluated the model using:
- **Accuracy Score**: Percentage of correctly predicted instances.
- **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Precision, recall, F1-score, and support for each class.
- **ROC-AUC Score**: Quantifies the model's ability to distinguish between churn and non-churn.
- **ROC Curve**: Visualizes the trade-off between true positive rate and false positive rate at various thresholds.

### **5. Save the Model**
The trained pipeline is saved using **joblib** for future use.

---

## **How to Run**

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/telco-customer-churn.git
cd telco-customer-churn
```

### **2. Run the Notebook**
Use Jupyter Notebook to execute the code:
```bash
jupyter notebook
```
Open the `main.ipynb` file and run all cells.

### **3. Model Output**
- The model will display metrics (accuracy, confusion matrix, ROC-AUC, etc.) after evaluation.
- The trained pipeline will be saved as `logistic_regression_model.pkl`.

---

## **Results**
- **Accuracy**: ~73%
- **ROC-AUC**: ~75%

The model is capable of identifying churners with reasonable accuracy and recall, making it a valuable tool for retention strategies.

---

## **Files in the Repository**
- `main.ipynb`: Jupyter Notebook with the full pipeline.
- `logistic_regression_model.pkl`: Saved trained pipeline.

---

## **Future Improvements**
- Hyperparameter tuning using GridSearchCV.
- Experimenting with advanced models like Random Forest or Gradient Boosting.
- Creating additional features (e.g., average spending rate).

---

## **License**
This project is open-source and available under the [MIT License](LICENSE).

