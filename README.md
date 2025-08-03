# Telecom-X-part-2-Data-Science-
# 📊 Customer Churn Prediction with Machine Learning

This project develops machine learning models to predict **customer churn** using a dataset from the telecom industry. It includes data cleaning, feature selection, undersampling/oversampling, and evaluation using K-Fold cross-validation and ROC/AUC metrics.

---

## 📁 Project Structure
📦Telecom-X-part-2-Data-Sience
┣ 📂data
┃ ┗ 📄 TelecomX_Data_Tratada.csv
┣ 📂notebooks
┃ ┗ 📄 TelecomX(Challenge_parte-2).ipynb
┣ 📄 README.md
┗ 📄 requirements.txt

---

## 📌 Project Goals

- Predict whether a customer will **churn** using classification models.
- Evaluate models using **precision, recall, F1-score, and AUC**.
- Use **K-Fold cross-validation** for performance stability.
- Handle class imbalance using **SMOTE** and **NearMiss**.

---

## 📊 Dataset Overview

- **Source**: `TelecomX_Data_Tratada.csv`
- **Target Variable**: `Churn` / `Cliente_Detractor` (binary)
- **Key Features**:
  - Demographics (e.g., `SeniorCitizen`, `Partner`, `Dependents`)
  - Services (`InternetService`, `PhoneService`, etc.)
  - Tenure (`Meses_Contrato`)
  - Billing (`Total_Facturado`, `Charges.Monthly`)

---

## 🧹 Preprocessing Steps

- Removed unnecessary features: `customerID`, `Tiene_*`, `Facturacion_Electronica`.
- One-hot encoding of categorical variables using **`OneHotEncoder`** via `ColumnTransformer`.
- Split data into **70% training / 30% test** using `train_test_split(stratify=y)`.

---

## ⚖️ Handling Class Imbalance

### ✅ Undersampling (NearMiss)

- Applied **NearMiss v1** on training data to reduce the majority class size.
- Best suited for **KNN** model.

### ✅ Oversampling (SMOTE)

- Applied **SMOTE** on training data to synthetically generate minority samples.
- Used with **Random Forest** and others.

---

## 🤖 Models Trained

### 🔷 K-Nearest Neighbors (KNN)

- Trained on:
  - Raw training data
  - Under-sampled data (NearMiss)
- Evaluated using:
  - Accuracy
  - ROC Curve & AUC (per fold)

### 🌲 Random Forest

- Trained on:
  - Raw training data
  - Over-sampled data (SMOTE)
- Evaluated using:
  - Accuracy
  - ROC Curve & AUC (per fold)

---

## 🔁 Cross-Validation

Used **5-Fold Cross-Validation (KFold)** to:
- Split data into 5 train/test sets
- Track performance consistency
- Plot **ROC curves for each fold**
- Average **AUC score** across folds

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-Score**
- **ROC Curve & AUC**
- **Box Plots** for:
  - `Meses_Contrato` vs `Cliente_Detractor`
  - `Total_Facturado` vs `Cliente_Detractor`

---

## 📊 Feature Selection & Analysis

- Computed **feature importances** from Random Forest.
- Iteratively evaluated top-N features (0–35 in steps) using precision, recall, F1-score.

---

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt

---

📚 Results Summary
Model	        AUC (Mean)	Method
KNN	          0.76	      NearMiss
RandomForest	0.86	      Undersampling

---

Randy Vivas
Junior Data Scientist 
