#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# LOAD DATASET
df = pd.read_csv("credit_risk_dataset.csv")

# Drop unnecessary column
if 'customer_id' in df.columns:
    df.drop('customer_id', axis=1, inplace=True)


#BASIC DATA INSPECTION
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.describe())

# Missing values
print("\nMissing Values:\n", df.isnull().sum())


# DATA CLEANING
# Clean string values
df['gender'] = df['gender'].str.strip().str.title()
df['employment_status'] = df['employment_status'].str.strip().str.title()

# Encoding (consistent mapping)
gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
employment_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}

df['gender'] = df['gender'].map(gender_map)
df['employment_status'] = df['employment_status'].map(employment_map)

# Drop any remaining NaN (after mapping)
df.dropna(inplace=True)

# Final check
print("\nMissing Values AFTER:\n", df.isnull().sum())


#EXPLORATORY DATA ANALYSIS (EDA)
# Univariate
sns.histplot(df['age'],bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.countplot(x='credit_risk', data=df)
plt.title("Credit Risk Distribution")
plt.show()

# Bivariate
sns.boxplot(x='credit_risk', y='annual_income', data=df)
plt.title("Income vs Credit Risk")
plt.show()

sns.boxplot(x='credit_risk', y='credit_score', data=df)
plt.title("Credit Score vs Credit Risk")
plt.show()

sns.boxplot(x='credit_risk', y='loan_application_amount', data=df)
plt.title("Loan Amount vs Credit Risk")
plt.show()

# Multivariate
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis', linewidths=0.5, fmt=".2f", annot_kws={"size": 8})
plt.title("Correlation Heatmap")
plt.show()


#GROUPBY ANALYSIS
print("\nAverage Income by Credit Risk:")
print(df.groupby('credit_risk')['annual_income'].mean())

print("\nAverage Credit Score by Credit Risk:")
print(df.groupby('credit_risk')['credit_score'].mean())

print("\nLoan Amount by Employment Status:")
print(df.groupby('employment_status')['loan_application_amount'].mean())


#FEATURE & TARGET
target = 'credit_risk'
X = df.drop(target, axis=1)
y = df[target]

#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#MODEL BUILDING
#model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

#model 2: Random Forest
rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


#MODEL EVALUATION
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

#CONFUSION MATRIX (RF)
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

#model comparison
print("\nModel Comparison")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# ROC - Logistic Regression
y_prob_lr = lr.predict_proba(X_test)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

# ROC - Random Forest
y_prob_rf = rf.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

# Plot
plt.figure()
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

#FEATURE IMPORTANCE (Random Forest)
importances = rf.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Importance Features:\n", feature_df.head())

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title("Feature Importance-Random Forest")
plt.show()

#SAVE CLEANED DATASET
df.to_csv("cleaned_credit_risk_dataset.csv", index=False)
print("Cleaned data is save successfully!")

print("\nProject Completed Successfully!")
