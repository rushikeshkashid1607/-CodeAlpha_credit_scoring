import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Dataset and Save as CSV
np.random.seed(42)
data = {
    'income': np.random.normal(50000, 15000, 1000),  # Random incomes around 50k
    'debt': np.random.normal(20000, 5000, 1000),     # Random debts around 20k
    'credit_history': np.random.randint(300, 850, 1000),  # Credit scores between 300-850
    'creditworthy': np.random.choice([0, 1], 1000)   # 0: Not creditworthy, 1: Creditworthy
}
df = pd.DataFrame(data)
df.to_csv('credit_data.csv', index=False)
print("Dataset generated and saved as 'credit_data.csv'.")

# Step 2: Load and Explore the Dataset
df = pd.read_csv('credit_data.csv')
print("\nDataset Preview:")
print(df.head())
print("\nDataset Summary:")
print(df.describe())
print("\nClass Distribution:")
print(df['creditworthy'].value_counts())

# Step 3: Data Preprocessing
# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Remove outliers using IQR for 'income' and 'debt'
Q1 = df[['income', 'debt']].quantile(0.25)
Q3 = df[['income', 'debt']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['income', 'debt']] < (Q1 - 1.5 * IQR)) | (df[['income', 'debt']] > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"\nDataset size after removing outliers: {df.shape}")

# Step 4: Feature Engineering
# Add debt-to-income ratio
df['debt_to_income'] = df['debt'] / df['income']

# Scale features
scaler = StandardScaler()
features = ['income', 'debt', 'credit_history', 'debt_to_income']
df[features] = scaler.fit_transform(df[features])
print("\nDataset with Engineered Features:")
print(df.head())

# Step 5: Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of creditworthy vs. income
plt.figure(figsize=(8, 6))
sns.boxplot(x='creditworthy', y='income', data=df)
plt.title('Income vs Creditworthiness')
plt.show()

# Step 6: Split Data and Train Model
X = df[['income', 'debt', 'credit_history', 'debt_to_income']]
y = df['creditworthy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate Model
# Predict on test set
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Scores: {scores}")
print(f"Average CV Score: {scores.mean():.2f}")

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance')
plt.show()

# Step 8: Save Model and Scaler
joblib.dump(model, 'credit_scoring_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved as 'credit_scoring_model.pkl' and 'scaler.pkl'.")

# Step 9: Prediction Function for New Applicants
def predict_creditworthiness(income, debt, credit_history):
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('credit_scoring_model.pkl')
    debt_to_income = debt / income
    features = scaler.transform([[income, debt, credit_history, debt_to_income]])
    prediction = model.predict(features)
    return "Creditworthy" if prediction[0] == 1 else "Not Creditworthy"

# Test the prediction function
test_applicant = {'income': 60000, 'debt': 15000, 'credit_history': 700}
result = predict_creditworthiness(test_applicant['income'], test_applicant['debt'], test_applicant['credit_history'])
print(f"\nPrediction for applicant (Income: {test_applicant['income']}, Debt: {test_applicant['debt']}, Credit History: {test_applicant['credit_history']}): {result}")