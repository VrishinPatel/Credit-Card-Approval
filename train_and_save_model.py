from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create 'model' directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Step 1: Fetch the dataset from UCI ML Repository
credit_approval = fetch_ucirepo(id=27)

# Step 2: Access the features and targets
X = credit_approval.data.features
y = credit_approval.data.targets

# Step 3: Data Preprocessing

# Handle missing values
numeric_cols = X.select_dtypes(include=[np.number]).columns
X.loc[:, numeric_cols] = X.loc[:, numeric_cols].fillna(X[numeric_cols].mean())

non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
X.loc[:, non_numeric_cols] = X.loc[:, non_numeric_cols].fillna(X[non_numeric_cols].mode().iloc[0])

# Encode categorical variables
label_encoders = {}
for column in non_numeric_cols:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Step 6: Save the model, scaler, and label encoders
joblib.dump(model, 'model/credit_approval_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')

print("Model, Scaler, and Label Encoders have been saved successfully.")
