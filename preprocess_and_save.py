from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Step 1: Fetch the dataset from UCI ML Repository
credit_approval = fetch_ucirepo(id=27)

# Step 2: Access the features and targets
X = credit_approval.data.features
y = credit_approval.data.targets

# Step 3: Data Preprocessing

# Handle missing values
# Fill numeric columns with mean
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

# Save the scaler and label encoders
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')

print("Scaler and Label Encoders have been saved successfully.")
