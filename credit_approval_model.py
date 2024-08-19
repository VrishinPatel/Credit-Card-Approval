# Import necessary libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from flask import Flask, request, jsonify

# Step 1: Fetch the dataset from UCI ML Repository
credit_approval = fetch_ucirepo(id=27)

# Step 2: Access the features and targets
X = credit_approval.data.features
y = credit_approval.data.targets

# Display metadata and variable information (Optional)
print(credit_approval.metadata)
print(credit_approval.variables)

# Step 3: Data Preprocessing

# Handle missing values by filling them with the mean (if any)
X = X.fillna(X.mean())

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Step 4: Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Step 8: Save the model, scaler, and label encoders
joblib.dump(model, 'credit_approval_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Step 9: Flask API for Prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Preprocess input data
    for column, le in label_encoders.items():
        if column in data:
            data[column] = le.transform([data[column]])[0]
    
    features = np.array([list(data.values())])
    features = scaler.transform(features)
    
    # Predict using the model
    prediction = model.predict(features)
    
    # Return prediction result
    return jsonify({'approval_status': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
