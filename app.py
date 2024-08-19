from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask application
app = Flask(__name__)

# Paths to model files
model_path = 'model/credit_approval_model.pkl'
scaler_path = 'model/scaler.pkl'
label_encoders_path = 'model/label_encoders.pkl'

# Check if model files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(label_encoders_path):
    raise FileNotFoundError("Model, scaler, or label encoders file not found. Make sure the files are in the correct directory.")

# Load the trained model, scaler, and label encoders
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(label_encoders_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        print(f"Received data: {data}")
        
        # Preprocess input data
        for column, le in label_encoders.items():
            if column in data:
                # Check if the data type is a string (likely categorical)
                if isinstance(data[column], str):
                    try:
                        # Transform categorical input using LabelEncoder
                        data[column] = le.transform([data[column]])[0]
                    except ValueError as ve:
                        return jsonify({'error': f"Value error for {column}: {ve}"}), 400
                else:
                    # Ensure numerical data is treated correctly
                    try:
                        data[column] = int(data[column])
                    except ValueError as ve:
                        return jsonify({'error': f"Could not convert {column} to integer: {ve}"}), 400
            else:
                return jsonify({'error': f'Missing value for {column}'}), 400

        print(f"Preprocessed data: {data}")
        
        # Convert data to a format that can be used by the model
        features = np.array([list(data.values())])
        print(f"Features before scaling: {features}")
        features = scaler.transform(features)
        print(f"Features after scaling: {features}")
        
        # Predict using the model
        prediction = model.predict(features)
        print(f"Prediction: {prediction}")
        
        # Map the prediction to a more descriptive response
        if prediction[0] == '+':
            result = 'Approved'
        else:
            result = 'Denied'
        
        return jsonify({'approval_status': result})
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
