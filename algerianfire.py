from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## Import ridge regression model and scaler
model = pickle.load(open('models/ridge_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data
        data = request.form.to_dict()
        
        # Convert form data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure all features are present and in the correct order
        expected_features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'] 

        input_data = input_data[expected_features]
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        
        # Return the prediction result
        return render_template('home.html', result=prediction[0])
        
    else:
        return render_template('home.html')    

if __name__ == '__main__':
    app.run(debug=True) 
    