from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form[key]) for key in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    return render_template('index.html', prediction=prediction, probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
