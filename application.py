import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os

application = Flask(__name__)
app=application
# Load model and scaler

model_path=os.path.join("models","ridge.pkl")
scaler_path=os.path.join('models','Scaler.pkl')

ridge_model = pickle.load(open(model_path, 'rb'))
standard_scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')   # form.html is your main input form

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get values from form and convert to float
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale and predict
            new_data_scaled = standard_scaler.transform(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            )
            result = ridge_model.predict(new_data_scaled)

            return render_template('form.html', result=result[0])

        except Exception as e:
            return render_template('form.html', result=f"Error: {e}")


    return render_template('form.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
