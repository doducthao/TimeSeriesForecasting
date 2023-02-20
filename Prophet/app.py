from flask import Flask, request, jsonify, render_template
import numpy as np 
import pandas as pd 
from prophet.serialize import model_from_json

app = Flask(__name__)

# Load the model
with open('model.json', 'r') as fin:
    model = model_from_json(fin.read())

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']
    date = pd.to_datetime(date, format="%Y-%m-%d")
    # Creating a future data frame for the test sample
    future = pd.DataFrame({'ds': [date]})
    # Making predictions for the test sample
    forecast = model.predict(future)
    # Extracting the predicted value
    predicted_value = forecast['yhat'].iloc[0]
    return render_template("home.html", prediction_text="The forecasted value is {:.2f}".format(predicted_value))

if __name__ == "__main__":
    app.run(debug=True)