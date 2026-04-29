from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the entire pipeline model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        input_data = {
            'MSZoning': request.form['MSZoning'],
            'LotArea': float(request.form['LotArea']),
            'OverallQual': int(request.form['OverallQual']),
            'OverallCond': int(request.form['OverallCond']),
            'YearBuilt': int(request.form['YearBuilt']),
            'GrLivArea': float(request.form['GrLivArea']),
            'FullBath': int(request.form['FullBath']),
            'BedroomAbvGr': int(request.form['BedroomAbvGr']),
            'GarageCars': int(request.form['GarageCars'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure all required columns are present
        expected_cols = model.named_steps['preprocessor'].feature_names_in_

        # Add missing columns with defaults
        for col in expected_cols:
            if col not in input_df.columns:
                # if the column name was numeric in train data
                input_df[col] = 0

        # Reorder columns to match training data
        input_df = input_df[expected_cols]

        # Predict
        prediction = model.predict(input_df)[0]
        output = round(prediction, 2)

        return render_template('result.html', prediction_text=f"🏠 Estimated House Price: ${output:,.2f}")

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
