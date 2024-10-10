# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

# Create a Flask application instance
app = Flask(__name__)

# Load the trained model from the 'model.pkl' file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    consumption = float(request.form['consumption'])
    
    # Reshape the input data for the model
    input_data = np.array([[consumption]])
    
    # Make a prediction using the trained model
    prediction = model.predict(input_data)
    
    # Return the prediction result
    return render_template('index.html', prediction=prediction[0])

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
