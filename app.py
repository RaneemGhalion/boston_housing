import pickle
from flask import Flask,request,app,json,url_for,render_template,jsonify
import numpy as np
import pandas as pd
# import jsonify
from statsmodels.api import add_constant

app=Flask(__name__) # create starting point for the application 

regmodel=pickle.load(open('regmodel.pkl','rb')) # read byte mode , load model 
scalar=pickle.load(open('scaling.pkl','rb')) 
@app.route('/')

def home ():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']  # Giving input should be in JSON API and store it in data variable
    print("Received data:", data)
    
    # Extract the features from the data
    features = [data['area'],
                data['bedrooms'],
                data['bathrooms'],
                data['stories'],
                data['parking'],
                data['x']]
    
    # Debug: Print the features list
    print("Features list:", features)
    
    # Ensure the features list matches the model's expected input
    if len(features) !=6:
        return jsonify({"error": "Incorrect number of features"}), 400
    
    # Convert to numpy array and reshape
    scaled_data = np.array(features).reshape(1, -1)
    
    # Debug: Print the scaled_data
    print("Scaled data before transformation:", scaled_data)
    
    # Transform the features using the scaler
    new_data = scalar.transform(scaled_data)
    
    # Debug: Print the new_data after transformation
    print("Data after transformation:", new_data)
    new_data = add_constant(new_data, has_constant='add')

    # Predict using the model
    output = regmodel.predict(new_data)
    
    # Debug: Print the prediction result
    print("Prediction output:", output[0])
    return  jsonify({"predicted_price": output[0]})
if __name__=="__main__":
    app.run(debug=True)