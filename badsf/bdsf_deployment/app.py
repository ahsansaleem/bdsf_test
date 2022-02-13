# importing required libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) 
model = pickle.load(open('bdsf_model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('myindex.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction =  model.predict(final_features)
    
    return render_template('myindex.html', prediction_text=str( np.exp(prediction)))


if __name__ == "__main__":
    app.run(debug=True)