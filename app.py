import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pickle

app = Flask(__name__)

filename = 'finalized_model.sav'
model = joblib.load(filename)
with open('vectorizer.pickle', 'rb') as handle:
	vectorizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # temp=request.get_data(as_text=True)
    inpt=request.form.get('sentence')
    message=vectorizer.transform([inpt])
    pred = model.predict(message)
    if pred == 1:
        output= "Spam"
    else:
        output= "No Spam"
    # return str(pred)
    return render_template('index.html',prediction_text='The entered sentence has {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)