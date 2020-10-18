from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('SentimentAnalysis_predict.pkl', 'rb'))
modeltransform = pickle.load(open('SentimentAnalysis_transform.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Input = request.form['Year']
        print('Input:'+Input)
        prediction=model.predict(modeltransform.transform([Input]))
        print(prediction)
        output1 = str(prediction)
        print(output1)
        return render_template('index.html',prediction_text="Your feedback is  {}".format(prediction))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

