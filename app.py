import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, url_for, render_template, request
import pickle
from joblib import dump, load

column_names = ["pregnancies", "glucose", "BP", "skinfold", "insulin", "bmi", "pedigree", "age", "class"]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
   app.run()

# column_names = ["pregnancies", "glucose", "BP", "skinfold", "insulin", "bmi", "pedigree", "age", "class"]
# Form Submission Route
@app.route('/send', methods=['POST','GET'])
def send():
    features = [int(x) for x in request.form.values()]
    patient = [np.array(features)]
    classifier = pickle.load(open('trainedModel.sav', 'rb'))
    scalar = load('std_scaler.bin')
    patient = scalar.transform(patient)
    pred = classifier.predict(patient)
    if pred == 1:
       return render_template('index.html', predicts = "Chances are that you have diabetes. Kindly consult a specialist for further treatment.")
    else:
       return render_template('index.html', predicts = "You're safe from diabetes. Enjoy!")