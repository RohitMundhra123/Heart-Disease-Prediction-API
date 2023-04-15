from flask import Flask,request,jsonify
import pickle
import numpy as np


model = pickle.load(open('heart_disease_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    gender = float(request.form.get('gender'))
    age = float(request.form.get('age'))
    hypertension = float(request.form.get('hypertension'))
    marital = float(request.form.get('marital'))
    work = float(request.form.get('work'))
    residence = float(request.form.get('residence'))
    glucose = float(request.form.get('glucose'))
    bmi = float(request.form.get('bmi'))
    smoking = float(request.form.get('smoking'))

    input_query = np.array([gender,age,hypertension,marital,work,residence,glucose*4.76,bmi,smoking]).reshape(1, -1)
    result = model.predict(input_query)[0]

    return jsonify({'heart disease':str(result)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)