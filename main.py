from flask import Flask,request,jsonify
import pickle
import numpy as np


model = pickle.load(open('heart_disease_model.pkl', 'rb'))
model1 = pickle.load(open('stroke_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    gender = int(request.form.get('gender'))
    age = int(request.form.get('age'))
    hypertension = int(request.form.get('hypertension'))
    ever_married = int(request.form.get('marital'))
    work_type = int(request.form.get('work'))
    residence_type = int(request.form.get('residence'))
    avg_glucose_level = int(request.form.get('glucose'))
    bmi = int(request.form.get('bmi'))
    smoking_status = int(request.form.get('smoking'))

    input_query = np.array([gender,age,hypertension,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1, -1)
    result = model.predict(input_query)[0]
    heart_disease = int(result)

    input_query2 = np.array([gender, age, hypertension,heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi,smoking_status]).reshape(1, -1)
    result2 = model1.predict(input_query2)[0]

    return jsonify({'heart disease':str(result)},{'heart stroke':str(result2)})

if __name__ == '__main__':
    app.run(debug=True)