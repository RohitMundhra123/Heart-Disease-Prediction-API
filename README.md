# Heart Disease Prediction API

This API provides a machine learning model to predict the likelihood of heart disease based on various input features.

## Usage

To use this API, make a POST request to the `/predict` endpoint with the following input features:

- Gender (0 for female, 1 for male)
- Age (in years)
- Hypertension (0 for no, 1 for yes)
- Marital status (0 for unmarried, 1 for married)
- Type of work (0 for self-employed, 1 for private, 2 for government, 3 for children)
- Type of residence (0 for rural, 1 for urban)
- Glucose level (in mg/dL)
- BMI (Body Mass Index)
- Smoking status (0 for never smoked, 1 for formerly smoked, 2 for currently smoking)

The API will return a JSON response with the predicted likelihood of heart disease (0 for no, 1 for yes).

## Deployment

This API is currently deployed on Render at https://smart-health-bo2c.onrender.com/predict. You can also run it locally by running the `app.py` file and sending requests to `http://localhost:5000/predict`.

## Example
```python
import requests

url = 'https://smart-health-bo2c.onrender.com/predict'
data = {
'gender': 1,
'age': 65,
'hypertension': 1,
'marital': 1,
'work': 2,
'residence': 1,
'glucose': 250,
'bmi': 30,
'smoking': 0
}
response = requests.post(url, data=data)

print(response.json())
