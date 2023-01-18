import uvicorn 
from fastapi import FastAPI
from iris import Iris
import pickle
import numpy as np
import pandas as pd


app = FastAPI()
pickle_in_knn = open("iris_knn.pkl", "rb")
classifier_knn=pickle.load(pickle_in_knn)

pickle_in_rf = open("iris_rf.pkl", "rb")
classifier_rf=pickle.load(pickle_in_rf)

pickle_in_svm = open("iris_svm.pkl", "rb")
classifier_svm=pickle.load(pickle_in_svm)

pickle_in_lr = open("iris_lr.pkl", "rb")
classifier_lr=pickle.load(pickle_in_lr)


@app.get('/')
def index():
    return {'message': 'Hello world'}


@app.post('/predict_from_knn')
def predict_iris(data:Iris):
    data = data.dict()
    sepal_length=data['sepal_length']
    sepal_width=data['sepal_width']
    petal_length=data['petal_length']
    petal_width=data['petal_width']

    prediction = classifier_knn.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if(prediction==0):
        prediction = 'Class Setosa'
    elif(prediction==1):
        prediction = 'Class Versicolor'
    elif(prediction==2):
        prediction = 'Class Virginica'
    return {
        'prediction': prediction
    }

@app.post('/predict_from_Random_forest')
def predict_iris(data:Iris):
    data = data.dict()
    sepal_length=data['sepal_length']
    sepal_width=data['sepal_width']
    petal_length=data['petal_length']
    petal_width=data['petal_width']

    prediction = classifier_rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if(prediction==0):
        prediction = 'Class Setosa'
    elif(prediction==1):
        prediction = 'Class Versicolor'
    elif(prediction==2):
        prediction = 'Class Virginica'
    return {
        'prediction': prediction
    }

@app.post('/predict_from_SVM')
def predict_iris(data:Iris):
    data = data.dict()
    sepal_length=data['sepal_length']
    sepal_width=data['sepal_width']
    petal_length=data['petal_length']
    petal_width=data['petal_width']

    prediction = classifier_svm.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if(prediction==0):
        prediction = 'Class Setosa'
    elif(prediction==1):
        prediction = 'Class Versicolor'
    elif(prediction==2):
        prediction = 'Class Virginica'
    return {
        'prediction': prediction
    }

@app.post('/predict_from_log_regression')
def predict_iris(data:Iris):
    data = data.dict()
    sepal_length=data['sepal_length']
    sepal_width=data['sepal_width']
    petal_length=data['petal_length']
    petal_width=data['petal_width']

    prediction = classifier_lr.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if(prediction==0):
        prediction = 'Class Setosa'
    elif(prediction==1):
        prediction = 'Class Versicolor'
    elif(prediction==2):
        prediction = 'Class Virginica'
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    