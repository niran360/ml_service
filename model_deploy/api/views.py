from django.shortcuts import render

# Create your views here.
import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np


# Create your views here.
@api_view(['GET'])
def index_page(request):
  return_data = {
    "error": "0",
    "message": "Successful",
  }
  return Response(return_data)


@api_view(["POST"])
def predict_survived(request):
  try:
    age = request.data.get('Age', None)
    passengerId = request.data.get('PassengerId', None)
    pclass = request.data.get('Pclass', None)
    fare = request.data.get('Fare', None)
    sex_female = request.data.get('Sex_female', None)
    embarked_C = request.data.get('Embarked_C', None)
    embarked_Q = request.data.get('Embarked_Q', None)
    embarked_S = request.data.get('Embarked_S', None)
    sibSp = request.data.get('SibSp', None)
    fields = [age, passengerId]
    if not None in fields:
      # Datapreprocessing Convert the values to float
      age = float(age)
      passengerId = float(passengerId)
      pclass = float(pclass)
      fare = float(fare)
      sex_female = float(sex_female)
      embarked_C = float(embarked_C)
      embarked_Q = float(embarked_Q)
      embarked_S = float(embarked_S)
      sibSp = float(sibSp)
      result = [age, passengerId]
      # Passing data to model & loading the model from disks
      model_path = 'ml_model/model.pkl'
      classifier = pickle.load(open(model_path, 'rb'))
      prediction = classifier.predict([result])[0]
      conf_score = np.max(classifier.predict_proba([result])) * 100
      predictions = {
        'error': '0',
        'message': 'Successfull',
        'prediction': prediction,
        'confidence_score': conf_score
      }
    else:
      predictions = {
        'error': '1',
        'message': 'Invalid Parameters'
      }
  except Exception as e:
    predictions = {
      'error': '2',
      "message": str(e)
    }

  return Response(predictions)