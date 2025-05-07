from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import joblib
import os
import pandas as pd
import json
from django.views.decorators.csrf import csrf_exempt

model = joblib.load(os.path.join('models', 'model.pkl'))
scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

# Create your views here.
@csrf_exempt
def predict_price(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            bedrooms = data['bedrooms']
            bathrooms = data['bathrooms']
            sqft_living = data['sqft_living']
            yr_built = data['yr_built']

            input_df = pd.DataFrame([{
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft_living': sqft_living,
                'yr_built': yr_built
            }])

            input_scaled = scaler.transform(input_df)
            predicted_price = model.predict(input_scaled)[0]

            return JsonResponse({'predicted_price': round(predicted_price, 2)})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    