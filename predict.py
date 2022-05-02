from fastapi import FastAPI, File, UploadFile
import sklearn
import pandas as pd
import numpy
import joblib
import io
 
app = FastAPI()
 
model_path = 'model_dt.jbl'
loaded_model = None
 
@app.post("{full_path:path}")
async def predict(data: UploadFile = File(...)):
    img = await pd.read_csv(data.file)
 
    # Resize image and convert to grayscale
    #img = img.resize((28, 28)).convert('L')
    #img_array = numpy.array(img)
 
    #image_data = numpy.reshape(img_array, (1, 28, 28))
 
    global loaded_model
    # Check if model is already loaded
 
    if not loaded_model:
        loaded_model = joblib.load(model_path)
 
    # Predict with the model
    prediction = loaded_model.predict(img)
 
    return f'Predicted_Digit: {prediction}'
