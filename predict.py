from werkzeug.wrappers import Request, Response
import io
import numpy
import pandas as pd
import sklearn 
import json

model_path = 'model_dt.jbl'

# Store our model
IrisModel = None

def read_input(request):
    # Ensure that we've received a file named 'data' through POST
    # If we have a valid request proceed, otherwise return None
    if request.method != 'POST' and 'data' not in request.files:
        return None
    
    # Load the data that was sent
    DataFile = request.files.get('data')
    dt = open(DataFile.stream,w+)
    dt.load()

    # Resize image to 28x28 and convert to grayscale
    dt = dt.drop(columns=['Id'],axis=1)

    return dt


def mypredictor(environ, start_response):
    # Get the request object from the environment
    request = Request(environ)

    global IrisModel
    if not IrisModel:
        IrisModel = joblib.load(model_path)

    # Get the image file from our request
    data = read_input(request)

    # If read_input didn't find a valid file
    if (data is None):
        response = Response("\nNo data", content_type='text/html')
        return response(environ, start_response)

    # Use our model to predict the class of the file sent over a form.
    prediction = IrisModel.predict(data)

    # Generate a JSON output with the prediction
    json_response = json.dumps("{Predicted_Digit: %s}" % prediction[0])

    # Send a response back with the prediction
    response = Response(json_response, content_type='application/json')
    return response(environ, start_response)

# When running locally
if __name__ == "__main__":
    from werkzeug.serving import run_simple

    # Update model path to point to our downloaded model when testing locally
    model_path = '.models/model_dt.jbl'

    # Run a local server on port 5000.
    run_simple("localhost", 8000, mypredictor)
