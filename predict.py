import json
from werkzeug.wrappers import Request, Response
import pandas as pd
import joblib

model = None

def predict(environ, start_response):
    # Load input image data from the HTTP request
    request = Request(environ)
    if not request.files:
        return Response('no file uploaded', 400)(environ, start_response)
    data_file = next(request.files.values())
    #data = process_image(Image.open(image_file))
    data = pd.read_csv(data_file)

    # The predictor must be lazily instantiated;
    # the TensorFlow graph can apparently not be shared
    # between processes.
    global model
    if not model:
        model = joblib.load('model_dt.jbl')
    prediction = model.predict(data)

    # The following line allows Valohai to track endpoint predictions
    # while the model is deployed. Here we remove the full predictions
    # details as we are only interested in tracking the rest of the results.
    #print(json.dumps({'vh_metadata': {k: v for k, v in prediction.items() if k != 'predictions'}}))

    # Return a JSON response
    response = Response(json.dumps(prediction), content_type='application/json')
    return response(environ, start_response)


# Run a local server for testing with `python deploy.py`
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8000, predict)
