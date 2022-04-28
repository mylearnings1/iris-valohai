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
    # Ensure that we've received a file named 'image' through POST
    # If we have a valid request proceed, otherwise return None
    if request.method != 'POST' and 'image' not in request.files:
        return None
