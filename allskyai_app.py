import numpy as np
from flask import Flask
import classification

app = Flask(__name__)

@app.route('/')
def index():
  result = classification.do_classification()
  return result


# app.run(host='0.0.0.0', port=3010, debug=True)
