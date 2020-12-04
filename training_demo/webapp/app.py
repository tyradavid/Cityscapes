# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:41:16 2020

@author: tyra1
"""

from flask import Flask, render_template, request

# Import your models
from cityscapes_predictor import CITYSCAPES_predictor

app = Flask(__name__)

# Instantiate your models
cityscapes = CITYSCAPES_predictor()


# Base endpoint to perform prediction.
@app.route('/', methods=['POST'])
def make_prediction():
    prediction = cityscapes.predict(request)
    return render_template('index.html', prediction=prediction, generated_text=None, tab_to_show='cityscapes')


@app.route('/', methods=['GET'])
def load():
    return render_template('index.html', prediction=None, generated_text=None, tab_to_show='cityscapes')


@app.route('/predict/image', methods=['POST'])
def make_image_prediction():
    prediction = cityscapes.predict(request)
    print(prediction)
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)
