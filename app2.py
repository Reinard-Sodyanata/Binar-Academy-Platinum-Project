from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: "API Documentation for Deep Learning"),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Deep Learning')
    },
host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,
                    config=swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

import re

def cleansing(sent):
  string = sent.lower()
  string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
  return string

file = open("resources_of_nn/feature.pickle",'rb')
feature_file_from_nn = pickle.load(file)
file.close()

model_file_from_nn = load_model('model_of_nn/model_of_nn.h5')

@swag_from("docs/neuralnet.yml", methods=['POST'])
@app.route('/neuralnet', methods=['POST'])
def neuralnet():
    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_nn.shape[1])

    prediction = model_file_from_nn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using NN",
        'data':{
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/neuralnet_file.yml", methods=['POST'])
@app.route('/neuralnet-file', methods=['POST'])
def nn_file():

    file = request.files['file']
    file_text = file.read().decode()
    text = [cleansing(file_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_nn.shape[1])

    prediction = model_file_from_nn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using NN File",
        'data':{
            'text': text,
            'sentiment': get_sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data

if __name__== '__main__':
    app.run()