from flask import Flask, jsonify, request
from flask_cors import CORS
from model import predict_preference
import json
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['GET'])
def predict():
    user_response = request.args.get('user_response', default='0', type=str)
    prompt = request.args.get('prompt', default='0', type=str)
    ai_response = request.args.get('ai_response', default='0', type=str)

    result = user_response + ai_response + prompt
    thing = predict_preference(prompt, user_response, ai_response)
    print(thing)

    return jsonify({'result': thing})

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    model_info_path = os.path.join('model_variations', 'model_info.json')
    
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        return jsonify(model_info)
    else:
        return jsonify({'error': 'Model information not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)