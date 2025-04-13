from flask import Flask, jsonify, request
from flask_cors import CORS
from model import predict_preference

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

    return jsonify({'result': 'result'})

if __name__ == '__main__':
    app.run(debug=True)