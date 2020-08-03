import generate_rap
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
app = Flask(__name__)
api = Api(app)

@app.route('/<string:text>', methods=['GET'])
def get_task(text):
    global model, tokenizer
    rap = generate_rap.from_prompt(model, tokenizer, text)
    rap_cleaned = rap.replace('<|endoftext|>', '')
    return jsonify({'rap': rap_cleaned})

global model, tokenizer
model, tokenizer = generate_rap.load_all()# add inputs for temperature etc in future

if __name__ == "__main__":
    app.run(host='0.0.0.0')
