from flask import Flask, request, jsonify
import kss

app = Flask(__name__)


@app.route('/split', methods=['POST'])
def split_sentences():
    text = request.json['text']
    sentences = kss.split_sentences(text)
    return jsonify({"sentences": sentences})


if __name__ == "__main__":
    app.run(port=5006)
