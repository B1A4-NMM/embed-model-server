from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer("intfloat/multilingual-e5-large")


@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json()
    input_text = data["text"]
    prefix = data.get("prefix", "query:")  # 기본값: query

    print(f"{prefix} : {input_text}")

    embedding = model.encode(
        f"{prefix} {input_text}", normalize_embeddings=True)
    return jsonify({"embedding": embedding.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
