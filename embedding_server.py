from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

app = Flask(__name__)

# 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def mean_pooling(model_output, attention_mask):
    """평균 풀링을 통한 문장 임베딩 생성"""
    token_embeddings = model_output[0]  # First element: token embeddings
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@app.route("/embed", methods=["POST"])
def embed():
    """텍스트를 임베딩으로 변환"""
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({
                "success": False,
                "error": "text field is required"
            }), 400

        text = data["text"]

        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "success": False,
                "error": "text must be a non-empty string"
            }), 400

        # 텍스트 임베딩 생성
        encoded_input = tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embedding = mean_pooling(
            model_output, encoded_input['attention_mask'])
        embedding = sentence_embedding.cpu().numpy()[0].tolist()

        return jsonify({
            "success": True,
            "embedding": embedding,
            "dimension": len(embedding)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "model": model_name
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
