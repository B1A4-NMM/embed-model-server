from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = SentenceTransformer('jhgan/ko-sroberta-multitask')


@app.route("/rerank", methods=["POST"])
def rerank():
    data = request.get_json()
    query = data["query"]
    candidates = data["candidates"]  # [{id, text}, ...]

    logger.info(f"=== 새로운 Reranking 요청 ===")
    logger.info(f"쿼리: {query}")
    logger.info(f"후보 문서 수: {len(candidates)}")
    logger.info(f"후보 문서: {candidates}")
    for i, candidate in enumerate(candidates, 1):
        logger.info(f"후보 {i}: id={candidate['id']} / text={candidate['text']}")
    logger.info("=" * 50)

    # 텍스트만 추출하여 임베딩
    candidate_texts = [c["text"] for c in candidates]
    query_embedding = model.encode(query)
    candidate_embeddings = model.encode(candidate_texts)

    similarities = cos_sim(query_embedding, candidate_embeddings)[0]
    scores = similarities.cpu().numpy()

    logger.info("=== 계산된 점수 ===")
    for i, (candidate, score) in enumerate(zip(candidates, scores)):
        logger.info(
            f"후보 {i+1} 점수: {score:.4f} - id={candidate['id']} / text={candidate['text']}")
    logger.info("=" * 50)

    reranked = sorted(
        [
            {
                "id": c["id"],
                "text": c["text"],
                "score": float(s)
            }
            for c, s in zip(candidates, scores)
        ],
        key=lambda x: x["score"], reverse=True
    )

    logger.info("=== 최종 Reranking 결과 ===")
    for i, item in enumerate(reranked, 1):
        logger.info(
            f"순위 {i}: {item['score']:.4f} - id={item['id']} / text={item['text']}")
    logger.info("=" * 50)

    return jsonify(reranked)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
