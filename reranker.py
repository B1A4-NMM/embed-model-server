from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 한국어에 특화된 sentence transformer 모델 사용
model = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 한국어 다중 작업 모델


@app.route("/rerank", methods=["POST"])
def rerank():
    data = request.get_json()
    query = data["query"]
    candidates = data["candidates"]

    # 로그로 쿼리와 후보 문서들 출력
    logger.info(f"=== 새로운 Reranking 요청 ===")
    logger.info(f"쿼리: {query}")
    logger.info(f"후보 문서 수: {len(candidates)}")
    for i, candidate in enumerate(candidates, 1):
        logger.info(f"후보 {i}: {candidate}")
    logger.info("=" * 50)

    # 쿼리와 후보 문서들을 임베딩
    query_embedding = model.encode(query)
    candidate_embeddings = model.encode(candidates)

    # 코사인 유사도 계산
    from sentence_transformers.util import cos_sim
    similarities = cos_sim(query_embedding, candidate_embeddings)[0]
    scores = similarities.cpu().numpy()

    # 점수도 로그로 출력
    logger.info("=== 계산된 점수 ===")
    for i, (candidate, score) in enumerate(zip(candidates, scores)):
        logger.info(f"후보 {i+1} 점수: {score:.4f} - {candidate}")
    logger.info("=" * 50)

    reranked = sorted(
        [{"text": c, "score": float(s)}
         for c, s in zip(candidates, scores)],
        key=lambda x: x["score"], reverse=True
    )

    # 최종 결과도 로그로 출력
    logger.info("=== 최종 Reranking 결과 ===")
    for i, item in enumerate(reranked, 1):
        logger.info(f"순위 {i}: {item['score']:.4f} - {item['text']}")
    logger.info("=" * 50)

    return jsonify(reranked)


if __name__ == "__main__":
    app.run(port=5002)
