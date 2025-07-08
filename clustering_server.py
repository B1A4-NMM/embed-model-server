from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from typing import List, Dict, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 전역 변수로 모델 로드
model = None


def load_model():
    """jhgan/ko-sroberta-multitask 모델을 로드합니다."""
    global model
    try:
        logger.info("Loading jhgan/ko-sroberta-multitask model...")
        model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_embeddings(sentences: List[str]) -> np.ndarray:
    """문장들을 임베딩으로 변환합니다."""
    if model is None:
        raise ValueError("Model not loaded")

    try:
        embeddings = model.encode(sentences, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise


def find_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10) -> Tuple[int, float]:
    """Silhouette score를 사용하여 최적의 클러스터 개수를 찾습니다."""
    if len(embeddings) < 2:
        return 1, 0.0

    max_clusters = min(max_clusters, len(embeddings) - 1)
    if max_clusters < 2:
        return 1, 0.0

    best_score = -1
    best_k = 1

    for k in range(2, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(embeddings)

            if len(np.unique(cluster_labels)) < 2:
                continue

            score = silhouette_score(embeddings, cluster_labels)

            if score > best_score:
                best_score = score
                best_k = k

        except Exception as e:
            logger.warning(
                f"Error calculating silhouette score for k={k}: {e}")
            continue

    return best_k, best_score


def get_cluster_representatives(embeddings: np.ndarray, sentences: List[str], n_clusters: int) -> List[Dict]:
    """각 클러스터의 대표 문장을 찾습니다."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_representatives = []

    for cluster_id in range(n_clusters):
        # 해당 클러스터에 속한 문장들의 인덱스
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            continue

        # 클러스터 내 문장들의 임베딩
        cluster_embeddings = embeddings[cluster_indices]

        # 클러스터 중심과의 거리 계산
        cluster_center = kmeans.cluster_centers_[cluster_id]

        # 각 문장과 중심과의 거리 계산
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)

        # 가장 중심에 가까운 문장을 대표 문장으로 선택
        representative_idx = cluster_indices[np.argmin(distances)]

        cluster_info = {
            "cluster_id": int(cluster_id),
            "representative_sentence": sentences[representative_idx],
            "representative_index": int(representative_idx),
            "cluster_size": int(len(cluster_indices)),
            "sentences": [sentences[i] for i in cluster_indices],
            "sentence_indices": [int(i) for i in cluster_indices]
        }

        cluster_representatives.append(cluster_info)

    return cluster_representatives


@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인 엔드포인트"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route('/cluster', methods=['POST'])
def cluster_sentences():
    """문장들을 클러스터링하고 결과를 반환합니다."""
    try:
        data = request.get_json()

        if not data or 'sentences' not in data:
            return jsonify({
                "error": "sentences field is required"
            }), 400

        sentences_input = data['sentences']

        # 입력 검증: 리스트이며 각 항목이 dict이고 id(int), text(str) 포함
        if not isinstance(sentences_input, list) or len(sentences_input) == 0:
            return jsonify({
                "error": "sentences must be a non-empty list"
            }), 400
        for s in sentences_input:
            if not (isinstance(s, dict) and 'id' in s and 'text' in s and isinstance(s['id'], int) and isinstance(s['text'], str)):
                return jsonify({
                    "error": "Each sentence must be an object with integer 'id' and string 'text'"
                }), 400

        # id, text 분리
        ids = [s['id'] for s in sentences_input]
        texts = [s['text'] for s in sentences_input]

        # 최대 클러스터 개수 설정 (기본값: 10)
        max_clusters = data.get('max_clusters', 20)

        logger.info(
            f"Processing {len(texts)} sentences with max_clusters={max_clusters}")

        # 문장들을 임베딩으로 변환
        embeddings = get_embeddings(texts)

        # 최적의 클러스터 개수 찾기
        optimal_k, silhouette_score_value = find_optimal_clusters(
            embeddings, max_clusters)

        logger.info(
            f"Optimal clusters: {optimal_k}, Silhouette score: {silhouette_score_value:.4f}")

        # 클러스터링 수행 및 대표 문장 찾기
        # KMeans 클러스터링
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings)

        clusters = []
        for cluster_id in range(optimal_k):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_embeddings = embeddings[cluster_indices]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(
                cluster_embeddings - cluster_center, axis=1)
            representative_idx_in_cluster = np.argmin(distances)
            representative_global_idx = cluster_indices[representative_idx_in_cluster]
            # 대표 문장 정보
            representative = {
                "id": ids[representative_global_idx],
                "text": texts[representative_global_idx]
            }
            # 클러스터 내 문장 정보
            cluster_sentences = [
                {"id": ids[i], "text": texts[i]} for i in cluster_indices
            ]
            clusters.append({
                "cluster_id": int(cluster_id),
                "representative_sentence": representative,
                "cluster_size": int(len(cluster_indices)),
                "sentences": cluster_sentences
            })

        response = {
            "optimal_clusters": optimal_k,
            "silhouette_score": round(silhouette_score_value, 4),
            "total_sentences": len(texts),
            "clusters": clusters
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/embeddings', methods=['POST'])
def get_sentence_embeddings():
    """문장들의 임베딩을 반환합니다."""
    try:
        data = request.get_json()

        if not data or 'sentences' not in data:
            return jsonify({
                "error": "sentences field is required"
            }), 400

        sentences = data['sentences']

        if not isinstance(sentences, list) or len(sentences) == 0:
            return jsonify({
                "error": "sentences must be a non-empty list"
            }), 400

        # 문장들을 임베딩으로 변환
        embeddings = get_embeddings(sentences)

        response = {
            "embeddings": embeddings.tolist(),
            "shape": embeddings.shape
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/silhouette_analysis', methods=['POST'])
def silhouette_analysis():
    """다양한 클러스터 개수에 대한 silhouette score를 분석합니다."""
    try:
        data = request.get_json()

        if not data or 'sentences' not in data:
            return jsonify({
                "error": "sentences field is required"
            }), 400

        sentences = data['sentences']

        if not isinstance(sentences, list) or len(sentences) == 0:
            return jsonify({
                "error": "sentences must be a non-empty list"
            }), 400

        max_clusters = data.get('max_clusters', 10)

        # 문장들을 임베딩으로 변환
        embeddings = get_embeddings(sentences)

        # 다양한 클러스터 개수에 대한 silhouette score 계산
        analysis_results = []
        max_k = min(max_clusters, len(embeddings) - 1)

        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(embeddings)

                if len(np.unique(cluster_labels)) < 2:
                    continue

                score = silhouette_score(embeddings, cluster_labels)

                analysis_results.append({
                    "clusters": k,
                    "silhouette_score": round(score, 4)
                })

            except Exception as e:
                logger.warning(
                    f"Error calculating silhouette score for k={k}: {e}")
                continue

        # 최적의 클러스터 개수 찾기
        if analysis_results:
            best_result = max(analysis_results,
                              key=lambda x: x['silhouette_score'])
        else:
            best_result = {"clusters": 1, "silhouette_score": 0.0}

        response = {
            "analysis": analysis_results,
            "optimal_clusters": best_result["clusters"],
            "optimal_silhouette_score": best_result["silhouette_score"]
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in silhouette analysis: {e}")
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    # 모델 로드
    load_model()

    # 서버 시작
    app.run(host='0.0.0.0', port=5005, debug=False)
