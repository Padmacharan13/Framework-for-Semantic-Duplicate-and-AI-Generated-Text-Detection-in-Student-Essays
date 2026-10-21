"""Fast, model-free validation tests for Flask API input handling.

Run from Plagarism_Detection/ with:
    pytest -q tests/test_api_validation.py

These tests replace the global model factory with a stub, so they do not load
transformer weights or require a running Flask server.
"""
import app as app_module


class DummyPreprocessor:
    def process_column(self, texts):
        class Result:
            sentences = ["simple sentence"]
            paragraphs = ["simple sentence"]
        return [Result(), Result()]


class DummyEmbedder:
    pass


class DummyWord2Vec:
    pass


class DummyIndex:
    def total_documents(self):
        return 0

    def search_candidates(self, query_text, embedder, top_k=5, min_score=0.0):
        return []


class DummyCrossEncoder:
    def rerank_candidates(self, query_text, candidates):
        return candidates


def _fake_models():
    return (
        DummyPreprocessor(),
        DummyEmbedder(),
        DummyWord2Vec(),
        DummyIndex(),
        DummyCrossEncoder(),
    )


def test_faiss_search_rejects_empty_query_without_model_load(monkeypatch):
    monkeypatch.setattr(app_module, "get_models", _fake_models)
    client = app_module.app.test_client()

    response = client.post("/api/faiss/search", json={"query": "   "})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_faiss_search_rejects_non_string_query(monkeypatch):
    monkeypatch.setattr(app_module, "get_models", _fake_models)
    client = app_module.app.test_client()

    response = client.post("/api/faiss/search", json={"query": ["not", "text"]})

    assert response.status_code == 400


def test_faiss_search_rejects_invalid_top_k(monkeypatch):
    monkeypatch.setattr(app_module, "get_models", _fake_models)
    client = app_module.app.test_client()

    response = client.post(
        "/api/faiss/search",
        json={"query": "test query", "top_k": "not-an-integer"},
    )

    assert response.status_code == 400


def test_faiss_search_rejects_non_boolean_rerank(monkeypatch):
    monkeypatch.setattr(app_module, "get_models", _fake_models)
    client = app_module.app.test_client()

    response = client.post(
        "/api/faiss/search",
        json={"query": "test query", "rerank": "false"},
    )

    assert response.status_code == 400


def test_faiss_search_rejects_out_of_range_min_score(monkeypatch):
    monkeypatch.setattr(app_module, "get_models", _fake_models)
    client = app_module.app.test_client()

    response = client.post(
        "/api/faiss/search",
        json={"query": "test query", "min_score": 2.0},
    )

    assert response.status_code == 400


def test_faiss_search_allows_negative_min_score_and_returns_shape(monkeypatch):
    monkeypatch.setattr(app_module, "get_models", _fake_models)
    client = app_module.app.test_client()

    response = client.post(
        "/api/faiss/search",
        json={
            "query": "test query",
            "top_k": 10,
            "min_score": -1.0,
            "rerank": False,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total_indexed"] == 0
    assert payload["candidates"] == []
    assert payload["reranked_with_cross_encoder"] is False


def test_ai_detection_rejects_empty_text(monkeypatch):
    client = app_module.app.test_client()

    response = client.post("/api/ai-detect", json={"text": "   "})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_ai_detection_rejects_non_string_text(monkeypatch):
    client = app_module.app.test_client()

    response = client.post("/api/ai-detect", json={"text": {"bad": "input"}})

    assert response.status_code == 400


def test_analyze_rejects_empty_document_before_processing(monkeypatch):
    monkeypatch.setattr(app_module, "get_models", _fake_models)
    client = app_module.app.test_client()

    response = client.post(
        "/api/analyze",
        json={"doc_a": "valid text", "doc_b": "   ", "threshold": 0.55},
    )

    assert response.status_code == 400
