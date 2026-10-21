# Audit remediation notes

This branch contains targeted fixes for the repository audit. The default branch is unchanged.

## Changes

- **Sentence alignment:** `align_passages` now greedily assigns highest-scoring sentence pairs one-to-one. A sentence in Document B cannot be counted as a match for multiple sentences in Document A. Ties are deterministic by source order after ranking.
- **Threshold validation:** the alignment function rejects values outside the cosine-score range, and the Flask API validates that the user threshold is numeric and in [0, 1].
- **Document coverage:** the document summary now accepts both sentence counts, returns coverage for A and B, caps coverage at 1.0, and computes the aggregate using the smaller coverage. The existing `matched_sentence_ratio` key is retained for compatibility.
- **Request size and input checks:** the Flask analysis and AI detection endpoints reject non-text/empty inputs and documents exceeding `VERITEXT_MAX_DOCUMENT_CHARS` (default 200,000 characters). Invalid threshold values return HTTP 400. Oversized inputs return HTTP 413.
- **Health endpoint:** readiness is based on core preprocessing/SBERT initialization and the response reports initialization state for each model component; an uninitialized service returns HTTP 503.
- **AI detection endpoint:** added `POST /api/ai-detect` accepting JSON `{"text": "..."}`. It is separate from duplicate detection. If checkpoints are absent or untrained, it returns HTTP 503 instead of presenting a made-up negative result.
- **AI checkpoint detection:** checkpoint detection now requires the checkpoint directory and a supported model-weight file to exist. Added the missing `Any` typing import.
- **AI detector evaluation:** threshold learning uses a seeded random validation sample rather than the first 120 rows, and final evaluation now uses the entire held-out test split instead of a maximum of 250 rows.
- **Regression tests:** added tests for one-to-one matching, symmetric document score, empty input summary, and invalid threshold rejection.

## Validation

The source changes and test cases were reviewed, but this branch has not been executed in a provisioned environment as part of this GitHub-only edit. Run the following from `Plagarism_Detection/` after installing the project requirements:

```bash
pytest -q tests/test_similarity_engine.py
python -m compileall app.py semantic_integrity train_ai_detector.py
```

The AI detector's full training/evaluation script is computationally expensive and requires the source datasets and model downloads. Its new full-test evaluation can take longer than the previous 250-example sample.

## Remaining limitations

- Greedy one-to-one matching is a practical deterministic assignment, not a globally optimal bipartite assignment. If global optimality is required, evaluate a Hungarian assignment implementation.
- The semantic benchmark remains a sentence-pair benchmark and must not be represented as whole-essay plagiarism accuracy.
- The FAISS demo corpus remains sample data unless a real corpus index is built and loaded.
- The AI detector remains a separate endpoint; it is not yet merged into the pairwise duplicate score or rendered as a new browser UI panel.
- Synthetic pair generation and cross-source leakage require a separate dataset audit. These fixes do not establish real-world academic-misconduct accuracy.
- Cross-Encoder outputs and the blended score are not calibrated probabilities of plagiarism. They should be described as model scores unless calibrated against a representative labelled dataset.
- Production authentication, authorization, rate limiting, corpus refresh/versioning and deployment security still need environment-specific design and testing.
