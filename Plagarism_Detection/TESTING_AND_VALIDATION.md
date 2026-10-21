# Testing and Validation Report

**Project:** Framework for Semantic Duplicate and AI-Generated Text Detection in Student Essays  
**Application directory:** `Plagarism_Detection/`  
**Report status:** Test plan and currently reported exploratory results; complete held-out validation is still pending.

## 1. Scope

The system exposes a Flask application for pairwise document analysis, FAISS candidate retrieval, Cross-Encoder reranking, and a separate AI-authorship detection endpoint. The architecture uses SBERT embeddings for semantic representation, a FAISS inner-product index for candidate retrieval, and a Cross-Encoder to rerank query-document candidates. The AI-authorship endpoint is distinct from duplicate detection and must be evaluated independently.

This report distinguishes:
- **Repository-reviewed evidence:** source code, existing unit tests, and evaluation metadata inspected in the repository.
- **Locally reported results:** output pasted from a developer's Windows terminal. These were not rerun by this report.
- **Not yet established:** results that require execution against the actual model artifacts, source corpus, and independently annotated test data.

## 2. Reproducible local commands

Run commands from `Plagarism_Detection/` after activating the project virtual environment and installing requirements.

### Static syntax check

```powershell
python -m compileall app.py semantic_integrity train_ai_detector.py tests
```

### Unit and API validation tests

```powershell
pytest -q
```

### Evaluation API

Start the application in Terminal 1:

```powershell
python app.py
```

Then, in Terminal 2:

```powershell
python -u .\evaluation\evaluate_pipeline.py --top-k 10
```

For reproducible results, retain the console output, the exact commit SHA, the selected model configuration, corpus/index version, and hashes/version identifiers for label files.

## 3. Automated test coverage

The repository's original `tests/test_similarity_engine.py` covers:
1. One-to-one sentence alignment: a sentence from Document B must not be counted as the match for multiple sentences in Document A.
2. Symmetry of the document-level aggregate when the two documents are swapped.
3. Empty-document summary behavior.
4. Rejection of an out-of-range alignment threshold.

The additional `tests/test_api_validation.py` covers:
1. Empty and non-string FAISS query inputs.
2. Invalid `top_k`, non-Boolean `rerank`, and out-of-range `min_score`.
3. A valid FAISS search request and expected JSON response shape using stubbed models.
4. Empty and non-string AI-detection input.
5. Empty document handling in the pairwise analysis endpoint.

These are input/logic tests and use stubs; they do not prove that real transformer checkpoints load correctly or that the service performs accurately on real essays.

## 4. Exploratory retrieval results reported from local execution

The developer provided the following results from `python -u .\evaluation\evaluate_pipeline.py --top-k 10`:

| Metric | Bi-Encoder | Bi-Encoder + Cross-Encoder |
|---|---:|---:|
| Precision@1 | 0.8571 | 0.9286 |
| Recall@3 | 1.0000 | 1.0000 |
| MRR | 0.9286 | 0.9524 |
| nDCG@3 | 0.9473 | 0.9643 |

The reranked system reports higher Precision@1, MRR, and nDCG@3 on the labelled query set, while Recall@3 is unchanged. These figures are **reported local results**, not independently reproduced by this report. Results averaged over 14 queries are preliminary and should not be generalized without a held-out dataset.

The local run also reported pair classification at threshold 0.50:
- TP = 10, FP = 0, TN = 60, FN = 0
- Precision, Recall, F1, Accuracy = 1.0000

This is not accepted as final benchmark performance. The repository's `evaluation/README.md` explicitly states its starter labels are synthetic and not independently adjudicated real-world plagiarism labels. A perfect score on such a small constructed set may reflect the way examples and labels were generated. Do not claim 100% real-world plagiarism detection accuracy from this run.

## 5. Important evaluation caveats

### Relevance is not plagiarism

Retrieval relevance asks whether a document is useful or related to the query. Plagiarism labels ask whether material has been copied or paraphrased in a way that meets an explicitly defined annotation policy. Same-topic independent writing can be relevant for retrieval while correctly receiving a negative plagiarism label.

### Scores are not calibrated probabilities

The Cross-Encoder output and the blended `hybrid_confidence` value are model scores. Unless calibrated against an appropriate representative validation set, do not describe them as probabilities that plagiarism occurred.

### Avoid leakage

Split by source essay/document rather than by individual sentence pair. Closely related paraphrases and source passages must not appear across training, validation, and test splits. Choose thresholds on validation data only; report final metrics once on a held-out test split.

### AI authorship is a separate task

AI-generated-text detection is not the same as semantic duplicate/plagiarism detection. Evaluate the detector independently with human-curated human-written and AI-written examples from relevant domains, reporting class counts, precision, recall, F1, confusion matrix, and calibration where possible. AI-detection outputs must not be treated as conclusive proof of authorship.

## 6. Remaining tests before a final research claim

- [ ] Run `python -m compileall app.py semantic_integrity train_ai_detector.py tests`.
- [ ] Run `pytest -q` and retain the actual output.
- [ ] Smoke-test the running Flask app: `/api/health`, `/api/presets`, `/api/analyze`, `/api/faiss/search`, and `/api/ai-detect`.
- [ ] Confirm the index contains the expected corpus documents and that every evaluation target ID exists in its metadata.
- [ ] Build a human-reviewed, documented test set with positive and hard-negative pairs.
- [ ] Separate train/validation/test by source document; tune thresholds only on validation data.
- [ ] Re-run retrieval metrics and pair-classification metrics on the held-out test set.
- [ ] Measure p50/p95 latency, throughput under concurrency, memory footprint, and cold-start model load time on the target machine.
- [ ] Record model IDs, dependency versions, dataset versions, commit SHA, and random seeds.
- [ ] Verify AI-detector checkpoint is actually trained and evaluate it separately on a held-out split.

## 7. Test outcome status

This document does not claim all checks passed. Source review and the locally reported retrieval output are available; live model evaluation, the newly added test suite, independent annotation, deployment-scale performance, and held-out validation must be executed in the intended local environment before the project is described as fully validated.
