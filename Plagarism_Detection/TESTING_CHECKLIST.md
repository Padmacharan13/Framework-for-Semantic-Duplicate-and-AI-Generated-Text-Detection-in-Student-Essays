# Final Validation Checklist

Run from `Plagarism_Detection/`.

## A. Syntax and regression tests

```powershell
python -m compileall app.py semantic_integrity train_ai_detector.py tests
pytest -q
```

## B. API smoke tests

Keep the Flask server running with `python app.py`, then run in a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
Invoke-RestMethod http://127.0.0.1:5000/api/presets
```

For validation errors, use PowerShell's `Invoke-WebRequest` with JSON bodies and confirm the expected HTTP status:
- `POST /api/faiss/search` with blank query: 400
- `POST /api/faiss/search` with `top_k=0`: 400
- `POST /api/analyze` with empty Document B: 400
- `POST /api/ai-detect` with blank text: 400

A valid `/api/ai-detect` call may return 503 if a trained checkpoint is unavailable. This is an expected setup error, not a test failure if documented.

## C. Current evaluation

```powershell
python -u .\evaluation\evaluate_pipeline.py --top-k 10
```

Check that output has no missing-ID warnings and verify that `evaluation/results/retrieval_per_query.csv` and `evaluation/results/pair_scores.csv` contain all expected rows.

## D. Release criteria

Do not state that the system is validated for real-world academic plagiarism until:
- the labels have a documented human annotation process,
- the test set is held out by source document,
- metric results are reproduced from a recorded commit and corpus/model version,
- and runtime/latency tests have been executed on the intended hardware.
