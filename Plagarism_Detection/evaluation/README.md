# Plagiarism evaluation starter dataset

This starter contains 14 queries and 70 query-document labels (14 queries × 5 known document IDs).

## Files
- `retrieval_queries.csv`: relevance labels for retrieval metrics.
- `pair_labels.csv`: pair labels for the API's hybrid-confidence classification metrics.
- `README.md`: run instructions and label caveats.

## Classes included
- Exact copied sentence
- Manually written paraphrase of a known passage
- Same-topic, independently written text (hard negatives)

## Critical limitations
1. These are synthetic starter labels constructed from text previously shown by the API, not independently adjudicated real-world plagiarism labels.
2. The paraphrases are deliberately written as examples and should be reviewed by you or another annotator before reporting research results.
3. A label of 1 means the query is intended to represent copied/paraphrased material from the target document. It does not establish who copied from whom or prove misconduct.
4. Retrieval relevance is different from plagiarism. A same-topic independent query can be relevant to a document for search while having `is_plagiarized=0` in `pair_labels.csv`.
5. Some indexed documents (ESSAY_001, ESSAY_002, ESSAY_006, ESSAY_008, ESSAY_009) were not included in pair labels because their full text was not available in the conversation. Add them only after verifying their actual content.
6. Do not present metrics from this small synthetic set as final benchmark results. Use it for pipeline debugging, then create a larger independently reviewed dataset with a held-out test split.

## Run
Copy these files into the project's `evaluation` folder, replacing the earlier template CSV files. Start Flask, then run from the project root:

```powershell
python .\evaluation\evaluate_pipeline.py --top-k 10
```

The script compares `rerank=false` against `rerank=true`, then computes pair metrics at the requested hybrid-confidence threshold (default 0.5). The exploratory best threshold is computed on the same data and is not an unbiased final estimate.
