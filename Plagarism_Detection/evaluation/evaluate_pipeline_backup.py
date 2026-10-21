#!/usr/bin/env python
"""
Evaluate the plagiarism-search API against manually labelled ground truth.

Expected API:
  POST /api/faiss/search
  JSON: {"query": "...", "top_k": 10, "rerank": true|false}
  Response includes a "candidates" list with "doc_id", "cross_encoder_score",
  "hybrid_confidence", and optionally ranking fields.

IMPORTANT:
- Fill the CSV files with real labels. Do not invent labels from model scores.
- Retrieval metrics assess whether known relevant documents are retrieved.
- Pair labels assess plagiarism only if human-reviewed ground truth says whether
  that source/query and candidate document contain copied/paraphrased material.
- A score threshold selected on this same dataset is exploratory and optimistic.
  Select thresholds on validation data and report final metrics on a separate test set.
"""

import argparse
import csv
import json
import math
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def post_json(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e

def parse_ids(raw):
    return {x.strip() for x in (raw or "").replace(",", ";").split(";") if x.strip()}

def dcg(relevances):
    return sum((1.0 / math.log2(i + 2)) for i, rel in enumerate(relevances) if rel)

def retrieval_metrics(ranked_ids, relevant_ids, k):
    top = ranked_ids[:k]
    hits = [1 if doc_id in relevant_ids else 0 for doc_id in top]
    precision = sum(hits) / k if k else 0.0
    recall = sum(hits) / len(relevant_ids) if relevant_ids else float("nan")
    rr = next((1.0 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
    ideal = [1] * min(len(relevant_ids), k)
    ideal_dcg = dcg(ideal)
    ndcg = dcg(hits) / ideal_dcg if ideal_dcg else float("nan")
    return precision, recall, rr, ndcg

def fmt(x):
    return "N/A" if isinstance(x, float) and math.isnan(x) else f"{x:.4f}"

def mean(values):
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")

def classification_metrics(labels, scores, threshold):
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, preds))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, preds))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, preds))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, preds))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def main():
    parser = argparse.ArgumentParser(description="Evaluate the plagiarism detection search API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--retrieval-csv", default="evaluation/retrieval_queries.csv")
    parser.add_argument("--pairs-csv", default="evaluation/pair_labels.csv")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Must be large enough to include all indexed docs for pair-score evaluation.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Provisional hybrid_confidence threshold; tune on validation data only.")
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    endpoint = args.base_url.rstrip("/") + "/api/faiss/search"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_path = Path(args.retrieval_csv)
    pairs_path = Path(args.pairs_csv)
    retrieval_rows = read_csv(retrieval_path) if retrieval_path.exists() else []
    pair_rows = read_csv(pairs_path) if pairs_path.exists() else []

    # Cache one API result per query and rerank setting.
    cache = {}

def get_results(query, rerank, include_all=False):
    key = (query, rerank, include_all)

    if key not in cache:
        payload = {
            "query": query,
            "top_k": args.top_k,
            "rerank": rerank,
        }

        if include_all:
            payload["min_score"] = -1.0

        response = post_json(endpoint, payload)

        if "error" in response:
            raise RuntimeError(
                f"API returned error for query {query!r}: "
                f"{response['error']}"
            )

        cache[key] = response.get("candidates", [])

        total = response.get("total_indexed")
        if total is not None and args.top_k < int(total):
            print(
                f"WARNING: top_k={args.top_k}, "
                f"but total_indexed={total}. "
                "Increase --top-k."
            )

    return cache[key]

    # Retrieval ranking metrics, separately for the bi-encoder and reranked output.
    if retrieval_rows:
        metric_rows = []
        for row in retrieval_rows:
            query = (row.get("query") or "").strip()
            relevant = parse_ids(row.get("relevant_doc_ids", ""))
            if not query or not relevant:
                continue
            for mode, rerank in (("bi_encoder", False), ("reranked", True)):
                candidates = get_results(query, rerank)
                ranked_ids = [str(c.get("doc_id", "")) for c in candidates]
                for k in (1, 3, 5):
                    p, r, rr, ndcg = retrieval_metrics(ranked_ids, relevant, k)
                    metric_rows.append({
                        "query": query, "mode": mode, "k": k,
                        "precision_at_k": p, "recall_at_k": r,
                        "reciprocal_rank": rr, "ndcg_at_k": ndcg,
                    })
        if metric_rows:
            with open(out_dir / "retrieval_per_query.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=metric_rows[0].keys())
                writer.writeheader()
                writer.writerows(metric_rows)
            print("\nRETRIEVAL METRICS (averaged over labelled queries)")
            print("mode        K   Precision@K  Recall@K    MRR       nDCG@K")
            for mode in ("bi_encoder", "reranked"):
                for k in (1, 3, 5):
                    subset = [x for x in metric_rows if x["mode"] == mode and x["k"] == k]
                    if subset:
                        print(f"{mode:11} {k:<3} {fmt(mean([x['precision_at_k'] for x in subset])):>12} "
                              f"{fmt(mean([x['recall_at_k'] for x in subset])):>10} "
                              f"{fmt(mean([x['reciprocal_rank'] for x in subset])):>9} "
                              f"{fmt(mean([x['ndcg_at_k'] for x in subset])):>10}")
            print(f"Per-query metrics saved to: {out_dir / 'retrieval_per_query.csv'}")
        else:
            print("No usable retrieval rows. Add query and relevant_doc_ids labels.")
    else:
        print(f"No retrieval labels found at {retrieval_path}; skipping retrieval metrics.")

    # Pair-level labels: source/query text paired with a candidate doc ID and human label.
    if pair_rows:
        labels, hybrid_scores, cross_scores, detail_rows = [], [], [], []
        for row in pair_rows:
            query = (row.get("query") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            label_raw = (row.get("is_plagiarized") or "").strip()
            if not query or not doc_id or label_raw not in {"0", "1"}:
                continue
            candidates = get_results(
    query, True, include_all=True
)
            found = next((c for c in candidates if str(c.get("doc_id", "")) == doc_id), None)
            if found is None:
                print(f"WARNING: {doc_id} not returned for query {query!r}; "
                      "increase --top-k or check the document ID.", file=sys.stderr)
                continue
            label = int(label_raw)
            hybrid = float(found.get("hybrid_confidence", 0.0) or 0.0)
            cross = float(found.get("cross_encoder_score", 0.0) or 0.0)
            labels.append(label)
            hybrid_scores.append(hybrid)
            cross_scores.append(cross)
            detail_rows.append({
                "query": query, "doc_id": doc_id, "is_plagiarized": label,
                "cross_encoder_score": cross, "hybrid_confidence": hybrid,
            })

        if detail_rows:
            with open(out_dir / "pair_scores.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
                writer.writeheader()
                writer.writerows(detail_rows)
            print("\nPAIR CLASSIFICATION (using hybrid_confidence)")
            print("These metrics are meaningful only if labels are human-reviewed ground truth.")
            report = classification_metrics(labels, hybrid_scores, args.threshold)
            print(f"Threshold={args.threshold:.3f}  Precision={report['precision']:.4f}  "
                  f"Recall={report['recall']:.4f}  F1={report['f1']:.4f}  "
                  f"Accuracy={report['accuracy']:.4f}")
            print(f"Confusion matrix counts: TP={report['tp']} FP={report['fp']} "
                  f"TN={report['tn']} FN={report['fn']}")

            # Exploratory threshold search. Do not report this as final test performance.
            thresholds = sorted({0.0, 1.0, args.threshold} |
                                {round(i / 100, 2) for i in range(1, 100)})
            candidates = [classification_metrics(labels, hybrid_scores, t) for t in thresholds]
            best = max(candidates, key=lambda x: (x["f1"], x["precision"]))
            print(f"Exploratory best threshold on these same labelled rows: {best['threshold']:.2f} "
                  f"(F1={best['f1']:.4f}, precision={best['precision']:.4f}, "
                  f"recall={best['recall']:.4f}). This is not an unbiased test score.")
            print(f"Pair scores saved to: {out_dir / 'pair_scores.csv'}")
        else:
            print("No usable pair labels. is_plagiarized must be 0 or 1.")
    else:
        print(f"No pair labels found at {pairs_path}; skipping pair classification metrics.")

    print("\nReminder: search relevance is not proof of plagiarism. Use human-reviewed labels,")
    print("separate validation/test sets, and report test metrics only after threshold tuning.")

if __name__ == "__main__":
    main()
