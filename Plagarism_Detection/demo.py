"""
demo.py
Interactive Web Application & CLI Launcher for Semantic Duplicate and Paraphrase Detection.

Run:
    python demo.py
This will automatically launch the Web Application server and open the browser at http://127.0.0.1:5000.

Optional CLI mode:
    python demo.py --cli
Runs a quick command-line batch demo and outputs HTML report to results/demo_report.html.
"""

from __future__ import annotations

import html
import logging
import sys
import threading
import time
import webbrowser

from config import DEMO_REPORT_HTML, WEB_HOST, WEB_PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DOCUMENT_A = """
Academic integrity relies heavily on the authentic submission of assignments and essays.
Digital submissions have scaled drastically, making manual comparison for duplicates
virtually impossible for instructors. There is an urgent need for automated pipelines
that can scan batches of student texts and index their underlying conceptual content.
Traditional string-matching methods fail when a student restructures an essay using
synonym substitution or syntax swapping while preserving the original meaning.
""".strip()

DOCUMENT_B = """
Instructors depend a great deal on students submitting their own original work for
courses to remain fair. As the number of digital submissions has grown enormously,
checking each one by hand for duplication is no longer realistic for teaching staff.
Traditional string-matching methods fail when a student restructures an essay using
synonym substitution or syntax swapping while preserving the original meaning.
Renewable energy sources such as solar power are becoming more cost effective every year.
""".strip()


def run_cli() -> None:
    from embeddings import Embedder
    from preprocessing import TextPreprocessor
    from similarity_engine import align_passages, document_level_summary

    logger.info("Running CLI Demo Mode...")
    preprocessor = TextPreprocessor()
    embedder = Embedder()

    proc_a, proc_b = preprocessor.process_column([DOCUMENT_A, DOCUMENT_B])
    matches = align_passages(embedder, proc_a.sentences, proc_b.sentences)
    summary = document_level_summary(matches, total_sentences_a=len(proc_a.sentences))

    print("\n=== DOCUMENT-LEVEL SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n=== MATCHED PASSAGES ===")
    for m in matches:
        print(f"[{m.tier.upper():<18}] sem={m.semantic_similarity:.2f} lex={m.lexical_similarity:.2f}")
        print(f"   A: {m.sentence_a}")
        print(f"   B: {m.sentence_b}\n")

    logger.info("CLI Demo finished.")


def open_browser():
    time.sleep(1.5)
    url = f"http://{WEB_HOST}:{WEB_PORT}"
    logger.info("Opening web dashboard in browser: %s", url)
    webbrowser.open(url)


def run_webapp() -> None:
    from app import app, get_models

    logger.info("Loading SBERT & spaCy models for Web Application...")
    get_models()

    threading.Thread(target=open_browser, daemon=True).start()

    logger.info("Starting VeriText AI Web Dashboard on http://%s:%s", WEB_HOST, WEB_PORT)
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        run_webapp()