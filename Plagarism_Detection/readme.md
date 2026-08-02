# Week 1 — Foundations + Domain Dataset (Hugging Face edition)

## Setup (one time)
```bash
pip install -r ../requirements.txt
python -m spacy download en_core_web_sm
```

No manual dataset downloads needed. MSRP and QQP are pulled directly from
the Hugging Face Hub the first time you run the code:

- **MSRP** → `datasets.load_dataset("glue", "mrpc")`
- **QQP** → `datasets.load_dataset("glue", "qqp")`

The first run downloads and caches both under `~/.cache/huggingface/datasets`
(a few hundred MB total). Every run after that loads instantly from the
local cache with no network call.

> If you're behind a restrictive firewall/proxy and the download fails, set
> `HF_ENDPOINT` or use `huggingface-cli login` / a mirror as needed — this
> only affects the very first download.

## Folder layout
```
project/
  requirements.txt
  data/
    processed/        <- created automatically, holds Week 1 output
    domain/            <- created automatically, holds your hand-written set
  week1/
    config.py
    preprocessing.py
    data_loader.py
    build_domain_dataset.py
    main_week1.py
```

## Run order

1. **Sanity-check preprocessing:**
   ```bash
   python preprocessing.py
   ```
   Should print cleaned text, sentences, and lemmas for two sample sentences.

2. **Pull + inspect MSRP and QQP from Hugging Face:**
   ```bash
   python data_loader.py
   ```
   First run will show a download progress bar for both datasets, then print
   row counts per source and a preview of the standardized table.

3. **Build your domain-specific paraphrase mini-set** (this is the part that
   makes your evaluation credible for essay-style text instead of just news/Quora):
   ```bash
   python build_domain_dataset.py generate
   ```
   Open `data/domain/domain_paraphrase_seed.csv` in Excel/Sheets. Each group
   member fills in their assigned `rewrite_text` cells by hand (write 3
   intensities per source paragraph — light paraphrase, heavy paraphrase,
   structural rewrite). Then run:
   ```bash
   python build_domain_dataset.py validate
   ```
   Fix any warnings it prints, re-run until it says "All rows valid."

4. **Run the full Week 1 pipeline:**
   ```bash
   python main_week1.py
   ```
   This loads MSRP + QQP from Hugging Face, your domain set from disk,
   batch-preprocesses all of them, and writes three parquet files into
   `data/processed/`. Those parquet files are what Week 2's embedding/
   retrieval code will load — don't rename them.

## Notes on the Hugging Face swap
- `HF_MSRP_SPLITS` includes `train`, `validation`, and `test` (GLUE MRPC's
  test split does ship with labels, unlike most GLUE tasks).
- `HF_QQP_SPLITS` only includes `train` and `validation` — GLUE's official
  QQP `test` split has no labels (`label = -1`, reserved for the leaderboard),
  so it's intentionally excluded.
- `QQP_SAMPLE_SIZE` in `config.py` subsamples QQP down from ~400k+364k rows
  for faster iteration during development. Set it to `None` in `config.py`
  if you want the full dataset for your final evaluation run.

## What "done" looks like for Week 1
- [ ] `python data_loader.py` runs without errors and shows row counts for both `msrp` and `qqp`
- [ ] `data/processed/msrp_processed.parquet` exists and has `text_a_sentences` populated
- [ ] `data/processed/qqp_processed.parquet` exists
- [ ] `data/domain/domain_paraphrase_labeled.csv` exists with no validation warnings
- [ ] `data/processed/domain_processed.parquet` exists
- [ ] You can explain, in one sentence, why the domain dataset matters (MSRP/QQP
      are news headlines and Quora questions, not academic essays)