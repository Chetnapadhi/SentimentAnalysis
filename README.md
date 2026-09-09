# Emoji-Aware Sentiment Analysis

**Do emojis carry signal that words miss?**

This project tests whether explicit emoji representations improve sentiment
classification beyond a controlled text-only baseline. The primary supervised
dataset is StockTwits with Bullish, Neutral, and Bearish labels.

## Dataset separation

The datasets are intentionally **not merged**:

- `ElKulako/stocktwits-emoji`: primary supervised sentiment experiment.
- `cardiffnlp/tweet_eval`, config `emoji`: auxiliary emoji-task representation learning.
- `bapcon/101k-emojis`: unlabeled YouTube comments for frequency, visualization, and optional external analysis.

The inspection script writes one report per source and never concatenates rows
or label spaces.

## Stage 1: inspect the datasets

From the project root, create or activate the environment and install the
dependencies:

```bash
python -m pip install -r requirements.txt
python -m src.data.inspect_datasets --dataset all
```

Reports are written to `data/inspection/stocktwits.json`,
`data/inspection/tweeteval_emoji.json`, and `data/inspection/youtube.json`.
They contain schemas, splits, label metadata, record counts, missing text,
duplicates, emoji presence, and unique emoji frequencies. These files are
generated artifacts and should not be treated as hand-written statistics.

### Stage 1 findings

The current loaded schemas are:

- StockTwits: `train` 211,758, `validation` 20,761, `test` 11,984; `text` only; 244,503 total records containing at least one emoji.
- 101K Emojis: `train` 101,354; fields include `text`, metadata, and engagement fields; no sentiment label.
- TweetEval Emoji: `train` 45,000, `test` 50,000, `validation` 5,000; `text` plus a 20-class emoji `label`.

### Stage 2 findings — StockTwits labels are RECOVERABLE

The sentiment labels ARE available in the HuggingFace repository as raw source
text files whose **filenames encode the class**. The HuggingFace dataset
builder only exposes the concatenated `text` column, but each row maps back to
a source file:

| Split | Bearish file | Neutral file | Bullish file |
|---|---|---|---|
| train | `train-emoji-bear-unmodified.txt` | `train-emoji-net-unmodified.txt` | `train-emoji-bull-unmodified.txt` |
| validation | `val_bear.txt` | `val_net.txt` | `val_bull.txt` |
| test | `test_emoji_bear.txt` | `test-emoji-net.txt` | `test-emoji-bull.txt` |

The adapter `src/data/stocktwits_adapter.py`:

1. Downloads the source files and builds `text -> label` lookup tables.
2. Matches every HuggingFace row to its label by text content (0 = Bearish, 1 = Neutral, 2 = Bullish).
3. Preserves the original train/validation/test split.
4. Drops rows with conflicting labels (a text appearing in more than one class file) rather than guessing.

#### Results of label recovery

| Split | HF rows | Labeled rows | Dropped (conflict) | Bearish | Neutral | Bullish |
|---|---|---|---|---|---|---|
| train | 211,758 | 210,699 | 1,059 | 35,843 | 63,593 | 111,263 |
| validation | 20,761 | 20,676 | 85 | 4,073 | 7,496 | 9,107 |
| test | 11,984 | 11,966 | 18 | 2,608 | 4,191 | 5,167 |
| **Total** | **244,503** | **243,341** | **1,162** | — | — | — |

Labeled CSVs are saved to `data/processed/stocktwits_{train,validation,test}.csv`.
The report is at `data/inspection/stocktwits_labels.json`.

> Reproduce with:
> ```bash
> python -m src.data.stocktwits_adapter --conflict-strategy drop
> ```

### Class imbalance note

The dataset is heavily imbalanced toward Bullish. This will be handled in the
training phase with class-weighted loss. Do not hide this imbalance.

### Data-integrity findings (Stage 2 validation)

A full validation pass was run (`src/data/validate_data.py`) and reported in
`data/validation/findings_report.json`. Key findings:

- **Label provenance: PASS.** Every retained row maps to exactly one documented
  source label value; zero unmatched rows.
- **Conflicting rows: genuine source artifact.** ~1,162 texts appear in more
  than one class file (e.g. `"buy the dip"` in both Neutral and Bearish files).
  These are kept at the `drop` behavior; no guessing or LLM used.
- **⚠️ Training duplication artifact (IMPORTANT).** The HF training split has
  210,699 rows but only **91,168 unique texts**. The dataset card documents the
  true training size as **91,758 observations** (57,932 bullish + 26,516
  neutral + 7,310 bearish). The extra ~119K training rows are duplicated texts.
  Training on the full 210,699 rows would over-count duplicates, so a decision
  on whether to deduplicate training is required before E0-E5.
- **⚠️ Validation↔test overlap.** 865 texts (7.2% of test) appear in both
  validation and test. This weakens independence between model selection and
  final evaluation and must be reported as a limitation.
- **Emoji extraction: PASS.** 100% of rows contain emojis; Unicode-aware
  extraction handles ZWJ (12,790), skin-tone (2) and variation-selector-16
  (27,748) sequences correctly. 1,758 unique emojis.
- **Class weights: train-only.** Inverse-frequency weights computed ONLY from
  the training distribution (Bearish 1.96, Neutral 1.10, Bullish 0.63).
  Validation/test are not rebalanced.
- **Colab feasibility: PASS.** Frozen BERT featurization ~7 minutes on a T4;
  cached embeddings ~617 MB; head training cheap. Full data is feasible.

### Final dataset construction (approved decisions)

The approved pipeline built the final versioned dataset via
`src/data/build_final_dataset.py` (`data/processed/experiment_manifest.json`):

1. **Training deduplicated** to unique text (210,699 → 91,121 rows after removing
   119,551 duplicate rows and 27 train↔test overlapping texts).
2. **Versioned files preserved** — originals kept as `*_raw.csv`, deduped as
   `stocktwits_train_dedup.csv`, final eval as `*_final.csv`.
3. **Class weights recomputed from final training only:**
   Bearish 4.17, Neutral 1.16, Bullish 0.53.
4. **Validation/test unchanged** per the evaluation policy below.

**Final split sizes:** train **91,121** · validation **20,676** · test **11,966**.

### Validation↔test overlap: evaluation policy

Investigation of the 865 validation↔test overlaps:

- **863/865 have identical labels**; only **2 differ** (`"buy the dip 🤣"`,
  `"shiba 🌎 shiba 🪐 ⭐️🦊"` — each appears in different class files across the
  two splits).
- The source data has **only a `text` column — no post ID or timestamp** to
  disambiguate records.
- The 865 overlapping texts each appear **exactly once** in the validation
  source AND once in the test source, so they are **distinct posts with
  identical wording** (very common in finance social media, e.g. "buy the dip",
  "to the moon"), not duplicated records.
- Validation (May 1–Jun 15, 2022) and test (Jun 16–30, 2022) are adjacent
  temporal windows per the dataset card, so post overlap in wording is expected.

**Recommended policy: A — keep validation/test unchanged, report the overlap as
a documented dataset limitation.** Because (a) the records are distinct posts
(not duplicated rows), (b) labels are 99.8% consistent across the two sets,
(c) no metadata exists to identify true underlying posts, and (d) removing them
would shrink and bias the evaluation set, the scientifically defensible choice
is to keep both sets and transparently report the ~7.2% wording overlap as a
limitation rather than silently drop records.

## Google Colab

Upload or clone this repository, then run:

```python
!pip install -r requirements.txt
!python -m src.data.inspect_datasets --dataset all --output-dir data/inspection
!python -m src.data.stocktwits_adapter --conflict-strategy drop
```

The first run downloads the datasets from Hugging Face and may take several
minutes. The exact observed schema and counts should be reviewed before model
implementation.

### GitHub / Google Drive / Colab Workflow

**GitHub** — source of truth for all code, config, notebooks, and small research
reports (`*.md`, `experiment_manifest.json`, `config.yaml`).

**Google Drive** — stores large experiment outputs:
- Model checkpoints (`*.pt`, `*.safetensors`)
- Embedding caches (`results/*/embeddings_cache/`)
- Full results directories (`results/E0/`, `results/E1/`)
- Logs

**Colab `/content`** — **ephemeral**. After training, copy results to Drive:

```bash
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Backup results
!cp -r results/E0 /content/drive/MyDrive/emoji-sentiment-results/
!cp -r results/E1 /content/drive/MyDrive/emoji-sentiment-results/
```

**Do NOT commit to GitHub:**
- `results/`, `data/raw/`, `data/processed/`, `data/inspection/`
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`)
- Embedding caches (`embeddings_cache/`, `results/*/embeddings_cache/`)
- Hugging Face cache (`~/.cache/huggingface/`, `.cache/`)
- Virtual environments (`.venv/`, `venv/`)

See `.gitignore` for full exclusion list.

### Reproducible Colab Workflow
1. Open `notebooks/00_colab_setup.ipynb` → Run all cells (clones repo, installs deps, verifies GPU)
2. Run experiment:
   - E0: `!RUN_E0=1 python run_e0.py`
   - E1: `!RUN_E1=1 python -m src.train_e1`
3. Backup results to Drive

Both experiments are **guarded** (`RUN_E0=1` / `RUN_E1=1`) so they cannot accidentally run.

## Planned experiments

The controlled sequence is E0 text-only, E1 random emoji embeddings with
concatenation, E2 pretrained emoji embeddings with concatenation, E3/E4
attention fusion, and E5 pretrained emoji embeddings with gated fusion. Every
experiment will use the same StockTwits train/validation/test records.

### E0 Official Result (T4 GPU, frozen BERT)
**Do not modify or recompute — this is the official baseline.**

| Metric | Value |
|---|---|
| Accuracy | 0.47693464816981446 |
| Macro Precision | 0.4613466600218872 |
| Macro Recall | 0.46850647140725094 |
| Macro F1 | 0.4621372050469918 |
| Best Val Macro F1 | 0.4637350022487232 |
| Best Epoch | 4 |

### Remaining experiments (not yet run)
E1–E5: No results claimed yet. Accuracy, macro-F1, ablations, plots, and research
findings will be populated only after actual evaluation.

## Current methodological decisions

- Seed: 42.
- Initial text encoder: frozen `bert-base-uncased`, configurable in `config.yaml`.
- Initial emoji aggregation: mean pooling over every extracted emoji.
- StockTwits labels are recovered from the source text files (filename-encoded classes) via `src/data/stocktwits_adapter.py`.
- Conflicting label texts (a text appearing in more than one class file) are **dropped**, never guessed. ~0.5% of rows affected.
- The YouTube dataset is not a sentiment dataset and will not be used as one without documented annotation.
- Gate values will be reported as interpretability signals, not causal explanations.


Step 1: Open Terminal (PowerShell or Command Prompt)
Navigate into your project root directory:

powershell


cd D:\DL
Step 2: Activate the Virtual Environment
powershell


# In PowerShell:
.\.venv\Scripts\Activate.ps1
# (Or if using standard CMD):
.\.venv\Scripts\activate.bat
Tip: If PowerShell gives an execution policy error on script activation, run this once: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Step 3: Run the Streamlit Dashboard
powershell


streamlit run app.py
(Or without activating manually, run directly with the virtualenv Python executable:)

powershell


.\.venv\Scripts\streamlit.exe run app.py
Step 4: Open in Browser
Once executed, the dashboard will open automatically or you can open: 👉 http://localhost:8501

Useful Extra Commands (Reference)
Stop the running server: Press Ctrl + C in the terminal.

Run on a specific port (if 8501 is busy):

powershell


.\.venv\Scripts\streamlit.exe run app.py --server.port 8502
Run the full analysis script standalone (reproducibility check):

powershell


.\.venv\Scripts\python.exe src/analysis/run_all_analysis.py