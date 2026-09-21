# Emoji-Aware Sentiment Analysis & Emotion Classification

> **Research Question:** *"Do emojis carry sentiment and emotional signal that words miss?"*

---

## 📌 Executive Summary for Collaborators & Evaluators

This repository contains the complete experimental and evaluation framework for studying the interaction between text representations and emoji embeddings in social media.

### 1. What This Repository Has Completed (Phase 1: Controlled Sentiment Study)
We designed and completed **six strictly controlled deep learning experiments (E0–E5)** on **$11,966$ canonical test posts** from StockTwits to isolate the impact of emoji embeddings, initialization strategies, and fusion mechanisms against a frozen BERT-base baseline:

| Model | Fusion Architecture | Emoji Embedding Init | Test Acc | Macro F1 | Key Research Finding |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **E0** | Text-Only Baseline (Frozen BERT) | *None (Emoji-blind)* | 47.69% | 46.21% | Control baseline using words only. |
| **E1** | Concatenation (Mean Pool) | Random (Trainable) | 44.83% | 43.35% | **Degrades ($-2.86$ pp F1)**: Naive emoji concat adds noise. |
| **E2** | Concatenation (Mean Pool) | TweetEval Pretrained (Frozen) | 52.05% | 50.41% | **$+4.20$ pp F1**: Pretraining prevents concatenation noise. |
| **E3** 🏆 | **Text-Conditioned Attention** | **Random (Trainable)** | **54.20%** | **51.60%** | **WINNER ($+5.39$ pp F1)**: Attention dynamically weights emojis. |
| **E4** | Text-Conditioned Attention | TweetEval Pretrained (Frozen) | 52.92% | 50.53% | $+4.32$ pp F1: Overfit to Twitter semantics vs finance slang. |
| **E5** 🛡️ | **32-Dimensional Gated Fusion** | TweetEval Pretrained (Frozen) | 52.33% | 50.91% | **Highest Bearish/Negative Recall ($49.81\%$)**; best risk detector. |

---

## ❓ FAQ: Will a Collaborator Be Able to Work Without the Checkpoints?

> ### **YES, 100% YES.**
> Because `.pt`, `data/`, and `results/` are gitignored, your friend can pull this repository and immediately:
> 1. **Run Smoke Tests & Validate Architectures** locally in under 15 seconds.
> 2. **Reproduce Any Model Training (E0–E5)** from scratch using the deterministic pipeline (`seed=42`).
> 3. **Run the Streamlit Dashboard & Analysis Notebook** (using either lightweight dummy runs or trained heads).
> 4. **Implement the New Emotion-Analysis Architecture** requested by your professor.

---

## 🎯 Teacher's Feedback & Proposed Project Pivot: From 3-Class Sentiment to Fine-Grained Emotion Analysis

### What the Teacher Pointed Out:
1. **Low Baseline Accuracy (~50–54%)**:
   - In social media, 3-class classification (**Positive / Negative / Neutral**) is notoriously noisy. Sarcasm (e.g., *"Great job losing my money 🤡💸"*) or mixed market opinions confuse coarse sentiment models.
2. **Coarse Sentiment Lacks Nuance**:
   - Classifying a sentence simply as "Positive" or "Negative" misses the actual psychological human intent. 
   - **The Teacher's Mandate:** The model should identify the **specific emotion** the person is trying to convey:
     - 😃 **Happiness / Excitement / Hype** (e.g., `🚀`, `🔥`, `🎉`)
     - 🙃 **Sarcasm / Irony / Mockery** (e.g., `🤡`, `🙃`, `😂`, `💀`)
     - 😢 **Sadness / Despair / Grief** (e.g., `😭`, `📉`, `💔`)
     - 😡 **Anger / Frustration** (e.g., `🤬`, `😤`, `💩`)
     - 😰 **Anxiety / Fear / Panic** (e.g., `😰`, `😬`, `👀`)
     - ⚖️ **Curiosity / Skepticism / Neutral Inquiry** (e.g., `🤔`, `🤷‍♂️`)

---

## 🛠️ Step-by-Step Roadmap for Your Friend: How to Build the Emotion Classifier

Here is the exact technical blueprint for your collaborator to implement the teacher's emotion classification model using our existing E3 attention and E5 gated fusion code:

### Step 1: Dataset Acquisition for Emotion Analysis
Instead of StockTwits (which only has Bullish/Bearish labels), use one of these standard open-source multi-emotion benchmark datasets:
- **GoEmotions (Google Research)**: 58,000 Reddit comments labeled with 27 fine-grained emotions (or 6 Ekman emotions: Joy, Sadness, Anger, Fear, Surprise, Love). Available on Hugging Face: `datasets.load_dataset("google-research-datasets/go_emotions")`.
- **TweetEval (Emotion)**: 4 emotion classes (Anger, Joy, Optimism, Sadness). Available on Hugging Face: `datasets.load_dataset("cardiffnlp/tweet_eval", "emotion")`.
- **SemEval-2018 Task 1 (Affect in Tweets)**: Multi-label emotion intensity in tweets with rich emoji usage.

### Step 2: Adapt the Vocabulary & Extraction
1. Use our existing Unicode-aware emoji extractor in [`src/data/preprocessing.py`](src/data/preprocessing.py):
   ```python
   from src.data.preprocessing import extract_emojis, remove_emojis
   
   text_no_emoji = remove_emojis(raw_tweet)
   emojis = extract_emojis(raw_tweet)
   ```
2. Build an emoji vocabulary using [`src/embeddings/emoji_encoder.py`](src/embeddings/emoji_encoder.py).

### Step 3: Upgrade the Output Head
In [`src/models/e3_attention_fusion.py`](src/models/e3_attention_fusion.py), update `num_labels`:
```python
# Change num_labels from 3 to N emotions (e.g., 6 emotions)
model = AttentionFusionModel(
    model_name="bert-base-uncased", # Or "cardiffnlp/twitter-roberta-base-emotion"
    text_dim=768,
    emoji_dim=32,
    num_labels=6,  # 0: Joy, 1: Sadness, 2: Anger, 3: Fear, 4: Sarcasm, 5: Surprise
    freeze_encoder=True
)
```

### Step 4: Boost Accuracy Beyond 54% (How to Solve the "Low Accuracy" Problem)
Explain to your friend that our Phase 1 model had ~54% accuracy because BERT was **completely frozen** and only a small 2-layer MLP was trained. To achieve **75%–85%+ accuracy** on emotion classification:
1. **Unfreeze Top BERT Layers**: Fine-tune the top 2 transformer layers of BERT with a small learning rate (`lr=2e-5`) rather than freezing all weights.
2. **Use a Twitter-native Backbone**: Replace generic `bert-base-uncased` with `vinai/bertweet-base` or `cardiffnlp/twitter-roberta-base-emotion`. These backbones already understand social media abbreviations, slang, and syntax.
3. **Multi-Task Emotion + Emoji Loss**: Train the model to jointly predict both the emotion label and the masked emoji to enforce strong multi-modal grounding.

---

## 💻 Quickstart Commands for Collaborators

### 1. Environment Setup (Windows / Linux / Mac)
```bash
# Clone the repository
git clone https://github.com/Chetnapadhi/SentimentAnalysis.git
cd SentimentAnalysis

# Create and activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Verify Architecture & Contracts (Smoke Tests)
Run the automated test suite to ensure the attention and gated modules run cleanly on CPU without errors:
```bash
python tests/smoke_test_e3_e4_e5.py
python smoke_test_e0.py
```
*(Expected output: `ALL SMOKE TESTS PASSED`)*

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** to interact with the live inference analyzer and side-by-side comparisons.

### 4. Run Model Training (Optional - Requires GPU / Colab)
To train or re-evaluate any specific experiment:
```bash
# Example: Train E3 Attention model
python -m src.train_e3
# Or on Google Colab with guarded flag:
RUN_E3=1 python run_e3.py
```

---

## 📁 Repository Structure

```
├── app.py                         # Streamlit Interactive Dashboard entrypoint
├── config.yaml                    # Master project configuration (hyperparameters, seeds)
├── requirements.txt               # Locked dependencies
├── dashboard/                     # Modular dashboard interface
│   ├── components.py              # Reusable UI widgets & headers
│   ├── inference.py               # Live model inference & tokenization
│   ├── styles.py                  # High-contrast custom styling & theme tokens
│   ├── visualizations.py          # Plotly distribution charts & heatmaps
│   └── pages/                     # Clean 5-view presentation dashboard
├── models/
│   └── emoji_embeddings/          # Extracted emoji vocabulary (emoji_vocab.json)
├── notebooks/                     # Google Colab & reproducible experiment notebooks
│   ├── 00_run_e0.ipynb            # Text-only baseline
│   ├── 01_E1_Concat_Fusion.ipynb  # Naive concatenation
│   ├── 02_e2_pretrained_emoji.ipynb
│   ├── 03_e3_attention_random.ipynb # Winning attention model
│   ├── 04_e4_attention_pretrained.ipynb
│   ├── 05_e5_gated_pretrained.ipynb
│   └── 06_final_analysis.ipynb    # Consolidated evaluation & visualizations
├── src/
│   ├── common_pipeline.py         # Shared featurization, caching & training loop
│   ├── data/                      # Adapters, unicode emoji extraction & deduplication
│   ├── embeddings/                # Emoji vocabulary builder & encoders
│   ├── models/                    # PyTorch architectures (E0 text, E3 attention, E5 gated)
│   └── analysis/                  # Metrics aggregation & transition matrices
└── tests/
    └── smoke_test_e3_e4_e5.py     # Automated architecture integrity tests
```

---

## 📜 Scientific Citation & Reproducibility Policy

- **Fixed Seed:** All experiments run deterministically with `seed=42`.
- **Data Integrity:** Dataset splits are isolated into train/val/test; no test records are used for vocabulary building or parameter tuning.
- **Fair Evaluation:** All models (E0 through E5) are scored against the exact same test instances with standardized Macro F1 and class-wise precision/recall metrics.