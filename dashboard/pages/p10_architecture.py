"""Page 10: Model Architecture & Mechanics."""

import streamlit as st
from dashboard.components import render_page_header


def render():
    render_page_header(
        title="Model Architectures",
        subtitle="Controlled architectural specifications and routing mechanisms for all six benchmarked configurations.",
    )

    tab_e0, tab_e12, tab_e34, tab_e5 = st.tabs([
        "E0: Text-Only Baseline",
        "E1 & E2: Concatenation Fusion",
        "E3 & E4: Attention Fusion",
        "E5: 32-d Gated Modulation",
    ])

    with tab_e0:
        st.markdown("### Architecture E0: Frozen BERT Text-Only Baseline")
        st.code(
            """
Input: "text_without_emoji" (N tokens)
      │
      ▼
┌──────────────┐
│  BERT-base   │ (768-d, frozen weights)
└──────────────┘
      │
      ▼
[Mean Pooling]  -> text_rep ∈ ℝ⁷⁶⁸
      │
      ▼
┌──────────────┐
│   MLP Head   │ Linear(768 -> 768) -> ReLU -> Dropout(0.1) -> Linear(768 -> 3)
└──────────────┘
      │
      ▼
Logits (Bearish, Neutral, Bullish)
            """,
            language="text",
        )
        st.markdown(
            """
            - **Strict Isolation:** `original_text` is stripped of all emoji characters before tokenization.
            - **Parameter Count:** Only the classification head is trained; BERT weights remain frozen.
            """
        )

    with tab_e12:
        st.markdown("### Architecture E1 & E2: Concatenation Fusion")
        st.code(
            """
Text Branch:                          Emoji Branch:
"text_without_emoji"                  ["🚀", "🔥"] (M tokens)
      │                                     │
      ▼                                     ▼
┌──────────────┐                      ┌──────────────┐
│  BERT-base   │ (Frozen)             │  Embedding   │ E1: Random Trainable
└──────────────┘                      └──────────────┘ E2: TweetEval Pretrained
      │                                     │
   text_rep (768-d)                    Mean-pooled emoji_rep (32-d)
      │                                     │
      └──────────────┬──────────────────────┘
                     ▼
            Concatenation (800-d)
                     │
                     ▼
         Linear(800 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 3)
                     │
                     ▼
                   Logits
            """,
            language="text",
        )
        st.markdown(
            """
            - **E1 vs E2 Pretraining Contrast:** In E1, random initialization acts as unregularized noise (-2.86 pp F1). In E2, pretraining provides essential regularization (+4.19 pp F1).
            """
        )

    with tab_e34:
        st.markdown("### Architecture E3 & E4: Text-Conditioned Attention Fusion")
        st.code(
            """
Text Branch:                          Emoji Branch:
"text_without_emoji"                  ["🚀", "🔥"] (M tokens)
      │                                     │
      ▼                                     ▼
┌──────────────┐                      ┌──────────────┐
│  BERT-base   │ (Frozen)             │  Embedding   │ E3: Random Trainable
└──────────────┘                      └──────────────┘ E4: TweetEval Pretrained
      │                                     │
   text_rep (768-d)                      emoji_emb (M × 32-d)
      │                                     │
      ├───────────┐                         │
      │           ▼                         ▼
      │      Q = Wq(text)             K = Wk(emoji), V = Wv(emoji)
      │           │                         │
      │           └───────────┬─────────────┘
      │                       ▼
      │             Scores = (Q @ K^T) / √32
      │             Alpha  = Softmax(Scores)
      │             Context = Alpha @ V (32-d)
      │                       │
      └───────────┬───────────┘
                  ▼
         Fused = [text; context] (800-d)
                  │
                  ▼
         Linear(800 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 3)
                  │
                  ▼
                Logits
            """,
            language="text",
        )
        st.markdown(
            """
            - **Zero Attention Dropout:** Attention weights strictly sum to 1 over valid emoji positions.
            - **Learned Fallback:** For zero-emoji instances, a learned parameter `emoji_absent` $\in \mathbb{R}^{32}$ is returned directly without computing softmax over empty tensors.
            """
        )

    with tab_e5:
        st.markdown("### Architecture E5: Pretrained Gated Modulation")
        st.code(
            """
Text Branch:                          Emoji Branch:
"text_without_emoji"                  ["🚀", "🔥"] (M tokens)
      │                                     │
      ▼                                     ▼
┌──────────────┐                      ┌──────────────┐
│  BERT-base   │ (Frozen)             │  TweetEval   │ (1610 × 32, Frozen)
└──────────────┘                      └──────────────┘
      │                                     │
   text_rep (768-d)                    Mean-pooled emoji_rep (32-d)
      │                                     │
      ├──────────────┬──────────────────────┤
      │              ▼                      │
      │       Concat [text; emoji] (800-d)  │
      │              │                      │
      │              ▼                      │
      │       Gate g = σ(Wg · [text; emoji] + bg) ∈ (0, 1)³²
      │              │                      │
      │              └──────────┬───────────┘
      │                         ▼
      │                 Gated_emoji = g ⊙ emoji_rep (32-d)
      │                         │
      └──────────────┬──────────┘
                     ▼
            Fused = [text; Gated_emoji] (800-d)
                     │
                     ▼
            Linear(800 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 3)
                     │
                     ▼
                   Logits
            """,
            language="text",
        )
        st.markdown(
            """
            - **Element-wise Gating:** Sigmoid gate $g$ modulates individual feature dimensions rather than sequence positions.
            """
        )
