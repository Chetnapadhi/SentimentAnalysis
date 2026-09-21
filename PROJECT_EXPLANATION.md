# Emoji-Aware Emotion Classification Project

## Teacher-Ready Project Explanation

This document explains what the project does, which algorithms were used, what was completed in Phase 1 and Phase 2, what the dashboard shows, and what is still incomplete.

The most important distinction is:

- **Phase 1** is a controlled 3-class financial sentiment study on StockTwits.
- **Phase 2** is the teacher-directed extension to fine-grained emotion classification using TweetEval Emotion and GoEmotions.

The current project should be presented primarily as an **emotion classification project with emoji-aware modeling**. Phase 1 is supporting background and a controlled baseline study.

---

## 1. Project Objective

### Original research question

> Do emojis carry sentiment and emotional signal that words miss?

A text-only model receives the sentence after emojis have been removed. An emoji-aware model receives:

1. The text representation.
2. The extracted emoji sequence.
3. A fusion mechanism that combines the two representations.

The project measures whether adding emoji information improves classification and whether different fusion algorithms behave differently.

### Teacher's requirement

The teacher identified two limitations in the original task:

1. **Positive/negative/neutral sentiment is too coarse.**
2. Social-media text can express more specific emotional states such as happiness, sadness, anger, fear, excitement, or surprise.

Therefore, Phase 2 changes the target from financial polarity to emotion categories.

Important accuracy statement:

- The teacher's examples mention emotions such as anxiety, sadness, happiness, anger, sarcasm, and curiosity.
- The implemented datasets do **not** contain exactly those six custom labels.
- The actual trained labels are:
  - GoEmotions Ekman-6: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`.
  - TweetEval Emotion: `anger`, `joy`, `optimism`, `sadness`.
- There is no separately trained `sarcasm` or `anxiety` class in the current checkpoints.
- `fear` is the closest available trained category to anxiety or panic.

This is scientifically important: the dashboard must not claim to predict a label that was never present in the training data.

---

## 2. Overall System Architecture

```text
Raw social-media sentence
          |
          v
Unicode-aware emoji extraction
          |
          +----------------------+
          |                      |
          v                      v
Text with emojis removed       Emoji sequence
          |                      |
          v                      v
BERT/RoBERTa text encoder      32-dimensional emoji embeddings
          |                      |
          +----------+-----------+
                     |
                     v
             Fusion mechanism
       concat / attention / gating
                     |
                     v
              MLP classifier
                     |
                     v
        Emotion or sentiment probabilities
```

The text branch and emoji branch are deliberately separated. This allows the experiments to test whether emoji information adds signal beyond the words.

---

## 3. Data Preprocessing

The shared preprocessing code is in `src/data/preprocessing.py` and the emotion adapters are in `src/data/emotion_adapter.py`.

### 3.1 Emoji extraction

The project uses Unicode-aware extraction rather than searching for a small hard-coded list. It extracts emoji sequences from the original text and stores them separately.

For every example, the canonical representation contains fields equivalent to:

```text
id
split
original_text
text_without_emoji
emoji_list
num_emojis
label
label_name
token_count
```

### 3.2 Text input

The text encoder receives `text_without_emoji`.

This is essential for a fair comparison:

- E0/EM0 is genuinely text-only.
- E3/EM3 and E5/EM5 receive the same emoji-free text branch plus a separate emoji branch.
- The emoji is not accidentally seen by both branches.

### 3.3 Emoji vocabulary

The emoji vocabulary is built from the training split only. This prevents evaluation information from leaking into vocabulary construction.

Each emoji is assigned an integer ID:

```text
0 = <UNK> or padding path
1...N = known training emojis
```

The maximum number of emojis per example is 8. Longer sequences are truncated and shorter sequences are padded with a mask.

### 3.4 Class imbalance

The training pipeline computes inverse-frequency class weights from the training split only. These weights are passed to weighted cross-entropy loss so that common classes do not dominate the optimization.

---

## 4. Algorithms and Models Used

## 4.1 Transformer text encoders

### Phase 1

Phase 1 uses `bert-base-uncased`.

- Hidden representation size: 768.
- Text representation: masked mean pooling over token representations.
- In the controlled E0-E5 experiments, BERT is frozen.
- Only the task head and emoji/fusion components are trained.

### Phase 2

Phase 2 uses:

```text
cardiffnlp/twitter-roberta-base
```

This is a Twitter-pretrained RoBERTa backbone without emotion-task supervision. It is a better domain match for social-media text than generic BERT.

The project deliberately does not use a checkpoint already fine-tuned on the same emotion test task, because that would risk test-set leakage.

### Mean pooling

Let the last hidden states be $H_1, H_2, ..., H_T$ and the attention mask be $m_t$. The pooled text vector is:

$$
text = \frac{\sum_{t=1}^{T} m_t H_t}{\max(\sum_{t=1}^{T}m_t, 1)}
$$

This produces a fixed 768-dimensional vector regardless of sentence length.

---

## 4.2 MLP classification head

All models use a feed-forward classification head after the text/fusion representation.

Typical structure:

```text
Linear(input_dim -> 256)
ReLU
Dropout(0.3)
Linear(256 -> number_of_labels)
```

The final layer outputs one logit per class. Softmax converts logits into probabilities during inference.

The output dimension changes according to the task:

- Phase 1: 3 outputs.
- TweetEval Phase 2: 4 outputs.
- GoEmotions Phase 2: 6 outputs.

---

## 4.3 E0 / EM0: text-only baseline

The baseline intentionally ignores emojis.

```text
text_without_emoji
        |
        v
frozen transformer
        |
        v
768-d mean-pooled text vector
        |
        v
MLP classifier
        |
        v
class probabilities
```

Purpose:

- Establish the performance of words alone.
- Provide the control condition for the emoji-aware models.
- Make it possible to measure whether emoji fusion adds useful information.

Phase 2 also includes a fine-tuned EM0 version. In that version, the top two transformer blocks are unfrozen and trained with a small learning rate.

---

## 4.4 E1: random emoji concatenation

E1 uses a trainable random emoji embedding and simple concatenation.

```text
text vector: 768 dimensions
emoji vector: 32 dimensions
concatenated vector: 800 dimensions
MLP classifier
```

Emoji pooling is mean pooling over the valid emoji embeddings.

The hypothesis is that emojis may add information. However, simple concatenation does not let the model decide which emoji matters in a context. In Phase 1, this configuration performed worse than the text-only baseline, showing that naive emoji concatenation can add noise.

---

## 4.5 E2: pretrained emoji concatenation

E2 keeps the concatenation architecture but initializes the emoji representation using a pretrained TweetEval emoji artifact.

Purpose:

- Test whether better emoji initialization fixes the weaknesses of random concatenation.
- Separate the effect of initialization from the effect of the fusion architecture.

The main comparison is:

```text
E1 = random emoji embeddings + concatenation
E2 = pretrained emoji embeddings + concatenation
```

If E2 improves over E1, the result suggests that initialization quality matters.

---

## 4.6 E3 / EM3: text-conditioned attention

E3 is the main attention-based emoji model.

The text vector acts as a query that decides how much attention to give each emoji.

For text vector $x$ and emoji embeddings $e_i$:

$$
Q = W_q x
$$

$$
K_i = W_k e_i, \qquad V_i = W_v e_i
$$

The attention score for emoji $i$ is:

$$
score_i = \frac{Q \cdot K_i}{\sqrt{d}}
$$

After masking padding positions, the scores are converted into attention weights:

$$
\alpha_i = softmax(score_i)
$$

The emoji context is:

$$
context = \sum_i \alpha_i V_i
$$

The final representation is:

```text
fused = concatenate(text_vector, emoji_context)
```

The fused vector has 800 dimensions in the current architecture.

### Why attention is useful

Simple mean pooling gives every valid emoji equal initial treatment. Attention allows the model to condition the emoji contribution on the sentence.

For example, the same emoji can have different meanings in different contexts. The text-conditioned query gives the model a mechanism to weight the emoji sequence differently depending on the surrounding words.

### Empty emoji handling

A sentence can contain no emoji. Softmax over an all-padding sequence would produce invalid values, so E3 explicitly detects the empty case and uses a learned `emoji_absent` fallback vector.

This prevents NaNs and gives the model a learned representation for the absence of emojis.

### Phase 2 EM3

EM3 uses the same attention idea with a variable output head:

- 4 outputs for TweetEval.
- 6 outputs for GoEmotions.

The Phase 2 EM3 fine-tuning runs unfreeze the top two Twitter-RoBERTa blocks.

---

## 4.7 E4: pretrained emoji attention

E4 uses the same text-conditioned attention architecture as E3, but the emoji embeddings are pretrained and frozen.

The comparison is:

```text
E3 = random trainable emoji embeddings + attention
E4 = pretrained frozen emoji embeddings + attention
```

This tests whether task-specific random embeddings can adapt better than a general emoji representation.

---

## 4.8 E5 / EM5: gated fusion

E5 uses mean-pooled emoji features and a learned feature-wise gate.

First, the emoji embeddings are mean-pooled into a 32-dimensional vector $e$.

The text and emoji vectors are concatenated:

$$
z = [text; e]
$$

The gate is:

$$
g = sigmoid(W_g z + b_g)
$$

The gate has 32 dimensions, not one scalar:

$$
g \in (0,1)^{32}
$$

The gated emoji vector is:

$$
e_{gated} = g \odot e
$$

The final representation is:

```text
fused = concatenate(text_vector, gated_emoji_vector)
```

A gate value near 1 means that feature dimension is passed through more strongly. A value near 0 suppresses that dimension. These gate values are interpretability signals, not causal explanations.

E5 safely handles no-emoji examples by returning a zero emoji vector before gating.

EM5 exists in the Phase 2 frozen runs. A Phase 2 EM5 fine-tuned run was not included in the recorded Colab suite.

---

## 5. Training Algorithm

The training procedure is implemented in the shared pipelines.

### 5.1 Loss function

The project uses weighted cross-entropy:

$$
L = -w_y \log p(y|x)
$$

where $w_y$ is the class weight calculated from the training split.

### 5.2 Optimizer

The training loop uses AdamW.

AdamW combines adaptive gradient updates with decoupled weight decay. It is commonly used for transformer fine-tuning.

### 5.3 Learning rates

Different parameter groups are used in fine-tuning mode:

- Classification/fusion head: higher learning rate.
- Unfrozen transformer layers: lower learning rate, typically `2e-5`.

This is called discriminative fine-tuning. The new task head can learn quickly while the pretrained language features change more cautiously.

### 5.4 Learning-rate scheduler

The pipeline uses a OneCycle learning-rate schedule over the training steps.

### 5.5 Gradient clipping

Gradients are clipped to a maximum norm of 1.0 to reduce unstable updates.

### 5.6 Model selection

The best checkpoint is selected using validation Macro F1, not test performance. The test set is used only for final evaluation.

### 5.7 Reproducibility

The configured seed is 42. The training and evaluation artifacts record model configuration, dataset, label names, and metrics.

---

## 6. Phase 1: StockTwits Sentiment Study

### 6.1 Task

Phase 1 classifies financial social-media posts into:

```text
0 = Bearish
1 = Neutral
2 = Bullish
```

The final canonical dataset sizes are:

| Split | Rows |
|---|---:|
| Train | 91,121 |
| Validation | 20,676 |
| Test | 11,966 |

The training data was deduplicated and train/test text overlaps were removed according to the project data-integrity process.

### 6.2 Controlled experiment matrix

| Model | Architecture | Emoji representation | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| E0 | Text-only baseline | No emoji branch | 47.69% | 46.21% |
| E1 | Concatenation | Random trainable | 44.83% | 43.35% |
| E2 | Concatenation | TweetEval pretrained | 52.05% | 50.41% |
| E3 | Text-conditioned attention | Random trainable | 54.20% | 51.60% |
| E4 | Text-conditioned attention | TweetEval pretrained | 52.92% | 50.53% |
| E5 | 32-dimensional gated fusion | TweetEval pretrained | 52.33% | 50.91% |

### 6.3 Phase 1 interpretation

The main findings are:

1. E0 establishes a text-only baseline.
2. E1 shows that random emoji concatenation can hurt performance.
3. E2 shows that pretrained emoji initialization improves concatenation.
4. E3 is the best Phase 1 model, suggesting text-conditioned attention is useful for this StockTwits task.
5. E5 has the strongest reported Bearish recall and may be useful when downside-risk detection is prioritized.

Phase 1 does **not** prove that emojis always improve sentiment classification. The result depends on initialization, architecture, dataset, and class.

### 6.4 Phase 1 limitations

- It is specific to financial StockTwits language.
- The task is coarse compared with human emotion.
- The test split is not a general measure of psychological emotion understanding.
- The study uses one main seed.
- No formal statistical significance test was performed.

---

## 7. Phase 2: Fine-Grained Emotion Study

Phase 2 implements the teacher-directed change from sentiment polarity to emotion categories.

### 7.1 Datasets

#### TweetEval Emotion

Labels:

```text
anger
joy
optimism
sadness
```

Dataset sizes used in the recorded run:

| Split | Rows |
|---|---:|
| Train | 3,257 |
| Validation | 374 |
| Test | 1,421 |

#### GoEmotions Ekman-6

The project downloads GoEmotions and maps the original labels into six emotion families:

```text
anger
disgust
fear
joy
sadness
surprise
```

The adapter drops neutral and ambiguous multi-family examples and keeps examples that map to exactly one family.

Dataset sizes used in the recorded run:

| Split | Rows |
|---|---:|
| Train | 26,732 |
| Validation | 3,354 |
| Test | 3,362 |

This is not the full 27-label GoEmotions task. It is a filtered single-label Ekman-6 task.

### 7.2 Recorded Phase 2 results

#### GoEmotions Ekman-6

| Experiment | Mode | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| EM0 | Fine-tuned | 77.42% | 67.89% | 78.37% |
| EM3 | Fine-tuned | 77.28% | 68.08% | 78.25% |
| EM0 | Frozen | 72.19% | 60.30% | 73.35% |
| EM3 | Frozen | 71.24% | 59.21% | 72.70% |
| EM5 | Frozen | 71.33% | 58.39% | 72.46% |

Majority baseline:

- Accuracy: 55.41%.
- Macro F1: 11.89%.
- Majority class: joy.

Interpretation:

- Fine-tuning improves both the text-only and attention models over frozen controls.
- EM3 has the highest Macro F1 by a small margin.
- EM0 has slightly higher accuracy.
- The emoji-aware result is not universally better on every metric.

#### TweetEval Emotion

| Experiment | Mode | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| EM0 | Fine-tuned | 79.94% | 77.67% | 80.14% |
| EM3 | Fine-tuned | 79.38% | 76.10% | 79.68% |
| EM5 | Frozen | 76.28% | 72.57% | 76.53% |
| EM3 | Frozen | 75.09% | 72.35% | 75.26% |
| EM0 | Frozen | 73.54% | 70.32% | 74.08% |

Majority baseline:

- Accuracy: 39.27%.
- Macro F1: 14.10%.
- Majority class: anger.

Interpretation:

- Fine-tuned EM0 is the strongest TweetEval model in the recorded results.
- Fine-tuned EM3 is close behind.
- The fine-tuned models greatly outperform the majority baseline.
- This does not prove that emojis improve the result in all cases.

### 7.3 Emoji slice results

The evaluation records separate performance for test examples with and without emojis. These are useful descriptive diagnostics.

They must not be interpreted as causal proof because the presence of an emoji was not randomly assigned.

Examples from the recorded metrics:

- GoEmotions EM3 fine-tuned:
  - With emoji accuracy: 85.00%, $n=80$.
  - Without emoji accuracy: 77.09%, $n=3,282$.
- TweetEval EM0 fine-tuned:
  - With emoji accuracy: 75.08%, $n=333$.
  - Without emoji accuracy: 81.43%, $n=1,088$.

The different direction across datasets is exactly why emoji effects should be reported rather than assumed.

---

## 8. What the Dashboard Does

The Streamlit dashboard is now emotion-first.

### Live Emotion Analyzer

The analyzer:

1. Accepts a sentence or social-media post.
2. Extracts emojis.
3. Removes emojis from the text branch.
4. Loads a trained Phase 2 EM0 or EM3 checkpoint.
5. Produces probabilities over the selected dataset's label space.
6. Displays the predicted emotion and confidence.

Available label spaces:

- GoEmotions Ekman-6.
- TweetEval Emotion four-class.

The live result is a model prediction, not a psychological diagnosis.

### Text vs Emoji Emotion

This view compares:

- EM0: fine-tuned text-only model.
- EM3: fine-tuned text-conditioned emoji-attention model.

Both models receive the same input sentence. The view is an illustration of model behavior; the controlled test-set results are the evidence used for research conclusions.

### Emotion Study

This view reports:

- Accuracy.
- Macro Precision.
- Macro Recall through the per-class metrics artifact.
- Macro F1.
- Weighted F1.
- Majority baseline.
- Frozen versus fine-tuned comparison.
- Emoji/no-emoji slices.
- Per-class precision, recall, F1, and support.

### Research Conclusions

This page summarizes:

- Why emotion classification was introduced.
- Why fine-tuning helps.
- Why emoji fusion is a research variable rather than a guaranteed improvement.
- Which limitations must be stated to the teacher.

---

## 9. What Has Been Completed

### Data and preprocessing

- StockTwits canonical dataset construction.
- StockTwits deduplication and split preparation.
- Unicode-aware emoji extraction.
- Text-without-emoji and emoji-list representations.
- Train-only emoji vocabulary construction.
- TweetEval Emotion adapter.
- GoEmotions Ekman-6 adapter.
- Class-distribution and coverage reporting.

### Models

- Phase 1 E0 text-only baseline.
- Phase 1 E1 random concatenation.
- Phase 1 E2 pretrained concatenation.
- Phase 1 E3 text-conditioned attention.
- Phase 1 E4 pretrained attention.
- Phase 1 E5 gated fusion.
- Phase 2 EM0 text-only baseline.
- Phase 2 EM3 attention fusion.
- Phase 2 EM5 frozen gated fusion.
- Variable output dimensions for 3, 4, and 6 classes.
- Empty-emoji safety paths.
- Trainable-only checkpoint saving for Phase 2.

### Evaluation

- Accuracy.
- Macro precision.
- Macro recall.
- Macro F1.
- Weighted F1.
- Classification reports.
- Confusion matrices.
- Majority baselines.
- Emoji/no-emoji slices.
- Per-class analysis.

### Verification

- E3/E4/E5 architecture smoke tests pass.
- Emotion results contain metrics and prediction artifacts.
- Dashboard modules compile.
- Live TweetEval emotion inference works.
- Live GoEmotions emoji-aware inference works.
- Emotion Study tables and class-level detail work for both datasets.

---

## 10. What Has Not Been Completed

These items should be described as future work, not as completed features.

### 10.1 Custom teacher-defined labels

There is no dedicated trained six-class dataset with exactly:

```text
happiness/excitement
sarcasm/irony
sadness/despair
anger/frustration
anxiety/fear
curiosity/skepticism
```

The project uses standard benchmark labels instead. To implement the exact teacher-defined categories, a new labeled dataset or a validated relabeling protocol is required.

### 10.2 Explicit sarcasm classification

Sarcasm is discussed in examples and limitations, but there is no separate sarcasm classifier in the current trained models.

### 10.3 Multi-task emotion plus emoji prediction

The teacher suggested jointly predicting emotion and masked emojis. That multi-task objective has not been implemented.

### 10.4 Multiple-seed confidence intervals

The main results use one seed, 42. Repeating training with multiple seeds and reporting mean, standard deviation, and confidence intervals remains future work.

### 10.5 Statistical significance testing

No paired bootstrap, permutation test, or McNemar test has been included in the final results.

### 10.6 EM5 fine-tuning in Phase 2

The recorded Phase 2 suite includes EM5 frozen runs, but not a fine-tuned EM5 result.

### 10.7 Production deployment

The dashboard is a local Streamlit research demonstration. It is not a production API, monitoring system, or clinically validated emotion assessment tool.

---

## 11. Recommended Teacher Explanation

A concise explanation to say aloud:

> Our project has two phases. In Phase 1, we studied three-class financial sentiment on StockTwits: Bearish, Neutral, and Bullish. We compared a text-only BERT baseline with emoji concatenation, attention, and gated fusion. The best Phase 1 model was E3, which used text-conditioned attention over emoji embeddings and achieved 54.20% accuracy and 51.60% Macro F1.
>
> The teacher said that three-class sentiment is too coarse, so in Phase 2 we changed the task to fine-grained emotion classification. We used TweetEval Emotion with four classes and GoEmotions reduced to a single-label Ekman-6 task with six classes. We changed the backbone to Twitter-RoBERTa and fine-tuned its top two layers with a small learning rate. The best TweetEval accuracy was 79.94% from fine-tuned EM0, and the best GoEmotions Macro F1 was 68.08% from fine-tuned EM3. Therefore, our conclusion is not that emojis always improve accuracy. Instead, we show how text-only and emoji-aware architectures behave under controlled evaluation, and we report the emoji contribution as a dataset-dependent research result.

---

## 12. Likely Teacher Questions and Answers

### Why is Macro F1 important?

Accuracy can be dominated by a majority class. Macro F1 calculates F1 for each class and averages the class scores equally, so minority-class performance matters.

### Why do you remove emojis from the text input?

To prevent leakage between the text and emoji branches. The experiment must isolate what the emoji branch contributes.

### Why use two datasets in Phase 2?

TweetEval provides a social-media emotion benchmark with four labels. GoEmotions provides a broader emotion corpus that can be grouped into six Ekman families. Using both tests whether the observations transfer across datasets.

### Why not use a model already fine-tuned for emotion?

A checkpoint fine-tuned on the same task can have seen similar evaluation data. Using Twitter-RoBERTa without emotion-task supervision avoids that direct task leakage.

### Why is EM0 sometimes better than EM3?

The text branch already contains strong information. Emojis are sparse and their meaning can be ambiguous. Adding an emoji branch can help some examples but add noise or redundant information for others.

### Does the project detect anxiety and sarcasm?

Not as separate trained classes. Fear is available in GoEmotions and is the closest current category to anxiety. Sarcasm is not a current output class. A dedicated labeled dataset would be needed for exact sarcasm classification.

### What does attention mean here?

Attention is a learned weighting mechanism. The text representation creates a query, and the model assigns weights to the emoji embeddings based on the sentence context.

### What does the gate mean?

The gate is a 32-dimensional sigmoid vector that scales emoji features. It indicates which learned emoji feature dimensions are passed through more strongly, but it is not a causal explanation.

### What is the main limitation?

The project uses benchmark labels and one random seed. The results demonstrate controlled model behavior, but they do not prove universal human emotion understanding or that emojis cause better predictions.

---

## 13. Final Project Status

### Completed and defensible

- A complete Phase 1 controlled emoji-sentiment benchmark.
- A Phase 2 teacher-directed emotion classification extension.
- Multiple fusion algorithms.
- Fine-tuned Twitter-native transformer experiments.
- Reproducible metrics and saved artifacts.
- An emotion-first interactive dashboard.
- Smoke tests and data-path validation.

### Correct final claim

> The project demonstrates a controlled comparison of text-only and emoji-aware neural architectures for social-media sentiment and emotion classification. Fine-grained emotion classification is the current main task. Emoji fusion can provide useful information, but its benefit depends on the dataset, model architecture, and training setup.

### Claims to avoid

Do not say:

- The model predicts every emotion listed in the teacher's examples.
- Sarcasm is already a trained output class.
- Emojis always improve accuracy.
- The model understands human psychology.
- The results generalize to all social-media platforms.
- A single test run proves statistical superiority.
