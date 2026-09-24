# Hybrid evaluation report

All parameters chosen on validation; test scored once. No model was trained.

## Sarcasm — TweetEval irony test (human-annotated, n=784)

Collection hashtags (#irony, #sarcasm, #not) removed from every tweet before scoring: they appear in 311/311 ironic and 120/473 non-ironic test tweets, so leaving them in measures hashtag lookup, not sarcasm.

| Method | Irony F1 | Precision | Recall | Macro F1 | Accuracy |
|---|---|---|---|---|---|
| always predict irony | 56.8 | 39.7 | 100.0 | 28.4 | 39.7 |
| rules only | 20.1 | 76.6 | 11.6 | 48.2 | 63.5 |
| pretrained irony model | 65.7 | 72.0 | 60.5 | 73.0 | 75.0 |
| pretrained + rules (tuned for irony F1) | 67.3 | 56.9 | 82.3 | 68.2 | 68.2 |
| pretrained + rules (tuned for macro F1), probability only | 67.1 | 67.3 | 66.9 | 72.8 | 74.0 |
| **two-tier 'detected' verdict — what the app acts on** | 19.0 | 91.7 | 10.6 | 48.0 | 64.2 |

Emoji-bearing test tweets (n=88):

| Method | Irony F1 | Precision | Recall | Macro F1 | Accuracy |
|---|---|---|---|---|---|
| rules only | 65.5 | 76.0 | 57.6 | 74.3 | 77.3 |
| pretrained irony model | 63.2 | 75.0 | 54.5 | 72.8 | 76.1 |
| pretrained + rules, probability only | 77.6 | 76.5 | 78.8 | 81.9 | 83.0 |
| **two-tier 'detected' verdict** | 67.9 | 90.0 | 54.5 | 77.1 | 80.7 |

Tuning objective for the used configuration is macro F1 (false alarms rewrite sincere emotions, so both error types count). This objective was adopted after the irony-F1-tuned result had been seen on test; both are shown.

## Emotion — held-out test sets

| Dataset / model | Setting | Acc (all) | Macro F1 (all) | Acc (emoji rows) | Macro F1 (emoji rows) | Acc (no-emoji rows) |
|---|---|---|---|---|---|---|
| tweeteval/EM0 (emoji rows n=333) | model only | 79.7 | 77.0 | 75.1 | 67.3 | 81.2 |
| tweeteval/EM0 (emoji rows n=333) | + emoji lexicon | 81.5 | 78.6 | 82.6 | 72.3 | 81.2 |
| tweeteval/EM0 (emoji rows n=333) | + emoji + sarcasm (validated) | 81.5 | 78.6 | 82.6 | 72.3 | 81.2 |
| tweeteval/EM0 (emoji rows n=333) | dashboard headline (intended if sarcasm detected) | 81.3 | 78.5 | 81.7 | 72.2 | 81.2 |
| tweeteval/EM3 (emoji rows n=333) | model only | 79.2 | 76.0 | 74.2 | 67.0 | 80.7 |
| tweeteval/EM3 (emoji rows n=333) | + emoji lexicon | 80.9 | 77.8 | 81.7 | 73.8 | 80.7 |
| tweeteval/EM3 (emoji rows n=333) | + emoji + sarcasm (validated) | 80.9 | 77.8 | 81.7 | 73.8 | 80.7 |
| tweeteval/EM3 (emoji rows n=333) | dashboard headline (intended if sarcasm detected) | 80.7 | 77.6 | 81.1 | 74.0 | 80.6 |
| goemotions/EM0 (emoji rows n=80) | model only | 77.6 | 68.5 | 90.0 | 83.2 | 77.3 |
| goemotions/EM0 (emoji rows n=80) | + emoji lexicon | 77.6 | 68.5 | 91.2 | 84.3 | 77.3 |
| goemotions/EM0 (emoji rows n=80) | + emoji + sarcasm (validated) | 77.6 | 68.5 | 91.2 | 84.3 | 77.3 |
| goemotions/EM0 (emoji rows n=80) | dashboard headline (intended if sarcasm detected) | 77.2 | 68.3 | 88.8 | 80.7 | 76.9 |
| goemotions/EM3 (emoji rows n=80) | model only | 75.8 | 66.4 | 85.0 | 71.9 | 75.6 |
| goemotions/EM3 (emoji rows n=80) | + emoji lexicon | 76.0 | 66.5 | 90.0 | 77.6 | 75.6 |
| goemotions/EM3 (emoji rows n=80) | + emoji + sarcasm (validated) | 76.0 | 66.5 | 90.0 | 77.6 | 75.6 |
| goemotions/EM3 (emoji rows n=80) | dashboard headline (intended if sarcasm detected) | 75.5 | 66.3 | 87.5 | 74.1 | 75.2 |

## Behavioural suite (hand-written, n=110) — capability check, not a benchmark

- Sarcasm (rules): F1 75.4, precision 100.0, recall 60.5
- Sarcasm (pretrained): F1 81.6, precision 81.6, recall 81.6
- Sarcasm (combined probability): F1 88.1, precision 80.4, recall 97.4
- Sarcasm (two-tier detected): F1 94.4, precision 100.0, recall 89.5

| Dataset / model | Model only | Hybrid (validated) | Dashboard headline |
|---|---|---|---|
| tweeteval/EM0 (n=107) | 58.9 | 72.0 | 85.0 |
| tweeteval/EM3 (n=107) | 58.9 | 75.7 | 86.0 |
| goemotions/EM0 (n=110) | 52.7 | 63.6 | 73.6 |
| goemotions/EM3 (n=110) | 50.0 | 60.9 | 71.8 |
