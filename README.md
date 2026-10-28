# Fake-News-Detection


Classifies news articles/headlines as **FAKE** or **REAL** using NLP (TF-IDF) and supervised machine learning,
and **explains** each decision. Includes a training pipeline, evaluation, saved model, CLI tool and a Streamlit web app.

## 1. Problem statement
Misinformation spreads quickly online. This project builds a text classifier that predicts whether a news item is
likely fake or real from patterns learned on labelled articles, and shows *why* (highlighted words, style red flags).

## 2. Dataset (69,913 de-duplicated items)
| Source | FAKE | REAL | Content |
|---|---:|---:|---|
| agnews | 0 | 11,975 | Short items, world/sports/business/sci-tech, many outlets |
| bbc | 0 | 2,127 | Full articles: sport, business, politics, tech, entertainment |
| ifnd_true | 0 | 11,972 | Indian news headlines (Tribune, ThePrint, Indian Express...) |
| isot | 16,779 | 21,154 | US politics/world; Reuters = REAL, unreliable sites = FAKE |
| mcintire | 2,971 | 2,935 | 2016 US election, mixed |

**Data curation decisions (important for the report):**
- *First version* used only ISOT + McIntire (US politics). It failed on a real Times of India cyclone report, because
  it had never seen Indian/disaster/human-interest writing. Adding diverse **REAL** sources (BBC, AG News, IFND-real) fixed this.
- The IFND "Fake" rows and the COVID tweet dataset were **deliberately excluded**: they are fact-checker headlines
  ("Fact-check: Did X say Y?") and tweets, not fake articles, and would teach cues like "viral"/"claim" = FAKE.
- **Source-leak removal:** every REAL ISOT article starts with "WASHINGTON (Reuters) -", and FAKE ones contain
  "Featured image via Getty" and split contractions ("don t"). A model that learns these scores ~99% yet fails in the
  real world, so `preprocess.py` strips them.
- **Duplicates removed** and the split is **by article**, so no article appears in both train and test.
- Class imbalance (more REAL than FAKE) handled with `class_weight="balanced"`.

## 3. Project structure
```
data/news_combined.csv          combined dataset
src/build_dataset.py            downloads + merges + de-duplicates data
src/preprocess.py               text cleaning (incl. source-leak removal)
src/train.py                    trains/compares models, saves best
src/evaluate_generalization.py  train on one dataset, test on the other
src/predict.py                  command-line prediction
src/explain.py                  explainability, style red flags, credibility index, URL fetching
app.py                          Streamlit web app
models/  outputs/               saved model, charts, reports
```

## 4. Pipeline
1. **Preprocess:** strip source giveaways, lowercase, remove URLs/HTML/punctuation, stopwords, lemmatise.
2. **Augment:** training set = full articles + headlines + 3-sentence snippets (so short inputs work).
3. **TF-IDF:** unigrams + bigrams, 100k features, sublinear TF.
4. **Models compared:** Logistic Regression, Linear SVM, SGD (modified Huber), Multinomial Naive Bayes.
5. **Selection:** best average of article-F1 and snippet-F1, preferring models that give probabilities and word weights
   (needed for the explanations). Saved with joblib.

## 5. Results (20% held-out items, never seen in training)
| Model | Accuracy | F1 (FAKE) | Short-text Acc | Short-text F1 |
|---|:-:|:-:|:-:|:-:|
| **Logistic Regression** | 95.6% | 92.5% | 88.3% | 86.4% |
| Linear SVM | 95.4% | 92.2% | 88.7% | 86.9% |
| SGD (modified Huber) | 95.3% | 91.9% | 88.3% | 86.6% |
| Multinomial Naive Bayes | 92.8% | 87.8% | 88.0% | 86.1% |

Selected model: **Logistic Regression**. Accuracy by source:

| Source | Accuracy |
|---|:-:|
| agnews | 95.1% |
| isot | 97.9% |
| mcintire | 79.2% |
| ifnd_true | 96.6% |
| bbc | 98.0% |

McIntire is the hardest source (its labels are noisier). BBC / Indian headlines / AG News were never the cause of
errors once included in training.

### Cross-dataset generalisation (train on one source, test on the other)
| Train on | Test on (unseen) | Accuracy | F1 |
|---|---|:-:|:-:|
| isot | mcintire | 57.0% | 67.2% |
| mcintire | isot | 63.6% | 61.0% |

Models trained on a single source transfer poorly because each dataset defines "fake" differently. This is why the
final model is trained on a mix, and it is the main limitation below.

## 6. Setup (Windows)
```
py -m pip install -r requirements.txt
py src\train.py                      (retrain once; ~1-2 minutes)
py -m streamlit run app.py
```
Command line: `py src\predict.py --text "your headline"`. To rebuild the data: `py src\build_dataset.py`.

## 7. Innovative features (`src/explain.py`, `app.py`)
- **Explainable AI:** words highlighted red (toward FAKE) / green (toward REAL) from the model's learned weights.
- **Writing-style red flags:** ALL-CAPS, exclamation spam, clickbait phrases, loaded words, missing sources/quotes.
- **Credibility Index (0-100):** 70% ML probability + 30% style signal (heuristic weights, not tuned).
- **UNCERTAIN verdict** when the model recognises too few words or the probability is near 50/50.
- URL analysis (encoding-safe, strips site-name suffix from titles), batch CSV classification, model dashboard.

## 8. Limitations (be upfront about these)
- **Detects writing style and source patterns, not factual truth.** A well-written falsehood in wire-news style is
  classified REAL, and an angry opinion piece can be classified FAKE.
- The FAKE class only comes from US-politics sites (ISOT/McIntire); fake news in other domains (health, Indian
  politics, gossip) is caught mainly through sensational style, so subtle fakes there can pass as REAL.
- Training data is mostly 2016-2020 news; newer events/topics are less reliable.
- Cross-dataset accuracy is far below in-dataset accuracy (section 5).
- Credibility Index weights and style rules are heuristic.

## 9. Future work
Fact-checking integration (e.g. Google Fact Check API), claim verification against trusted sources, source/domain
credibility scoring, fine-tuned transformers (BERT/RoBERTa) on more diverse fake-news data, multilingual support.

