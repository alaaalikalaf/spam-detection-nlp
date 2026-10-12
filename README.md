# SMS Spam Detection (NLP, algorithms built from scratch)

An NLP project that classifies SMS messages as **spam** or **ham**.
The text pipeline and the machine learning models are implemented **manually with NumPy** (no scikit-learn): TF-IDF, train/test split, oversampling, metrics and the classifiers.

## What was done
- Data loading, cleaning and duplicate removal (5,169 unique messages, 12.6% spam)
- Exploratory data analysis (class balance, message length, top terms)
- Text preprocessing: lowercasing, URL/number/punctuation removal, stopwords, simple stemming, unigrams + bigrams
- TF-IDF vectorization implemented from scratch (4,000 features)
- Manual train/test split and **random oversampling** on the training set only
- Models implemented from scratch: Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, Perceptron, Ensemble Voting
- Evaluation with accuracy, precision, recall, F1 and confusion matrices

## Results (test set, 1,034 messages)
| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Multinomial NB | 0.9603 | 0.8272 | 0.9116 | 0.8673 |
| **Bernoulli NB** | **0.9778** | **0.9493** | 0.8912 | **0.9193** |
| Logistic Regression | 0.9729 | 0.9161 | 0.8912 | 0.9034 |
| Perceptron | 0.9023 | 0.6198 | 0.8095 | 0.7021 |
| Ensemble Voting | 0.9739 | 0.9225 | 0.8912 | 0.9066 |

## Dataset
SMS Spam Collection (`spam.csv`). Download it (for example from Kaggle or the UCI repository) and place it next to the notebook.

## How to run
```
pip install -r requirements.txt
jupyter notebook spam_detection_analysis.ipynb
```

## Tools
Python, NumPy, Pandas, Matplotlib, Seaborn
