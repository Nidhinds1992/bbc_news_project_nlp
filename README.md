# BBC News Text Classification using NLP | TF-IDF + SVM

## Project Overview
This project is a Natural Language Processing (NLP) based text classification system developed using the BBC News dataset.

The model classifies news articles into categories such as:
- Business
- Entertainment
- Politics
- Sport
- Tech

The final model uses:
- TF-IDF Vectorizer
- Linear SVM (LinearSVC)

---

## Model Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,3),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True
    )),
    ('svm', LinearSVC(C=1))
])
```

---

## Features
- NLP text preprocessing
- TF-IDF vectorization
- N-gram feature engineering
- Multi-class text classification
- FastAPI deployment
- Render cloud hosting

---

## Tech Stack
Python | Scikit-learn | NLP | FastAPI | Render | GitHub

---

## Author
Nidhin D S  
Generative AI / Machine Learning Engineer
