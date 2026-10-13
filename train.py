"""Train the TF-IDF + Passive Aggressive model and save it to model.joblib.

Expects data/news.csv with a text column (or title + text) and a label column
with REAL/FAKE (or 1/0). Falls back to a tiny built-in sample so the app still
deploys, but accuracy will be poor until you add a real dataset.
"""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/news.csv"

SAMPLE_ROWS = [
    ("Scientists publish peer-reviewed study on crop yields in drought conditions.", "REAL"),
    ("Central bank holds interest rates steady, citing inflation data.", "REAL"),
    ("City council approves budget for new public library branch.", "REAL"),
    ("Local hospital reports increase in flu vaccinations this season.", "REAL"),
    ("SHOCKING: Government secretly replacing tap water with mind control chemicals!", "FAKE"),
    ("You won't believe this one weird trick doctors don't want you to know!", "FAKE"),
    ("Celebrity claims moon landing was filmed in a basement, insiders confirm.", "FAKE"),
    ("Miracle cure erases all cancer overnight, big pharma hiding it, share now!", "FAKE"),
]


def load_kaggle_split():
    """Kaggle 'Fake and Real News' layout: data/Fake.csv and data/True.csv."""
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")
    fake["label"] = "FAKE"
    real["label"] = "REAL"
    df = pd.concat([fake, real], ignore_index=True)
    parts = [c for c in ("title", "text") if c in df.columns]
    df["text"] = df[parts].fillna("").astype(str).agg(" ".join, axis=1)
    return df["text"].tolist(), df["label"].tolist()


def load_data():
    if os.path.exists("data/Fake.csv") and os.path.exists("data/True.csv"):
        return load_kaggle_split()

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        parts = [c for c in ("title", "text") if c in df.columns]
        if not parts:
            raise SystemExit("data/news.csv needs a 'text' or 'title' column.")
        df["text"] = df[parts].fillna("").astype(str).agg(" ".join, axis=1)
        if "label" not in df.columns:
            raise SystemExit("data/news.csv needs a 'label' column (REAL/FAKE or 1/0).")
        labels = df["label"].astype(str).str.upper().replace({"1": "REAL", "0": "FAKE"})
        return df["text"].tolist(), labels.tolist()

    print("WARNING: data/news.csv not found; training on a tiny sample only.")
    texts, labels = zip(*SAMPLE_ROWS)
    return list(texts), list(labels)


def main():
    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42,
        stratify=labels if len(set(labels)) > 1 and len(labels) >= 10 else None,
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    print(f"Test accuracy: {accuracy:.3f} on {len(y_test)} articles")

    joblib.dump({"vectorizer": vectorizer, "model": model}, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
