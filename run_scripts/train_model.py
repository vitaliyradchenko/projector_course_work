import os

import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_FOLDER = "model"


def main() -> None:
    train = fetch_20newsgroups(subset="train")

    pipeline = Pipeline(
        [("tf_idf", TfidfVectorizer(ngram_range=(1, 3), max_features=10000)), ("clf", LogisticRegression(C=1))]
    )
    pipeline.fit(train["data"], train["target"])
    joblib.dump(pipeline, os.path.join(MODEL_FOLDER, "model.pkl"))
    joblib.dump(train.target_names, os.path.join(MODEL_FOLDER, "target_names.pkl"))


if __name__ == "__main__":
    main()
