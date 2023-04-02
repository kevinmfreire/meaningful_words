'''
train.py is a script for training machine learning models and saving it to
as a pickle file.
'''

import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluate import classification_summary
from pre_process import extract_features, tokenizer_porter
from utils import train_test_data


def train():
    '''Training function for the Logistic Tegression algorithm.'''
    clean_data_file = "../data/processed/clean_df.csv"
    target = "sentiment"

    tf_idf = TfidfVectorizer(
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        tokenizer=tokenizer_porter,
        use_idf=True,
        norm="l2",
        smooth_idf=True,
    )

    clean_df = pd.read_csv(clean_data_file)
    label, feature = extract_features(tf_idf, clean_df, target)

    X_train, X_test, y_train, y_test = train_test_data(label, feature)

    # Building Logistic Regression Classifier
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)

    tfidf_path = "../models/td_idf.pickle"
    LR_model_path = "../models/log_reg_model.pickle"
    pickle.dump(tf_idf, open(tfidf_path, "wb"))
    pickle.dump(log_reg_model, open(LR_model_path, "wb"))

    pred = log_reg_model.predict(X_test)
    pred_prob = log_reg_model.predict_proba(X_test)
    classification_summary(pred, pred_prob, y_test, "Logistic Regression (LR)")


if __name__ == "__main__":
    train()
