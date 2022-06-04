import pandas as pd
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluate import classification_summary
from utils import train_test_data
from pre_process import extract_features, tokenizer_porter

def train():
  clean_data_file = '../data/processed/clean_df.csv'
  target = 'sentiment'

  # porter = PorterStemmer()
  tf_idf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer_porter, use_idf=True, norm='l2', smooth_idf=True)

  clean_df = pd.read_csv(clean_data_file)
  label, feature = extract_features(tf_idf, clean_df, target)

  # label, feature = load_data(label_file, feature_file)
  X_train, X_test, y_train, y_test = train_test_data(label, feature)

  # Building Logistic Regression Classifier
  log_reg_model = LogisticRegression()
  log_reg_model.fit(X_train, y_train)

  tfidf_path = '../models/td_idf.pickle'
  LR_model_path = '../models/log_reg_model.pickle'
  pickle.dump(tf_idf, open(tfidf_path, "wb"))
  pickle.dump(log_reg_model, open(LR_model_path, "wb"))

  pred = log_reg_model.predict(X_test)
  pred_prob = log_reg_model.predict_proba(X_test)
  classification_summary(pred,pred_prob, y_test,'Logistic Regression (LR)')

if __name__ == '__main__':
  train()