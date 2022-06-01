import pandas as pd
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from evaluate import classification_summary, train_test_data
from pre_process import tokenize_data

# This tokenizer is used for the TF-IDF feature extraction and the stemmer is called in if __name__ == '__main__' section 
def tokenizer_porter(text):
  return [porter.stem(word) for word in text.split()]

if __name__ == '__main__':

  # label_file = '../data/processed/label.npy'
  # feature_file = '../data/processed/feature.npy'
  clean_data_file = '../data/processed/clean_data.pkl'
  target = 'sentiment'

  porter = PorterStemmer()
  tf_idf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer_porter, use_idf=True, norm='l2', smooth_idf=True)

  clean_df = pd.read_pickle(clean_data_file)
  label, feature = tokenize_data(tf_idf, clean_df, target)

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

  # Building  Decision Tree Classifier
  # DT_model = DecisionTreeClassifier()
  # DT_model.fit(X_train, y_train)
  # pred = DT_model.predict(X_test)
  # pred_prob = DT_model.predict_proba(X_test)
  # classification_summary(pred, pred_prob, 'Decision Tree Classifier (DT)')