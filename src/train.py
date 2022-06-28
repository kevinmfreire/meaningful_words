import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluate import classification_summary
from utils import train_test_data
from pre_process import extract_features, tokenizer_porter

# Word2Vec Packages
from collections import defaultdict
import multiprocessing

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec, KeyedVectors
from time import time  # To time our operations

# Using Word2vec for feature extraction, below I've taken examples from https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook
# Using Gensim Phrases package to detect common phrases (bigrams) from a list of sentences
def detect_common_phrases(clean_df):
  sent = [row.split() for row in clean_df['text']]
  phrases = Phrases(sent, min_count=30, progress_per=10000)

  # Using Phraser() to cut down memory consumption pf Phrases() by discarding model state not strictly
  # needed for the bigram detection task
  bigram = Phraser(phrases)

  # Transform the corpus based on the bigrams detected
  sentences = bigram[sent]
  return sentences

def train_word2vec(clean_data_file):
  cores = multiprocessing.cpu_count() # Count the number of cores in a computer

  clean_df = pd.read_csv(clean_data_file)
  clean_df = clean_df.dropna().reset_index(drop=True)
  sentences = detect_common_phrases(clean_df)

  # Initialize model
  w2v_model = Word2Vec(min_count=20, window=2, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=cores-1)

  # Building the vocabulary table
  w2v_model.build_vocab(sentences, progress_per=10000)

  # Training model
  w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

  # Save word2vec model for future training
  w2v_model_path = '../models/w2v_model.model'
  w2v_model.save(w2v_model_path)

  # Seperate trained vectors into KeyedVectors sp that you don't need the full model state and can be discarded keeping only the vectors and their keys.
  # This results in a much smaller and faster object that can be mapped for lighning fast loading and sharing vectors in RAM
  w2v_vector_path = '../models/w2v_model.wordvectors'
  word_vectors = w2v_model.wv
  word_vectors.save(w2v_vector_path)

  return w2v_model

def train_classifier_w2v(clean_data_file):
  target='sentiment'
  text='text'
  clean_df = pd.read_csv(clean_data_file)
  clean_df = clean_df.dropna().reset_index(drop=True)
  X_train, X_test, y_train, y_test = train_test_data(clean_df[target].values, clean_df[text].values)


  w2v = KeyedVectors.load('../models/w2v_model.wordvectors', mmap='r')
  # w2v = w2v.index_to_key

  unique_words = set(w2v.index_to_key)

  X_train_vect = np.array([np.array([w2v[word] for word in sent.split() if word in unique_words]) for sent in X_train])
  X_test_vect = np.array([np.array([w2v[word] for word in sent.split() if word in unique_words]) for sent in X_test])
  # X_train_vect = [[w2v[word] for word in sent.split() if word in unique_words] for sent in X_train]
  # X_test_vect = [[w2v[word] for word in sent.split() if word in unique_words] for sent in X_test]

  # Compute sentence vectors by averaging the word vectors for the words contained in the sentence
  X_train_vect_avg = []
  for v in X_train_vect:
    if v.size:
      X_train_vect_avg.append(v.mean(axis=0))
    else:
      X_train_vect_avg.append(np.zeros(100, dtype=float))
          
  X_test_vect_avg = []
  for v in X_test_vect:
    if v.size:
      X_test_vect_avg.append(v.mean(axis=0))
    else:
      X_test_vect_avg.append(np.zeros(100, dtype=float))

  log_reg_model = LogisticRegression()
  log_reg_model.fit(X_train_vect_avg, y_train.values.ravel())

  pred = log_reg_model.predict(X_test_vect_avg)
  pred_prob = log_reg_model.predict_proba(X_test_vect_avg)
  classification_summary(pred,pred_prob, y_test,'Logistic Regression (LR)')

def train_classifier_tfidf(clean_data_file):
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
  
  clean_data_file = '../data/processed/clean_df.csv'

  # train_classifier()
  # quit()
  # w2v_model = train_word2vec(clean_data_file)

  train_classifier_w2v(clean_data_file)

  wv = KeyedVectors.load('../models/w2v_model.wordvectors', mmap='r')
  print(wv['mother'].shape)

  # train()