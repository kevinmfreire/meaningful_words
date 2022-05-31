import pandas as pd
import re
import nltk
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Dwnloading NLTK packages (Only need to be run once)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')

# Split data by scale factor for faster preprocessing
def split_data(data, scale):
  neg_df = data[data['sentiment']=='negative']
  pos_df = data[data['sentiment']=='positive']
  neg_df = neg_df[0:(len(neg_df)//scale)+1]
  pos_df = pos_df[0:(len(pos_df)//scale)+1]
  new_df = pd.concat([neg_df, pos_df], axis=0)
  return new_df

def preprocessor(text):
  text = re.sub('[^a-zA-Z]',' ', text)    # remove punctuation
  text = text.lower()                     # convert to lowercase
  text = text.strip()                     # remove leading and tailing whitespacess
  text = ''.join([i for i in text if i in string.ascii_lowercase+' '])
  text = ' '.join([word for word in text.split() if word.isalnum()])  
  text = ' '.join([WordNetLemmatizer().lemmatize(word,pos='v') for word in text.split()]) 
  text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
  return text

def upload_data(path):
  '''
  DATA UPLOAD AND EXPLORATION
  '''
  # Load dataset
  # DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
  # DATASET_ENCODING = "ISO-8859-1"
  # tweet_df = pd.read_csv(path, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
  tweet_df = pd.read_csv(path, header=0)

  # Drop any unwanted columns
  # tweet_df.drop(['ids','date', 'flag', 'user'], axis=1, inplace=True)
  tweet_df.drop(['selected_text', 'textID'], axis=1, inplace=True)

  # The Sentiment140 dataset has labels 0-Negative and 4-Positive, let's decode them
  # to_sentiment = {0: "negative", 4: "positive"}
  # tweet_df.sentiment = tweet_df.sentiment.apply(lambda x: to_sentiment[x])

  print('\n\033[1mData Dimension:\033[0m Dataset consists of {} columns & {} records.'.format(tweet_df.shape[1], tweet_df.shape[0]))
  print(tweet_df.describe())
  return tweet_df

def process_data(dataframe):
  '''
  DATA PROCESSING:

  Inputs: Pandas dataframe, target variable
  returns: Labels and features of dataset
  '''
  # Remove null values
  print('\nNumber of null values in dataset:\n{}'.format(dataframe.isnull().sum()))
  dataframe.dropna(inplace=True)
  original_df = dataframe.copy(deep=True)

  # Remove Duplicates (if any)
  r, c = original_df.shape

  df_dedup = dataframe.drop_duplicates()
  df_dedup.reset_index(drop=True, inplace=True)

  if df_dedup.shape==(r,c):
    print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
  else:
    print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {r-df_dedup.shape[0]}')

  # Let's split data by factor to just apply quicker preprocessing
  # split_df = split_data(tweet_df_dedup, 4)

  # Data cleaning and preprocessing
  df_clean = df_dedup.copy()
  # tweet_df_clean = split_df.copy()
  df_clean['text'] = df_dedup['text'].apply(preprocessor)
  print(df_clean.head())

  save_path_cleanDF = '../data/processed/clean_data.pkl'
  df_clean.to_pickle(save_path_cleanDF)

  return df_clean

def tokenize_data(tf_idf_model, dataFrame, target):
  tf_idf = tf_idf_model
  label=dataFrame[target].values
  features=tf_idf.fit_transform(dataFrame.text)
  save_path_label = '../data/processed/label.npy'
  save_path_feature = '../data/processed/feature.npy'
  np.save(save_path_label, label, allow_pickle=True)
  np.save(save_path_feature, features, allow_pickle=True)
  return label, features

if __name__ == '__main__':
  
  path = '../data/raw/tweets.csv'
  tweet_df = upload_data(path)
  process_data(tweet_df)
  print('Data finsihed processing!')