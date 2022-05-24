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

# Label decoder
def label_decoder(label):
    return to_sentiment[label]

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

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

'''
DATA UPLOAD AND EXPLORATION
'''
# Load dataset
path = './data/tweets.csv'
DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
tweet_df = pd.read_csv(path, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

# Drop any unwanted columns
tweet_df.drop(['ids','date', 'flag', 'user'], axis=1, inplace=True)

# The Sentiment140 dataset has labels 0-Negative and 4-Positive, let's decode them
to_sentiment = {0: "negative", 4: "positive"}
tweet_df.sentiment = tweet_df.sentiment.apply(lambda x: label_decoder(x))

# Keep an original copy of the tweet dataset
target = 'sentiment'
original_df = tweet_df.copy(deep=True)
print('\n\033[1mData Dimension:\033[0m Dataset consists of {} columns & {} records.'.format(tweet_df.shape[1], tweet_df.shape[0]))
print(tweet_df.describe())

'''
DATA PROCESSING
'''

# Remove Duplicates (if any)
counter = 0
r, c = original_df.shape

tweet_df_dedup = tweet_df.drop_duplicates()
tweet_df_dedup.reset_index(drop=True, inplace=True)

if tweet_df_dedup.shape==(r,c):
  print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
else:
  print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {r-tweet_df_dedup.shape[0]}')

# Let's split data by factor to just apply quicker preprocessing
split_df = split_data(tweet_df_dedup, 4)

# Data cleaning and preprocessing
# tweet_df_clean = tweet_df_dedup.copy()
tweet_df_clean = split_df.copy()
tweet_df_clean['text'] = tweet_df_clean['text'].apply(preprocessor)
print(tweet_df_clean.head())

porter = PorterStemmer()
tf_idf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer_porter, use_idf=True, norm='l2', smooth_idf=True)
label=tweet_df_clean[target].values
features=tf_idf.fit_transform(tweet_df_clean.text)

save_path_label = './data/label.npy'
save_path_feature = './data/feature.npy'
np.save(save_path_label, label, allow_pickle=True)
np.save(save_path_feature, features, allow_pickle=True)