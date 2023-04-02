'''
pre_process.py is a script develop to keep several functions for processing text data.
'''

import re
import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


def label_decoder(label):
    '''This function returns models prediction vlaue as positive, negative or neutral sentiment.'''
    to_sentiment = {0: "negative", 1: "neutral", 2: "positive"}
    return to_sentiment[label]


def tokenizer_porter(text):
    '''Tokenize text data using the PorterStemmer for stemming words to utmost root word.'''
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


def split_data(data, scale):
    '''Split data by scale factor for faster preprocessing'''
    neg_df = data[data["sentiment"] == "negative"]
    pos_df = data[data["sentiment"] == "positive"]
    neg_df = neg_df[0: (len(neg_df) // scale) + 1]
    pos_df = pos_df[0: (len(pos_df) // scale) + 1]
    new_df = pd.concat([neg_df, pos_df], axis=0)
    return new_df


def text_preprocessor(text):
    '''
    Remove punctuation, convert to lowercase, remove leading and tailing whitespaces.
    Then lowercase all words, checks if text is alphanumeric, and reduces words to
    utmost root meaning based on context.
    '''
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.strip()
    text = "".join([i for i in text if i in string.ascii_lowercase + " "])
    text = " ".join([word for word in text.split() if word.isalnum()])
    text = " ".join(
        [WordNetLemmatizer().lemmatize(word, pos="v") for word in text.split()]
    )
    text = " ".join(
        [word for word in text.split() if word not in stopwords.words("english")]
    )
    return text


def upload_data(path):
    """
    DATA UPLOAD AND EXPLORATION
    """
    # Load dataset
    tweet_df = pd.read_csv(path, header=0)

    # Drop any unwanted columns
    tweet_df.drop(["selected_text", "textID"], axis=1, inplace=True)

    print(
        "\n\033[1mData Dimension:\033[0m Dataset consists of {} columns & {} records.".format(
            tweet_df.shape[1], tweet_df.shape[0]
        )
    )
    print(tweet_df.describe())
    return tweet_df


def process_data(dataframe):
    """
    DATA PROCESSING:

    Inputs: Pandas dataframe, target variable
    returns: Labels and features of dataset
    """
    # Remove null values
    print("\nNumber of null values in dataset:\n{}".format(dataframe.isnull().sum()))
    dataframe.dropna(inplace=True)
    original_df = dataframe.copy(deep=True)

    # Remove Duplicates (if any)
    r, c = original_df.shape

    df_dedup = dataframe.drop_duplicates()
    df_dedup.reset_index(drop=True, inplace=True)

    if df_dedup.shape == (r, c):
        print("\n\033[1mInference:\033[0m The dataset doesn't have any duplicates")
    else:
        print(
            f"\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {r-df_dedup.shape[0]}"
        )

    # Data cleaning and preprocessing
    df_clean = df_dedup.copy()
    df_clean["text"] = df_dedup["text"].apply(text_preprocessor)
    print(df_clean.head())

    return df_clean


def extract_features(tf_idf_model, dataFrame, target):
    '''
    Utilizing TF-IDF to extract features by converting each word to a numerical value
    based on the word frequency within the given corpus.
    '''
    tf_idf = tf_idf_model
    label = dataFrame[target].values
    features = tf_idf.fit_transform(dataFrame.text.values.astype("U"))
    save_path_label = "../data/processed/label.npy"
    save_path_feature = "../data/processed/feature.npy"
    np.save(save_path_label, label, allow_pickle=True)
    np.save(save_path_feature, features, allow_pickle=True)
    return label, features


if __name__ == "__main__":
    path = "../data/raw/Tweets.csv"
    clean_data_path = "../data/processed/clean_df.csv"

    tweet_df = upload_data(path)
    clean_df = process_data(tweet_df)
    clean_df.to_csv(clean_data_path)

    print("Data finsihed processing!")
