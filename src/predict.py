import pickle

from pre_process import preprocessor
from nltk.stem.porter import PorterStemmer

def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

def predict(text):
    label_decoder = {0:'negative', 1:'neautral', 2:'positive'}

    tf_idf = pickle.load(open("../models/td_idf.pickle", "rb"))
    model = pickle.load(open("../models/log_reg_model.pickle", "rb"))

    p_text = preprocessor(text)
    p_text = [p_text]

    features = tf_idf.transform(p_text)
    pred = model.predict(features)

    return label_decoder.get(pred[0])

if __name__ == "__main__":

    text = input("Type tweet: ")

    pred = predict(text)
    print('Your tweet is classified as {}'.format(pred))
