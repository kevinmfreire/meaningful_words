import pickle

from pre_process import preprocessor
from nltk.stem.porter import PorterStemmer

def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

def predict(text, tf_idf, model):
    label_decoder = {0:'negative', 1:'neautral', 2:'positive'}
    text = [text]
    features = tf_idf.transform(text)
    pred = model.predict(features)
    # output = label_decoder.get(pred[0])
    # print(output)
    # quit()
    print('Your tweet is classified as {}'.format(label_decoder.get(pred[0])))

if __name__ == "__main__":
    tf_idf_path = '../models/td_idf.pickle'
    model_path = '../models/log_reg_model.pickle'

    tf_idf = pickle.load(open(tf_idf_path, "rb"))
    model = pickle.load(open(model_path, "rb"))

    text = input("Type tweet: ")
    clean_text = preprocessor(text)
    predict(clean_text, tf_idf, model)
