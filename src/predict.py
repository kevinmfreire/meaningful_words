'''
predict.py script is for creating a ML pipeline from input text,
to text preprocessing, to feature extraction and model prediction.
'''

import pickle

from pre_process import text_preprocessor


def classify(text, tf_idf, model):
    '''
    Pipeline for classifying input text and providing sentiment.
    '''
    label_decoder = {0: "negative", 1: "neutral", 2: "positive"}

    p_text = text_preprocessor(text)
    p_text = [p_text]

    features = tf_idf.transform(p_text)
    pred = model.predict(features)

    return label_decoder.get(pred[0])


if __name__ == "__main__":
    tf_idf = pickle.load(open("../models/td_idf.pickle", "rb"))
    model = pickle.load(open("../models/log_reg_model.pickle", "rb"))

    text = input("Type tweet: ")

    pred = classify(text, tf_idf, model)
    print("Your tweet is classified as {}".format(pred))
