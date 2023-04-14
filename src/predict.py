'''
predict.py script is for creating a ML pipeline from input text,
to text preprocessing, to feature extraction and model prediction.
'''

import pickle

from pre_process import text_preprocessor


def classify(text, tf_idf, model):
    '''
    The classify function takes in three arguments: 
        * text: A string of text that the function classifies.
        * tf_idf: A fitted TfidfVectorizer object that transforms the input text into a vector of tf-idf features.
        * model: A trained machine learning model that the function uses to classify the input text

    The function first initializes a dictionary called label_decoder, which maps integer labels to sentiment labels of "negative", "neutral", or "positive".
    The input text is preprocessed using the text_preprocessor function, which is not shown in the code snippet. The preprocessed text is then stored in a list.
    The tf_idf object is used to transform the preprocessed text into a vector of features.

    Returns: 
        * The sentiment label is returned as a string.

    The model object is used to predict the label of the input text based on the vector of features. The predicted label is an integer value.
    The predicted integer label is then used to look up the corresponding sentiment label in the label_decoder dictionary. 
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
