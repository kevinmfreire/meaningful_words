import streamlit as st
import pickle
import nltk_setup
import sys
sys.path.append("src/")
from src.predict import classify
from src.utils import render_svg

tf_idf = pickle.load(open("./models/td_idf.pickle", "rb"))
model = pickle.load(open("./models/log_reg_model.pickle", "rb"))

svg = """
        <svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 128 128">
            <path d="M40.254 127.637c48.305 0 74.719-48.957 74.719-91.403 0-1.39 0-2.777-.075-4.156 5.141-4.547 9.579-10.18 13.102-16.633-4.79 2.602-9.871 4.305-15.078 5.063 5.48-4.02 9.582-10.336 11.539-17.774-5.156 3.743-10.797 6.38-16.68 7.801-8.136-10.586-21.07-13.18-31.547-6.32-10.472 6.86-15.882 21.46-13.199 35.617C41.922 38.539 22.246 26.336 8.915 6.27 1.933 20.94 5.487 39.723 17.022 49.16c-4.148-.172-8.207-1.555-11.832-4.031v.41c0 15.273 8.786 28.438 21.02 31.492a21.596 21.596 0 01-11.863.543c3.437 13.094 13.297 22.07 24.535 22.328-9.305 8.918-20.793 13.75-32.617 13.72-2.094 0-4.188-.15-6.266-.446 12.008 9.433 25.98 14.441 40.254 14.422" fill="#1da1f2"/>  
        </svg>
    """
html = render_svg(svg)

if __name__=='__main__':
    st.set_page_config(layout="wide")
    st.write(html, unsafe_allow_html=True)
    st.write("""
            # Sentiment Classification App

            This web application allows any user to input a 'Tweet' and the machine learning model will classify your tweet as positive, neutral or negative.

            ## How it works
            I created a pipeline for the users input, and it works as follows:
            * Once the user inputs their tweet it goes through a text preprocessor removing punctuations, stop words, etc.
            * It then passes through a trained Term Frequency Inverse Document Frequency (TF-IDF) model to tokenize the input tweet, transforming the data that is acceptable to the trained ML model.
            * It then passes through a Linear Regression model which has three outputs labeled as negative (0), neautral (1) or positive(2).
            * The model then predicts the sentiment of the tweet and displays result.

            ## Demo
            Try it out yourself! Input something you would tweet and see what the mcahine learning model thinks of it.
            """)

    user_input = st.text_input("Input Tweet")

    if user_input:
        prediction = classify(user_input, tf_idf, model)
        st.write('Your Tweet "{}" is classified as {}.'.format(user_input, prediction))