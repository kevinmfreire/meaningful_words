from flask import Flask, request, render_template
import pickle
import sys

sys.path.append("./src")
sys.path.append("./models")

from src.predict import classify

app = Flask(__name__)
tf_idf = pickle.load(open("./models/td_idf.pickle", "rb"))
model = pickle.load(open("./models/log_reg_model.pickle", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    tweet = request.form['tweet']
    prediction = classify(tweet, tf_idf, model)

    return render_template('index.html', prediction_text=f'Your tweet "{tweet}" is classified as {prediction} sentiment.')

if __name__=="__main__":
    app.run(host="0.0.0.0")