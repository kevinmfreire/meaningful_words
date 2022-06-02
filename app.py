from flask import Flask, request, render_template
import pickle

from src.predict import predict
from src.pre_process import preprocessor

app = Flask(__name__)
model = pickle.load(open('models/log_reg_model.pickle', 'rb'))

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    tweet = request.form["tweet"]
    prediction = model.predict([tweet])  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = prediction

    return render_template('index.html', prediction_text=f'Your tweet "{tweet}" is classified as {output} sentiment.')

if __name__=="__main__":
    app.run(host="0.0.0.0")