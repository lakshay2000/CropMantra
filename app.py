import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

str=" "

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    s= str.join(prediction)

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The predicted crop is {}'.format(s))


if __name__=="__main__":
    app.run(debug=True)