from flask import Flask, redirect, url_for
from flask import render_template, request
from predict import predict


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def result():
    user_input = request.form['user_input']

    probability, sentiment = predict(user_input)

    sentiment = "Positive" if sentiment == 1 else "Negative"
    return render_template('result.html', sentiment=sentiment, probability=probability)

@app.route('/back_to_index')
def back_to_index():
    return redirect(url_for('index'))

if __name__ == '__main__':
   app.run(debug = True)