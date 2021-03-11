from flask import Flask, render_template, request
import numpy as np
# from sklearn.externals import joblib

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def home():
    return render_template("home.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0")