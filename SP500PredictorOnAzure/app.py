from flask import Flask, render_template, request
import numpy as np
# from sklearn.externals import joblib
from joblib import load
from datetime import date
from dateutil.relativedelta import relativedelta
from fredapi import Fred
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def getData():

    fred = Fred(api_key="39046de0f8665857c262677588e22e46")

    observation_start = (date.today() - relativedelta(months=3)).strftime("%Y-%m-%d") # '2020-12-14'
    observation_end = date.today().strftime("%Y-%m-%d") # '2021-03-14'
    datestr = observation_start + "__" + observation_end

    """
    daily
    """
    filenames = ["SP500", "CPALTT01USM657N", "WM1NS", "BAA10Y", "GDP"]
    data = []
    for index in filenames:
        s = fred.get_series(index, observation_start=observation_start, observation_end=observation_end)
        data.append(s.values[-1])
        s.to_csv("%s.csv" % (index), index=False)
    return data

def getPredict(data):
    reg = load("static/reg.joblib")

    return data[0] + reg.predict([data[1:]])

@app.route("/")
def home():
    # return render_template("home.html")
    data = getData()
    predicted = getPredict(data)[0]

    sp500 = pd.read_csv("SP500.csv")
    plt.plot(range(len(sp500.values)), sp500.values)
    plt.scatter(len(sp500.values), predicted, c="r", s=50)


    plt.ylim([0,5000])
    plt.savefig("static/SP500.png")

    return render_template("simple_img.html")

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0")