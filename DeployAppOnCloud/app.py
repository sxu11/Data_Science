from flask import Flask, render_template, request
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            NewYork = float(request.form['NewYork'])
            Califonia = float(request.form['California'])
            Florida = float(request.form['Florida'])
            RnD_Spend = float(request.form['RnD_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Market_Spend = float(request.form['Market_Spend'])

            pred_args = [NewYork, Califonia, Florida, RnD_Spend, Admin_Spend, Market_Spend]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)

            mul_reg = open("output/multiple_linear_model.pkl", "rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "Please check if the values are entered correctly"
        return render_template("predict.html", prediction = model_prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0")