from flask import Flask, request, render_template
from flask_cors import CORS
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import datetime

import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import train_model

app = Flask(__name__)

with open("model.pkl", 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = str(request.form.values)
    output = model.predict([text])
    prediction = output[0]
    # make_plot(prediction)
    if prediction < 0:
        prediction = str(prediction)[:6]
    else:
        prediction = str(prediction)[0:5]
    return render_template('predict.html',
                           prediction_text="Sentiment prediction: " + str(prediction))



# @app.route('/plot.png')
# def get_plot():
#     fig = plt.imread("plot.png")
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')


# def make_plot(prediction):
      # df = train_model.request_data(train_model.credentials)
      # dates, sentiment = train_model.date_average_pairs(df)

#     fig, ax = plt.subplots()
#     ax.plot(dates, sentiment, color='blue', label="Historical")

#     predict_date = datetime.timedelta(days=1)
#     ax.scatter(predict_date, prediction, color='red', label="Prediction")

#     ax.plot([dates[-1], predict_date], [sentiment[-1], prediction], color='red')

#     ax.set_xlabel('Date')
#     ax.set_ylabel('Sentiment')
#     ax.legend()
#     fig.autofmt_xdate()

if __name__ == "__main__":
    app.run(debug=True)


