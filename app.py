# import keras.models
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# import matplotlib.pyplot as plt


app = Flask(__name__)


model = load_model("static/model.h5")


def predict(img):
    pr = model.predict(img)
    # print(np.round(pr[0]*100), 3)
    return np.argmax(pr)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/result', methods=['GET', 'POST'])
def result():
    img = request.files['img']
    img.save('static/1.png')

    img = cv2.imread("static/1.png")
    img = cv2.resize(img, (28, 28))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.invert(img)

    img = np.reshape(img, (1, 28, 28, 1))
    img = img.astype('float32')
    # img = img/255.0

    # plt.imshow(img.reshape(28, 28, 1))
    # plt.show()

    pred = predict(img)

    # print(img)

    return render_template("result.html", pred=pred)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, port=81)
