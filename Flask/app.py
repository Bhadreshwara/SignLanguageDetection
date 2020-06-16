from flask import Flask
from flask import render_template
from flask import request, redirect
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import tensorflow as tf
from keras.models import load_model
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

from tensorflow.keras.initializers import glorot_uniform

loaded_model = tf.keras.models.load_model("CNN_Model.h5", custom_objects={'GlorotUniform': glorot_uniform()})
def img_class(model, img):
    img_arr = np.asarray(img)
    pred_probab = model.predict(img_arr)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


@app.route('/')
def hello_world():
    return 'Try to use /upload-image'

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files['file']
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(image.filename))
            image.save(file_path)
            
            image = cv2.imread(file_path)
            img_gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Applying Gaussian BLur
            img_gry_blr = cv2.GaussianBlur(img_gry, (5, 5), 0)

            # Resizing
            img_1 = cv2.resize(img_gry_blr, (28, 28), interpolation=cv2.INTER_AREA)
            img_2 = np.resize(img_1, (28, 28, 1))
            img_3 = np.expand_dims(img_2, axis=0)

            pred_probab, pred_class = img_class(loaded_model, img_3)

            char_op = chr(pred_class + 65)
            print(char_op)
            # image = request.files["image"]

            return char_op

    return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0')
