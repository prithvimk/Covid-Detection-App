import os
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
)
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.python.keras.saving.model_config import model_from_json

# os.chdir('D:\Work\Avtaar\Projects\covid\MyApp')


STATIC = r"Static"
UPLOAD_FOLDER = r"Static/Uploads"
MODEL_FOLDER = r"Static/Models"
CLASSES = {0: "Covid", 1: "Lung Opacity", 2: "Normal", 3: "Viral Pneumonia"}
# Initalise the Flask app
app = Flask(__name__, template_folder='templates')

# Loads pre-trained model
# model = load_model(r"Static\Models\CovidModelwithLRDecay.h5")
# model = pickle.load(open('MyApp/Static/Models/CovidModelwithLRDecay.pkl', 'rb'))

with open(r"Static\Models\CovidModelwithLRDecay_kaggle.json", "r") as file:
    model_json = file.read()

model = model_from_json(model_json)
model.load_weights(r"Static\Models\CovidModelwithLRDecay_kaggle_weights.h5")
print("Model loaded successfully.")


def predict(fullpath):
    data = image.load_img(fullpath, target_size=(256, 256, 3))
    # (x,y,3) ==> (1,x,y,3)
    data = np.expand_dims(data, axis=0)
    # Scaling
    data = data.astype("float") / 255
    # Prediction
    result = model.predict(data)
    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    else:
        file = request.files["image"]
        fullpath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullpath)

        result = predict(fullpath)
        class_idx = np.argmax(result)

        pred_prob = result[0, class_idx]
        label = CLASSES[class_idx]
        accuracy = round(pred_prob * 100, 2)

        return render_template(
            "predict.html",
            image_file_name=file.filename,
            label=label,
            accuracy=accuracy,
        )


@app.route("/upload/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
