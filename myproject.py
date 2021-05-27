from flask.helpers import send_file
from MLManager import MLManager
from flask import Flask,request,send_from_directory,jsonify
import os
import shutil

app = Flask(__name__)
model = MLManager("best.pt")

# save the label and image path result
model_result = None
image_result_path = None

def delete_uploaded_image(path_to_img):
    os.remove(path_to_img)

@app.route("/")
def hello():
    return "Hi From Flask"

@app.route("/detectLabel",methods=["POST","GET"])
def detect():
    global model,image_result_path
    if request.method == "GET":
        return "To Detect from image use POST with 'file' with image to predict"
    
    # Delete previous image result
    model.delete_detection_folder()
    image_result_path = None

    # Handle POST
    print("Getting image from POST")
    uploaded_image = request.files["file"]

    # save to directory of images to infer
    dir = os.path.join("images-to-infer",uploaded_image.filename)
    print(f"Save image to {dir}")
    uploaded_image.save(dir)

    # get result
    res = model.predict_image(dir)
    image_result_path = os.path.join("detection-result","exp",uploaded_image.filename)
    delete_uploaded_image(dir)
    return jsonify(res)

@app.route("/getImageDetectionResult",methods=["GET"])
def getImageDetectionResult():
    global image_result_path
    if image_result_path == None:
        return "Please made a detection request via /detectLabel"
    else:
        return send_file(image_result_path)


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)