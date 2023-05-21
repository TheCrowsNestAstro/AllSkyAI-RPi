import os
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageOps
import numpy as np
import time
import datetime
import shortuuid
import urllib.request
import logging
import json
from dotenv import dotenv_values

config = dotenv_values(".env")

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

working_dir = os.getcwd()
logging.debug(f"Current working dir: {working_dir}")

def load_labels(filename):
    logging.debug('load_labels')
    with open(filename, 'r') as f:
        result = [line.strip() for line in f.readlines()]
        logging.debug("Loaded labels: {labels}".format(labels=result))
        return result

def get_utc_timestamp():
    dt = datetime.datetime.now(datetime.timezone.utc)
    utc_time = dt.replace(tzinfo=datetime.timezone.utc)
    utc_timestamp = utc_time.timestamp()
    return int(utc_timestamp)

def download_image_from_url(url):
    logging.debug('download_image_from_url')

    url += "?" + str(get_utc_timestamp())

    temp_path = os.path.join(working_dir, "tmp")

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    temp_image_base = shortuuid.uuid() + ".jpg"
    temp_image_path = os.path.join(temp_path, temp_image_base)

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]

    urllib.request.urlcleanup()
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, temp_image_path)
    logging.debug(f"Saved downloaded image to: {temp_image_path}")

    return temp_image_path

def load_image(width, height, color_mode, image_path):
    img = Image.open(image_path)

    if color_mode == "grayscale":
        img = ImageOps.grayscale(img)

    img = img.resize((width, height), Image.Resampling.LANCZOS)  # width, height
    img_array = np.array(img, dtype=np.float32)
    logging.debug(f"Original image shape: {img_array.shape}")
    img_array = np.expand_dims(img_array, 0)

    if color_mode == "grayscale":
        img_array = np.expand_dims(img_array, 3)

    logging.debug(f"Resampled/Reshaped image shape: {img_array.shape}")

    logging.debug("Removing temporary image")

    if os.path.isfile(image_path):
        os.remove(image_path)

    return img_array

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image):
    set_input_tensor(interpreter, image)

    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    label_index = np.argmax(output)
    score = softmax(output)
    confidence = 100 * np.max(score)
    logging.debug(f"CONFIDENCE: {confidence}")

    return label_index, confidence

def do_classification():
    model_path = os.path.join(working_dir, "models", config["TFLITE_MODEL"])
    label_path = os.path.join(working_dir, "models", config["LABELS_MAP"])

    interpreter = Interpreter(model_path)
    logging.debug("Model Loaded Successfully.")

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    logging.debug(f"TF Lite input shape: {height}, {width}")

    # Load an image to be classified.
    image_path = download_image_from_url(config["ALLSKY_URL"])
    logging.debug(f"Downloading image from: {config['ALLSKY_URL']}")
    img_array = load_image(width=width, height=height, color_mode=config["COLOR_MODE"], image_path=image_path)

    # Classify the image.
    time1 = time.time()
    label_id, confidence = classify_image(interpreter, img_array)

    time2 = time.time()
    classification_time = np.round(time2-time1, 3)

    logging.debug(f"Classificaiton Time = {classification_time} seconds.")

    # Read class labels.
    labels = load_labels(label_path)
    classification_label = labels[label_id]

    logging.debug(f"AllSkyAI: {classification_label}, Confidence: {confidence}%")

    data_json = dict()
    data_json['classification'] = classification_label
    data_json['confidence'] = round(confidence, 3)
    data_json['utc'] = get_utc_timestamp()
    data_json['inference'] = classification_time
    logging.debug(json.dumps(data_json))
    return json.dumps(data_json)
