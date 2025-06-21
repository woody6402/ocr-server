from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import os
import yaml
import uuid

import numpy as np
import tensorflow as tf
from math import atan2, pi, fmod
import logging

from flask import render_template
from werkzeug.utils import secure_filename
from flask import redirect, url_for
from flask import send_from_directory

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

CONFIG_FILE = "config.yaml"
OUTPUT_DIR = "/app/images"

# Stelle sicher, dass der Output-Ordner existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lade Konfiguration
def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


config = load_config()

def run_lcd_model(image):
    return "[lcd_v2 result]"

    
# Optional global cache für das Modell
interpreter_cache = {}


def run_model(image: Image.Image, model_name: str) -> dict:
    #return run_seg_model(image, model_name)

    if model_name.startswith("dig-"):
        return run_digit_model(image, model_name)
    elif model_name.startswith("ana-") or "analog" in model_name:
        return run_analog_model(image, model_name)
    else:
        return run_tesseract_model(image, model_name)


def run_digit_model(image: Image.Image, model_name: str) -> dict:
    model_path = f"models/{model_name}.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']  # [1, H, W, 3]
    width, height = input_shape[2], input_shape[1]

    img = image.resize((width, height)).convert("RGB")
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class = int(np.argmax(output_data))
    logging.info(f"[DIFITAL MODEL] Digit: {predicted_class}")

    return {
        "class": predicted_class,
        "scores": output_data.tolist()
    }



import pytesseract

def run_tesseract_model(image: Image.Image, model_name: str = "tesseract") -> dict:
    try:
        text = pytesseract.image_to_string(image, config="--psm 7").strip()
        logging.info(f"[TESSERACT MODEL] Text: {text}")
        return {
            "class": text
        }
    except Exception as e:
        return {
            "text": f"ERROR: {str(e)}"
        }


def digit_from_vector(f1: float, f2: float) -> float:
    """
    Konvertiert f1/f2-Vektorausgabe in eine analoge Position im Bereich [0.0, 10.0).
    """
    raw_angle = atan2(f1, f2)
    normalized = fmod(raw_angle / (2 * pi) + 2, 1)
    return normalized * 10


def run_analog_model(image: Image.Image, model_name: str) -> dict:
    model_path = f"models/{model_name}.tflite"
    #interpreter = tf.lite.Interpreter(model_path=model_path)
    #interpreter.allocate_tensors()
    
    if model_name not in interpreter_cache:
        interpreter_cache[model_name] = tf.lite.Interpreter(model_path=model_path)
        interpreter_cache[model_name].allocate_tensors()
    interpreter = interpreter_cache[model_name]


    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    width, height = input_shape[2], input_shape[1]

    img = image.resize((width, height)).convert("RGB")
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    f1, f2 = output_data[0], output_data[1]
    digit = digit_from_vector(f1, f2)
    logging.info(f"[ANALOG MODEL] Digit: {digit}")
    return {
        "raw": output_data.tolist(),
        "class": digit, 
        "scores": output_data.tolist()
    }


@app.route("/segment", methods=["POST"])
def segment_and_ocr():
    if "identifier" not in request.form or "image" not in request.files:
        return jsonify({"error": "Missing identifier or image"}), 400

    identifier = request.form["identifier"]
    image_file = request.files["image"]

    if identifier not in config:
        return jsonify({"error": f"Identifier '{identifier}' not found in config"}), 404

    image = Image.open(image_file.stream).convert("RGB")
    definition = config[identifier]

    results = []

    for key, section in definition.items():
        model = section.get("model", "tesseract")
        logging.info(f"[PIC Parsing] Seciton: {key}, Model: {model}")
        if "vorkomma" in section or "nachkomma" in section:
            digits = []
            for group in ["vorkomma", "nachkomma"]:
                for rect in section.get(group, []):
                    x, y, w, h = rect
                    segment = image.crop((x, y, x + w, y + h))
                    result = run_model(segment, model)
                    digits.append(str(result.get("class", "?")))
            #value = "".join(digits[:len(section.get("vorkomma", []))]) + "." + "".join(digits[len(section.get("vorkomma", [])):])
            
            vorkomma_len = len(section.get("vorkomma", []))
            vorkomma_part = "".join(digits[:vorkomma_len]) or "0"
            nachkomma_part = "".join(digits[vorkomma_len:]) or "0"
            
            value = f"{vorkomma_part}.{nachkomma_part}"

            logging.info(f"[PIC Parsing] Value: {value}")
        elif "rects" in section:            
            # 
            values = []
            for rect in section["rects"]:
                x, y, w, h = rect
                segment = image.crop((x, y, x + w, y + h))
                result = run_model(segment, model)
                values.append(result.get("class", ""))
                
            if model == "tesseract":
                value = "".join(str(v) for v in values if v != "")    
            else:
                value = values[0] 
                                
            logging.info(f"[PIC Parsing] Value: {value}")
        else:
            value = None

        results.append({
            "id": key,
            "value": value
        })

    return jsonify({"identifier": identifier, "results": results})
    
    

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    image = request.files.get("image")
    if image:
        filename = secure_filename(image.filename)
        path = os.path.join("images", filename)
        image.save(path)
        return redirect(url_for("editor", image_name=filename))
    return "No image uploaded", 400

@app.route("/editor/<image_name>")
def editor(image_name):
    return render_template("editor.html", image_name=image_name)

@app.route("/save_segments", methods=["POST"])
def save_segments():
    data = request.json
    identifier = data.get("identifier")
    segments = data.get("segments")

    if not identifier or not segments:
        return jsonify({"error": "Missing data"}), 400

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config[identifier] = segments

    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    return jsonify({"status": "ok"})

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory("images", filename)

MODEL_MAP = {
    "digital": "dig-cont_0900_s3_q",
    "analog": "ana-cont_1500_s2_q",
    "tesseract": "tesseract"    
}

@app.route("/test-config", methods=["POST"])
def test_config():
    save = request.args.get("save", "false").lower() == "true"
    data = request.get_data(as_text=True)
    try:
        logging.info(f"[CTest] Data: {data}")
        parsed = yaml.safe_load(data)
        if not isinstance(parsed, dict):
            return jsonify({"error": "YAML root must be a dictionary"}), 400

        global config

        # Mapping vereinfachter Modelnamen
        for ident, blocks in parsed.items():
            for section_name, section in blocks.items():
                if isinstance(section, dict) and "model" in section:
                    raw_model = section["model"]
                    section["model"] = MODEL_MAP.get(raw_model, raw_model)

        if save:
            # Alte Konfiguration laden, mit neuer verschmelzen, schreiben
            with open(CONFIG_FILE, "r") as f:
                existing_config = yaml.safe_load(f) or {}
            existing_config.update(parsed)
            with open(CONFIG_FILE, "w") as f:
                #yaml.dump(existing_config, f)
                yaml.dump(existing_config, f, Dumper=FlowStyleListDumper, sort_keys=False)
            config.update(parsed)
            return jsonify({"status": "saved", "updated_keys": list(parsed.keys())})
        else:
            # Nur im RAM aktualisieren
            config.update(parsed)
            return jsonify({"status": "loaded (not saved)", "updated_keys": list(parsed.keys())})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




class FlowStyleListDumper(yaml.SafeDumper):
    pass

# Nur Listen mit ausschließlich Zahlen im Flow-Style (z. B. [1, 2, 3, 4])
def represent_list(dumper, data):
    if all(isinstance(i, (int, float)) for i in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    else:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data)

FlowStyleListDumper.add_representer(list, represent_list)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

