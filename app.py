from PIL import Image
from PIL import ImageEnhance, ImageOps, ImageFilter

import pytesseract
import os
import yaml
import uuid
import logging
import colorsys
import re
import string
import json

import numpy as np

# import tensorflow as tf

from tflite_runtime.interpreter import Interpreter

from math import atan2, pi, fmod

from flask import Flask, request, jsonify
from flask import render_template
from flask import redirect, url_for
from flask import send_from_directory

from werkzeug.utils import secure_filename

import paho.mqtt.publish as publish


from transformations import apply_transformations  # Stelle sicher, dass du transformations.py hast



MODEL_MAP = {
    "digital-cont": "dig-cont_0900_s3_q",
    "digital-class11": "dig-class11_2000_s2_q",    
    "digital-class100": "dig-class100-0180-s2-q",    
    "analog": "ana-cont_1505_s2_q",
    "tesseract": "tesseract"    
}

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

CONFIG_FILE = "/app/config/config.yaml"
OUTPUT_DIR = "/app/images"

# Stelle sicher, dass der Output-Ordner existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lade Konfiguration
def load_config():
    with open(CONFIG_FILE, "r") as f:
        x = yaml.safe_load(f)
        logging.info(f"[LOAD Config]: {x}")    
        return x


config = load_config()

def run_lcd_model(image):
    return "[lcd_v2 result]"

    
# Optional global cache für das Modell
interpreter_cache = {}





def run_led_model(image: Image.Image, model_name: str = "led") -> dict:
    # Durchschnitts-RGB und max-Helligkeit berechnen
    arr = np.array(image.convert("RGB"))
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    mean_rgb = tuple(map(int, mean_rgb))

    # Max-Helligkeit berechnen (für bessere Einschätzung von LED-Status)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness_array = 0.2126 * r + 0.7152 * g + 0.0722 * b
    max_brightness = brightness_array.max()

    # Helligkeitsschwelle zum Erkennen, ob LED "an" ist
    if max_brightness < 100:
        return {
            "class": "off",
            "rgb": mean_rgb,
            "brightness": round(max_brightness)
        }

    # Farbe ermitteln (falls LED an)
    r_norm, g_norm, b_norm = [x / 255.0 for x in mean_rgb]
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    hue = h * 360

    if 0 <= hue <= 40 or hue > 320:
        color = "red"
    elif 40 < hue <= 75:
        color = "yellow"
    elif 75 < hue <= 160:
        color = "green"
    elif 160 < hue <= 250:
        color = "blue"
    else:
        color = "unknown"

    return {
        "class": color,
        "rgb": mean_rgb,
        "brightness": round(max_brightness)
    }

    

def run_model(image: Image.Image, model_name: str) -> dict:

    model_name = MODEL_MAP.get(model_name, model_name)

    if model_name.startswith("dig-"):
        return run_digit_model(image, model_name)
    elif model_name.startswith("ana-") or "analog" in model_name:
        return run_analog_model(image, model_name)
    elif model_name.startswith("color-") or model_name == "color":
        return run_color_model(image, model_name)
    elif model_name.startswith("led-") or model_name == "led":
        return run_led_model(image, model_name)
    elif model_name.startswith("lcd-"):
        return run_lcd_sequence_model(image, model_name)
    else:
        return run_tesseract_model(image, model_name)




def classify_color(rgb):
    r, g, b = rgb
    if r > 180 and g < 100 and b < 100:
        return "red"
    elif g > 180 and r < 100 and b < 100:
        return "green"
    elif b > 180 and r < 100 and g < 100:
        return "blue"
    elif r > 180 and g > 180 and b < 100:
        return "yellow"
    else:
        return "unknown"


def get_average_rgb(image: Image.Image) -> tuple:
    arr = np.array(image.convert("RGB"))
    mean_color = arr.reshape(-1, 3).mean(axis=0)
    return tuple(map(int, mean_color))

def run_color_model(image: Image.Image, model_name: str = "color") -> dict:
    rgb = get_average_rgb(image)
    return {
        "class": f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"
    }


def extract_num_classes(model_name: str) -> int:
    """Extrahiert die Anzahl der Klassen aus dem Modellnamen (z. B. 'class100')"""
    match = re.search(r'class(\d+)', model_name)
    return int(match.group(1)) if match else 10  # fallback zu 10 Klassen

def run_digit_model(image: Image.Image, model_name: str) -> dict:
    model_path = f"models/{model_name}.tflite"
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']  # [1, H, W, 3]
    width, height = input_shape[2], input_shape[1]

    # Bild vorbereiten
    img = image.resize((width, height)).convert("RGB")
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Modell aufrufen
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Vorhersage bestimmen
    predicted_index = int(np.argmax(output_data))
    num_classes = extract_num_classes(model_name)

    # Ziffer berechnen
    if num_classes == 100:
        digit_value = str(int(round(predicted_index / 10.0, 1)))  # z. B. 50 → 5.0
    elif num_classes == 11:
        digit_value = str(int(predicted_index)) if predicted_index < 10 else "?"  # 0–9, 10 = "N"
    else:
        digit_value = str(int(predicted_index)) # fallback: 0–(n-1)

    logging.info(f"[DIGIT MODEL] Class Index: {predicted_index}")
    logging.info(f"[DIGIT MODEL] Calculated Value: {digit_value}")
#    logging.info(f"[DIGIT MODEL] Raw Scores: {output_data}")

    return {
        "class_index": predicted_index,
        "class": digit_value,
        "scores": output_data.tolist()
    }

    

def run_digit_model_x(image: Image.Image, model_name: str) -> dict:
    model_path = f"models/{model_name}.tflite"
    interpreter = Interpreter(model_path=model_path)
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
    logging.info(f"[DIGITAL MODEL] Digit: {predicted_class}")
#    logging.info(f"[DIGITAL MODEL] Digit: {output_data}")
   

    return {
        "class": predicted_class,
        "scores": output_data.tolist()
    }





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


alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)


def prepare_input_pil1(image: Image.Image, input_shape) -> np.ndarray:
    _, height, width, channels = input_shape

    # 1. Format konvertieren
    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    # 2. Resize
    image = image.resize((200, 31))

    # 3. In Array & normalisieren
    img_array = np.asarray(image, dtype=np.float32) / 255.0

    # 4. ggf. Channel-Dimension hinzufügen
    if channels == 1 and img_array.ndim == 2:
        img_array = img_array[:, :, np.newaxis]  # → [H, W, 1]

    # 5. Batch-Dimension hinzufügen
    img_array = np.expand_dims(img_array, axis=0)  # → [1, H, W, C]
    
    return img_array

def prepare_input_pil(image: Image.Image, input_shape) -> np.ndarray:
    _, height, width, channels = input_shape

    image = image.convert("L")
    image = image.resize((width, height))
    input_data = np.asarray(image, dtype=np.float32) / 255.0
    #input_data = input_data[:, :, np.newaxis]  # [1, 32, 20, 1]
    input_data = np.stack([input_data]*3, axis=-1)
    #input_data = np.expand_dims(input_data, 3) 
    input_data = np.expand_dims(input_data, axis=0)  # → [1, H, W, C]
    input_data = input_data.astype('float32')/255
    
    return input_data

def run_lcd_sequence_model(image: Image.Image, model_name: str) -> dict:
    model_path = f"models/{model_name}.tflite"
    
    if model_name not in interpreter_cache:
        interpreter_cache[model_name] = Interpreter(model_path=model_path)
        interpreter_cache[model_name].allocate_tensors()
    interpreter = interpreter_cache[model_name]
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()  

    # Resize and normalize the input image
    input_shape = input_details[0]['shape']
    #width, height = input_shape[2], input_shape[1]

    #img = image.resize((width, height)).convert("RGB")
    #img_array = np.asarray(img, dtype=np.float32) / 255.0
    #img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, H, W, 1)
    #img_array = np.expand_dims(img_array, axis=0)
 
    img_array = prepare_input_pil(image,input_shape)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])  # shape: (timesteps,)
    logging.info(f"[LCD MODEL] {output} {alphabet} {blank_index}")    
    text = "".join(alphabet[int(i)] for i in output[0] if int(i) not in [blank_index, -1])

    return {
        "class": text,
        "raw_indices": output.tolist()
    }


def run_analog_model(image: Image.Image, model_name: str) -> dict:
    model_path = f"models/{model_name}.tflite"
    #interpreter = tf.lite.Interpreter(model_path=model_path)
    #interpreter.allocate_tensors()
    
    if model_name not in interpreter_cache:
        interpreter_cache[model_name] = Interpreter(model_path=model_path)
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



def is_valid_value(value: str, match_pattern: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return bool(re.match(match_pattern, value))
    except re.error:
        return False



def apply_enhancement(image, identifier, enhance_steps):

    for step in enhance_steps:
    
        step_str = (
            "_".join(f"{k}-{v}" for k, v in step.items())
            if isinstance(step, dict)
            else str(step)
        )
        
        logging.info(f"[PIC Enhance]: {step_str}")
        if isinstance(step, str):
            # Einfache Schritte ohne Parameter
            if step == "grayscale":
                image = image.convert("L")
            elif step == "autocontrast":
                image = ImageOps.autocontrast(image)
            elif step == "invert":
                image = ImageOps.invert(image)

        elif isinstance(step, dict):
            # Schritte mit Parametern
            if "contrast" in step:
                factor = float(step["contrast"])
                image = ImageEnhance.Contrast(image).enhance(factor)
            elif "threshold" in step:
                t = int(step["threshold"])
                if image.mode != "L":
                    image = image.convert("L")
                image = image.point(lambda x: 255 if x > t else 0)
                image = image.convert("1")
            elif "brightness" in step:
                factor = float(step["brightness"])  # z. B. 1.8
                image = ImageEnhance.Brightness(image).enhance(factor) 
                
            elif "sharpen" in step:
                factor = float(step["sharpen"])
                image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(150 * factor)))

        # 🔽 Bild zur Kontrolle speichern
        image.save(f"images/{identifier}_{step_str}_debug.jpg")
    return image
    
value_last = {}

def apply_matchtests(identifier,key,section,value):

        # Pattern match anwenden
        if "match" in section and isinstance(value, str):
            match = re.search(section["match"], value)
            orig = value
            value = match.group(0) if match else ""
            logging.info(f"[Validation] Scanned value '{orig}' matched: '{value}'")

        # Bereichsprüfung (range)
        if "range" in section and isinstance(value, str):
            try:
                val_f = float(value)
                r_min, r_max = section["range"]
                if not (r_min <= val_f <= r_max):
                    logging.warning(f"[Validation] Value '{val_f}' for '{key}' out of range {r_min}–{r_max}")
                    value = ""
            except Exception as e:
                logging.warning(f"[Validation] Failed to check range for value '{value}': {e}")
                value = ""

        # Vorwert-Prüfung (previous)
        if "previous" in section and isinstance(value, str):
            try:
                val_f = float(value)
                prev_val = value_last.get((identifier, key))
                max_diff = section["previous"]

                if prev_val is not None and abs(val_f - prev_val) > max_diff:
                    logging.warning(f"[Validation] Value '{val_f}' for '{key}' differs too much from previous '{prev_val}' (>{max_diff})")
                    value = ""
                else:
                    value_last[(identifier, key)] = val_f
            except Exception as e:
                logging.warning(f"[Validation] Failed to check previous for value '{value}': {e}")
                value = ""

        return value
    
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
    
    # Dynamische Rotation, falls im YAML gesetzt
    
    rotation_angle = definition.get("rotate", 0)  # z. B. 90, -90, 180
    if rotation_angle != 0:
        image = image.rotate(rotation_angle, expand=True)
        
    # 🔽 Bild zur Kontrolle speichern
    image.save(f"images/{identifier}_debug.jpg")
        
    # Bildoptimierung je nach Konfiguration
    image = apply_enhancement(image, identifier, definition.get("enhance", []))
        
    # 🔽 Bild zur Kontrolle speichern
    image.save(f"images/{identifier}_Edebug.jpg")

    results = []
    
    padding = definition.get("padding", 0.0) 

    for key, section in definition.items():
        if key == "transform" or key == "rotate" or key == "enhance" or key == "padding" :
            continue  # skip transform block

        model = section.get("model", "tesseract")
        logging.info(f"[PIC Parsing] Section: {key}, Model: {model}")

        value = None

        if "predecimal" in section or "postdecimal" in section:
            digits = []
            for group in ["predecimal", "postdecimal"]:
                for rect in section.get(group, []):
                    x, y, w, h = rect
                    # segment = image.crop((x, y, x + w, y + h))
                    
                    segment = image.crop((
                        max(0, x - int(w * padding)),
                        max(0, y - int(h * padding)),
                        min(image.width,  x + w + int(w * padding)),
                        min(image.height, y + h + int(h * padding))
                    ))
                    segment.save(f"images/{identifier}_{key}_{x}{y}{w}{h}_debug.jpg")
                    
                    result = run_model(segment, model)
                    digits.append(str(result.get("class", "?")))

            vorkomma_len = len(section.get("predecimal", []))
            vorkomma_part = "".join(digits[:vorkomma_len]) or "0"
            nachkomma_part = "".join(digits[vorkomma_len:]) or "0"
            value = f"{vorkomma_part}.{nachkomma_part}"

        elif "rects" in section:
            values = []
            for rect in section["rects"]:
                x, y, w, h = rect
                segment = image.crop((x, y, x + w, y + h))
                segment.save(f"images/{identifier}_{key}_{x}{y}_debug.jpg")
                result = run_model(segment, model)
                values.append(result.get("class", ""))

            if model == "tesseract" or model.startswith("lcd-"): 
                value = "".join(str(v) for v in values if v != "")
            else:
                value = values[0]


        value = apply_matchtests(identifier, key, section, value)

        logging.info(f"[PIC Parsing] Final Value: {value}")
        results.append({
            "id": key,
            "value": value
        })

    # Transformationen anwenden, falls definiert
    transform_conf = definition.get("transform")
    if transform_conf:
        try:
            apply_transformations(transform_conf, results)
        except Exception as e:
            logging.error(f"[Transform] Failed to apply transformations: {e}")
    # MQTT senden, falls konfiguriert
    mqtt_config = config.get("mqtt")
    if mqtt_config:
        try:
            publish_results_to_mqtt(results, identifier, mqtt_config)
        except Exception as e:
            logging.error(f"[MQTT] Failed to send: {e}")

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

import os

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    
    path = os.path.join("images", filename)
    file.save(path)

    return jsonify({"filename": filename})


@app.route("/reload-config", methods=["POST"])
def reload_config():
    global config
    try:
        config = load_config()
        logging.info("[Config] Konfiguration erfolgreich neu geladen.")
        return jsonify({"status": "reloaded", "keys": list(config.keys())})
    except Exception as e:
        logging.error(f"[Config] Fehler beim Laden: {e}")
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



def publish_results_to_mqtt(results: list, identifier: str, mqtt_config: dict):
    topic = mqtt_config.get("topic", f"meters/{identifier}")
    payload = {
        "identifier": identifier,
        "values": results
    }

    publish.single(
        topic,
        payload=json.dumps(payload),
        hostname=mqtt_config["host"],
        port=mqtt_config.get("port", 1883),
        auth={
            "username": mqtt_config.get("username"),
            "password": mqtt_config.get("password")
        } if "username" in mqtt_config else None
    )
    
    logging.info(f"[MQTT:{topic}] Data:\n {payload}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

