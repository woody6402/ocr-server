... work in progress

# 🔍 Segment Editor + OCR for Meter Readings

A Flask-based web tool for interactive configuration and recognition of analog and digital meter displays – using TFLite models, Tesseract OCR, and image segmentation.

It's mainly thought o be used in HA to send a picture of a meter and to generate values on the server side.

## 🚀 Features

- 📤 Upload custom meter images
- ✏️ Interactive drawing of image segments
- ⚙️ YAML-based configuration generation
- 🧠 Supports:
  - TFLite digital models (`digital`)
  - TFLite analog models (`analog`)
  - Tesseract OCR (`tesseract`)
- 🔁 Test and save configuration directly in the browser
- 🎯 Live inference via the `/segment` API

## 🖼️ Screenshot

<img src="SegmentEditor.png" alt="Segment Editor UI" width="400"/>

## 🧑‍💻 Getting Started Locally

### Prerequisites

- Docker + Docker Compose

### Start the Application

```bash
docker-compose up --build
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🧪 Workflow

1. 📷 Upload a meter image via `/`
2. 🖱️ Draw segments using the editor (`/editor/<image>`)
3. ⚙️ Assign model type and segment group (e.g., `pre-decimal`, `post-decimal`, or `rects`)
4. 💾 Save the YAML config
5. 🔍 Click "Scan" to test inference with the selected config

## ⚙️ API Endpoints

| URL                 | Purpose                        |
|---------------------|--------------------------------|
| `/`                 | Upload a new image             |
| `/editor/<image>`   | Interactive segment editor     |
| `/segment`          | Run inference on an image      |
| `/test-config`      | Test/save YAML configuration   |
| `/images/<file>`    | Access uploaded images         |

## 📁 Project Structure

```
.
├── app.py              # Flask app with OCR and inference logic
├── transformation.py   # Customer specific transformation
├── templates/editor.html  # Browser-based segment editor
├── config/
│   └── config.yaml    # YAML config storage  
├── models/             # .tflite models
├── images/             # Uploaded images
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## 🔧 YAML-Konfiguration für OCR-Segmentierung

Die Datei `config.yaml` definiert, wie Bildsegmente ausgeschnitten, mit Modellen verarbeitet und die Ergebnisse validiert werden. Jedes Objekt ist einem `identifier` zugeordnet, z. B. `"uhr-1"`.

### 📁 Struktur

```yaml
<identifier>:
  rotate: -90               # optional: Bildrotation in Grad (z. B. -90)
  padding: 0.1              # optional: 10 % Puffer um jedes Rechteck

  enhance:                  # optional: Bildverbesserung in definierter Reihenfolge
    - grayscale
    - autocontrast
    - contrast: 1.5
    - threshold: 120

  <key>:                   # z. B. "temp", "zeit"
    model: tesseract       # z. B. "tesseract", "dig-class100", "lcd-s1"
    rects:                 # Liste der Bildsegmente als [x, y, w, h]
      - [100, 200, 40, 30]

    match: \d+(\.\d+)?   # optional: RegEx zur Filterung
    range: [22.0, 28.0]    # optional: gültiger Wertebereich
    previous: 2.0          # optional: max. Abweichung vom letzten gültigen Wert

    # alternativ für klassische Zähler:
    predecimal:
      - [x, y, w, h]       # Ziffern vor dem Dezimalpunkt
    postdecimal:
      - [x, y, w, h]       # Ziffern nach dem Dezimalpunkt
```

### ✅ Validierungsoptionen

| Option     | Beschreibung                                                               |
|------------|----------------------------------------------------------------------------|
| `match`    | RegEx zur Filterung des erkannten Strings                                  |
| `range`    | Erwarteter Zahlenbereich `[min, max]`                                      |
| `previous` | Max. Abweichung zum vorherigen gültigen Wert (z. B. zur Glättung)          |

### 📌 Beispiel

```yaml
uhr-1:
  rotate: -90
  enhance:
    - grayscale
    - autocontrast
  padding: 0.05

  zeit:
    model: dig-class100
    predecimal:
      - [100, 200, 20, 30]
      - [130, 200, 20, 30]
    postdecimal:
      - [160, 200, 20, 30]
    range: [0.0, 9.9]
    previous: 1.0

  temp:
    model: tesseract
    rects:
      - [50, 100, 60, 25]
    match: \d+(\.\d+)?
    range: [15.0, 35.0]
```

### 🔄 Live-Konfigurations-Reload

Die Konfiguration kann zur Laufzeit neu geladen werden:

```bash
curl -X POST http://<host>:5000/test-config?save=true \
     --data-binary @config.yaml \
     -H "Content-Type: text/yaml"
```

## 🔗 Related Projects

- [jomjol/AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) – OCR system for ESP32 camera-based devices

## 📖 License

MIT License – free to use with attribution.\
the tflite files are under ownership of AI-on-the-edge-device and others and the corresponding rules do apply ...

## Todos
- send generated values to a MQTT server  - done V0.91
- allow match: <regex> patterns to validate resulsts . done V0.92.1
- allow custom code to work on scanned values and to modify return values. done V0.92.1
- allow simplified model names in config.yaml. done V0.92.1
- correct handling of digital models: -cont, -11, 100. done V0.92.1
- add a possible rotation to the image. done V0.92.1
- add matching criteria range V0.92.3
- add image filter: enhance (contrast, sharpen, brightness, sharpen, autocontrast): V0.92.3
- add padding (exact image crop and adjustable bar around): V0.92.3


