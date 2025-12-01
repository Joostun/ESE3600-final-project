#!/usr/bin/env python3
import io
import os
import re
import threading
from datetime import datetime

from flask import Flask, render_template, make_response, jsonify, request
import serial
import cv2
import numpy as np
import base64

# ====== CONFIGURE THIS ======
SERIAL_PORT = "/dev/ttyACM0"  # change if needed
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2.0          # seconds
TRAINING_ROOT = "training_data"
# ============================

app = Flask(__name__)
os.makedirs(TRAINING_ROOT, exist_ok=True)

# ---- Serial setup ----
ser_lock = threading.Lock()
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)

# ---- Global state for labeling (now grayscale) ----
current_image = None     # enhanced grayscale image as numpy array (H, W)
current_chunk_size = 40  # default now 40x40
current_rows = 0
current_cols = 0


def read_line():
    """Read one line (without trailing \r\n) from serial."""
    line = ser.readline()
    if not line:
        raise RuntimeError("No data from serial (timeout?)")
    return line.decode("utf-8", errors="ignore").strip()


def read_exact(n):
    """Read exactly n bytes from serial."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise RuntimeError("Serial returned no data while reading image")
        buf.extend(chunk)
    return bytes(buf)


def capture_frame_from_esp():
    """Send 'r' to ESP32, read JPEG bytes framed as STARTIMG/ENDIMG."""
    with ser_lock:
        ser.reset_input_buffer()
        ser.write(b"r")

        size = None
        while size is None:
            line = read_line()
            m = re.match(r"^STARTIMG\s+(\d+)", line)
            if m:
                size = int(m.group(1))

        jpeg_bytes = read_exact(size)

        try:
            while True:
                line = read_line()
                if line == "ENDIMG":
                    break
        except Exception:
            pass

        return jpeg_bytes


def enhance_image(gray):
    """
    Stronger enhancement in grayscale:
    - CLAHE for local contrast
    - aggressive unsharp mask
    """
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    # Unsharp mask (stronger weights)
    gauss = cv2.GaussianBlur(eq, (0, 0), 1.5)
    sharp = cv2.addWeighted(eq, 1.8, gauss, -0.8, 0)

    # Clip to [0,255] & uint8
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp


@app.route("/")
def index():
    return render_template("index.html")


# ---------- LIVE VIEW ENDPOINT (raw color JPEG from ESP32) ----------
@app.route("/frame")
def frame():
    """Return one raw JPEG frame directly from the ESP32 (for preview)."""
    try:
        jpeg_bytes = capture_frame_from_esp()
    except Exception as e:
        resp = make_response(str(e), 500)
        return resp

    resp = make_response(jpeg_bytes)
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


# ---------- LABELING PREP ----------
@app.route("/prepare_labeling", methods=["POST"])
def prepare_labeling():
    """
    Capture a frame from ESP32, convert to grayscale, enhance it, store
    globally, compute chunk grid, and return metadata.
    """
    global current_image, current_chunk_size, current_rows, current_cols

    chunk_size = int(request.form.get("chunk_size", "40"))  # default 40

    try:
        jpeg_bytes = capture_frame_from_esp()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Decode JPEG to BGR
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to decode JPEG from ESP32"}), 500

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance more aggressively in grayscale
    enhanced = enhance_image(gray)

    h, w = enhanced.shape
    rows = (h + chunk_size - 1) // chunk_size
    cols = (w + chunk_size - 1) // chunk_size

    current_image = enhanced
    current_chunk_size = chunk_size
    current_rows = rows
    current_cols = cols

    return jsonify({
        "width": w,
        "height": h,
        "rows": rows,
        "cols": cols,
        "chunk_size": chunk_size
    })


@app.route("/enhanced_full")
def enhanced_full():
    """Serve enhanced grayscale image as JPEG."""
    global current_image
    if current_image is None:
        return "No enhanced image available", 400
    ok, buf = cv2.imencode(".jpg", current_image)
    if not ok:
        return "Encode failed", 500
    resp = make_response(buf.tobytes())
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


# ---------- CHUNK FETCH ----------
@app.route("/get_chunk")
def get_chunk():
    global current_image, current_chunk_size, current_rows, current_cols

    if current_image is None:
        return "No image loaded", 400

    row = int(request.args.get("row", "0"))
    col = int(request.args.get("col", "0"))

    h, w = current_image.shape
    cs = current_chunk_size

    y0 = row * cs
    x0 = col * cs
    y1 = min(y0 + cs, h)
    x1 = min(x0 + cs, w)

    if y0 >= h or x0 >= w:
        return "Chunk out of bounds", 400

    chunk = current_image[y0:y1, x0:x1]

    ok, buf = cv2.imencode(".jpg", chunk)
    if not ok:
        return "Encode failed", 500

    b64 = base64.b64encode(buf).decode("ascii")
    data_url = "data:image/jpeg;base64," + b64

    return jsonify({
        "data_url": data_url,
        "row": row,
        "col": col,
        "width": int(x1 - x0),
        "height": int(y1 - y0)
    })


# ---------- LABEL CHUNK ----------
@app.route("/label_chunk", methods=["POST"])
def label_chunk():
    global current_image, current_chunk_size

    if current_image is None:
        return jsonify({"error": "No image loaded"}), 400

    data = request.get_json()
    row = int(data["row"])
    col = int(data["col"])
    label = data["label"]

    safe_label = label.replace("/", "_")
    label_dir = os.path.join(TRAINING_ROOT, safe_label)
    os.makedirs(label_dir, exist_ok=True)

    h, w = current_image.shape
    cs = current_chunk_size

    y0 = row * cs
    x0 = col * cs
    y1 = min(y0 + cs, h)
    x1 = min(x0 + cs, w)

    if y0 >= h or x0 >= w:
        return jsonify({"error": "Chunk out of bounds"}), 400

    chunk = current_image[y0:y1, x0:x1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{label}_{row}_{col}_{timestamp}.jpg"
    out_path = os.path.join(label_dir, fname)

    cv2.imwrite(out_path, chunk)

    return jsonify({
        "status": "ok",
        "saved_to": out_path
    })


if __name__ == "__main__":
    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE}...")
    app.run(host="0.0.0.0", port=5000, debug=True)
