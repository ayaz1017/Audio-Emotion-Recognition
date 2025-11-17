from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import pickle
import os
import uuid
import math

app = Flask(__name__)

MODEL_PATH = "emotion_detection_model.h5"
ENCODER_PATH = "label_encoder.pkl"

# Load model + encoder
model = load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# ------------------------------
# Extract 40 MFCC audio features
# ------------------------------
def extract_features_file(file_path, duration=2.5, offset=0.0, sr=22050):
    try:
        data, sample_rate = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
        if data.size == 0:
            return None

        mfccs = np.mean(librosa.feature.mfcc(
            y=data, sr=sample_rate, n_mfcc=40
        ).T, axis=0)

        return mfccs
    except Exception as e:
        print("extract_features error:", e)
        return None

# ------------------------------
# HOME PAGE
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ------------------------------
# FULL AUDIO UPLOAD PREDICTION + TIMELINE (3s chunks)
# ------------------------------
@app.route("/predict_upload", methods=["POST"])
def predict_upload():

    if "file" not in request.files:
        return render_template("index.html", upload_msg="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", upload_msg="No file selected")

    os.makedirs("static", exist_ok=True)
    filename = f"upload_{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join("static", filename)
    file.save(filepath)

    # === Full-file single prediction (original behavior) ===
    feats_full = extract_features_file(filepath, duration=3.0, offset=0.5)
    if feats_full is None:
        # still try timeline approach below, but return error if nothing worked
        return render_template("index.html", upload_msg="Could not process audio")

    feats_full = np.expand_dims(feats_full, axis=0)
    pred_full = model.predict(feats_full)
    idx_full = int(np.argmax(pred_full))
    conf_full = float(np.max(pred_full))
    emotion_full = encoder.inverse_transform([idx_full])[0]

    # === Build timeline: chunk audio every 3.0 seconds, starting at 0.0 ===
    try:
        total_duration = librosa.get_duration(filename=filepath)
    except Exception as e:
        print("Could not get duration:", e)
        total_duration = None

    timeline = []
    if total_duration:
        chunk_dur = 3.0
        num_chunks = math.ceil(total_duration / chunk_dur)
        for i in range(num_chunks):
            start = round(i * chunk_dur, 3)
            end = round(min((i + 1) * chunk_dur, total_duration), 3)

            feats_chunk = extract_features_file(filepath, duration=chunk_dur, offset=start)
            if feats_chunk is None:
                # skip empty chunk (could be last very short slice)
                continue

            x = np.expand_dims(feats_chunk, axis=0)
            pred = model.predict(x)
            idx = int(np.argmax(pred))
            conf = float(np.max(pred))
            emotion = encoder.inverse_transform([idx])[0]

            timeline.append({
                "start": float(start),
                "end": float(end),
                "emotion": str(emotion),
                "confidence": float(conf)
            })

    # Optionally remove saved upload after processing (comment out if you want to keep file)
    # try:
    #     os.remove(filepath)
    # except:
    #     pass

    return render_template(
        "result.html",
        prediction_text=emotion_full,
        prediction_conf=f"{conf_full:.2f}",
        emotion_timeline=timeline  # pass list-of-dicts; template will use tojson
    )

# ------------------------------
# LIVE AUDIO CHUNK PREDICTION (unchanged)
# ------------------------------
@app.route("/predict_chunk", methods=["POST"])
def predict_chunk():

    if "chunk" not in request.files:
        return jsonify({"error": "No audio chunk"}), 400

    ch = request.files["chunk"]

    os.makedirs("temp_chunks", exist_ok=True)
    fname = f"temp_chunks/{uuid.uuid4().hex}.webm"
    ch.save(fname)

    feats = extract_features_file(fname, duration=2.5)

    try:
        os.remove(fname)
    except:
        pass

    if feats is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    x = np.expand_dims(feats, axis=0)
    pred = model.predict(x)

    idx = int(np.argmax(pred))
    conf = float(np.max(pred))
    emotion = encoder.inverse_transform([idx])[0]

    return jsonify({
        "emotion": emotion,
        "confidence": conf
    })


if __name__ == "__main__":
    app.run(debug=True)
