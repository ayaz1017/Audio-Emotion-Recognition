from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import pickle
import os
import uuid
import math
from datetime import datetime

from database import init_db, get_db
from llm_summary import generate_clinical_summary

# -------------------------
# APP INIT
# -------------------------
app = Flask(__name__)
init_db()

MODEL_PATH = "emotion_detection_model.h5"
ENCODER_PATH = "label_encoder.pkl"

model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# -------------------------
# AUDIO FEATURE EXTRACTION
# -------------------------
def extract_features_file(file_path, duration=2.5, offset=0.0, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
        if y.size == 0:
            return None

        mfcc = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
            axis=0
        )
        return mfcc
    except Exception as e:
        print("Audio processing error:", e)
        return None

# -------------------------
# SPEECH BIOMARKERS
# -------------------------
def extract_speech_biomarkers(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    rms_energy = float(np.mean(librosa.feature.rms(y=y)))
    silence_ratio = round(np.sum(np.abs(y) < 0.01) / len(y), 3)

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    pitch_variability = round(float(np.std(pitch_vals)) if len(pitch_vals) else 0.0, 2)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    speech_rate = round(
        len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))
        / (len(y) / sr),
        2
    )

    return {
        "rms_energy": round(rms_energy, 4),
        "silence_ratio": silence_ratio,
        "pitch_variability": pitch_variability,
        "speech_rate": speech_rate
    }

# -------------------------
# CLINICAL ANALYSIS
# -------------------------
def compute_emotional_instability(timeline):
    if len(timeline) < 2:
        return 0.0

    confs = [t["confidence"] for t in timeline]
    emotions = [t["emotion"] for t in timeline]

    conf_var = np.std(confs)
    switches = sum(
        1 for i in range(1, len(emotions))
        if emotions[i] != emotions[i - 1]
    )

    return round((0.6 * conf_var) + (0.4 * (switches / len(emotions))), 3)

def analyze_trend(history):
    if len(history) < 2:
        return "Insufficient data for trend analysis"

    inst = [h["instability"] for h in history]

    if inst[0] > inst[-1]:
        return "Emotional instability increasing over time"
    elif inst[0] < inst[-1]:
        return "Emotional stability improving over time"
    return "Emotional state appears stable"

def detect_risks(timeline, instability, speech):
    risks = []
    negative = {"sad", "angry", "fear", "disgust"}

    neg_ratio = sum(
        1 for t in timeline if t["emotion"] in negative
    ) / max(len(timeline), 1)

    if neg_ratio > 0.6:
        risks.append("Sustained negative emotional affect")

    if instability > 0.35:
        risks.append("High emotional volatility")

    if speech["silence_ratio"] > 0.4:
        risks.append("Prolonged silence / withdrawal")

    if speech["rms_energy"] < 0.02:
        risks.append("Low vocal energy (flat affect)")

    return risks

# -------------------------
# DATABASE HELPERS
# -------------------------
def get_patient_history(patient_id):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT session_date, dominant_emotion,
               avg_confidence, instability_score
        FROM sessions
        WHERE patient_id = ?
        ORDER BY session_date DESC
        LIMIT 5
    """, (patient_id,))

    rows = cur.fetchall()
    conn.close()

    return [{
        "date": r[0],
        "emotion": r[1],
        "confidence": round(r[2], 2),
        "instability": round(r[3], 3)
    } for r in rows]

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_upload", methods=["POST"])
def predict_upload():

    patient_id = request.form.get("patient_id")
    if not patient_id:
        return "Patient ID missing.", 400

    file = request.files.get("file")
    if not file or file.filename == "":
        return "No audio file uploaded.", 400

    # 🔒 AUDIO-ONLY SAFETY CHECK
    if not file.filename.lower().endswith((".wav", ".mp3")):
        return "Only .wav or .mp3 audio files are supported.", 400

    os.makedirs("static", exist_ok=True)
    filepath = os.path.join("static", f"{uuid.uuid4().hex}_{file.filename}")
    file.save(filepath)

    # -------------------------
    # EMOTION PREDICTION
    # -------------------------
    feats = extract_features_file(filepath, duration=3.0, offset=0.5)
    if feats is None:
        return "Audio processing failed.", 500

    pred = model.predict(np.expand_dims(feats, axis=0))
    idx = int(np.argmax(pred))
    emotion_full = encoder.inverse_transform([idx])[0]
    conf_full = float(np.max(pred))

    # -------------------------
    # TIMELINE ANALYSIS
    # -------------------------
    timeline = []
    total_dur = librosa.get_duration(filename=filepath)
    chunk = 3.0

    for i in range(math.ceil(total_dur / chunk)):
        start = round(i * chunk, 2)
        f = extract_features_file(filepath, duration=chunk, offset=start)
        if f is None:
            continue

        p = model.predict(np.expand_dims(f, axis=0))
        idx = int(np.argmax(p))

        timeline.append({
            "start": start,
            "end": round(min(start + chunk, total_dur), 2),
            "emotion": encoder.inverse_transform([idx])[0],
            "confidence": round(float(np.max(p)), 3)
        })

    # -------------------------
    # CLINICAL PIPELINE
    # -------------------------
    speech = extract_speech_biomarkers(filepath)
    instability = compute_emotional_instability(timeline)
    risks = detect_risks(timeline, instability, speech)

    # -------------------------
    # SAVE SESSION
    # -------------------------
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sessions (
            patient_id, session_date, dominant_emotion,
            avg_confidence, instability_score,
            silence_ratio, speech_rate
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        emotion_full,
        conf_full,
        instability,
        speech["silence_ratio"],
        speech["speech_rate"]
    ))
    conn.commit()
    conn.close()

    history = get_patient_history(patient_id)
    trend = analyze_trend(history)

    clinical_summary = generate_clinical_summary(
        patient_id,
        emotion_full,
        instability,
        speech,
        risks,
        trend,
        history
    )

    return render_template(
        "result.html",
        patient_id=patient_id,
        prediction_text=emotion_full,
        prediction_conf=round(conf_full, 2),
        emotion_timeline=timeline,
        instability=instability,
        speech=speech,
        risks=risks,
        history=history,
        trend=trend,
        clinical_summary=clinical_summary
    )

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
