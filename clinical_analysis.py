import numpy as np
from collections import Counter


def compute_emotional_instability(timeline):
    """
    Measures emotional volatility based on:
    - Confidence variance
    - Emotion switching frequency
    """
    if not timeline or len(timeline) < 2:
        return 0.0

    confidences = [t["confidence"] for t in timeline]
    emotions = [t["emotion"] for t in timeline]

    confidence_variance = np.std(confidences)

    switches = sum(
        1 for i in range(1, len(emotions))
        if emotions[i] != emotions[i - 1]
    )
    switch_rate = switches / len(emotions)

    instability_score = round(
        (0.6 * confidence_variance) + (0.4 * switch_rate), 3
    )

    return instability_score


def detect_psychiatric_risk(timeline, instability_score):
    """
    Rule-based clinical risk indicators
    (NOT diagnosis)
    """
    risk_flags = []

    if not timeline:
        return risk_flags

    emotion_counts = Counter(t["emotion"] for t in timeline)
    total = sum(emotion_counts.values())

    negative_emotions = {"sad", "angry", "fear", "disgust"}
    negative_ratio = sum(
        emotion_counts[e] for e in negative_emotions
        if e in emotion_counts
    ) / total

    if negative_ratio > 0.6:
        risk_flags.append("Sustained negative affect")

    if instability_score > 0.35:
        risk_flags.append("High emotional volatility")

    if len(set(emotion_counts.keys())) == 1:
        risk_flags.append("Emotion rigidity detected")

    return risk_flags



def generate_clinical_summary(timeline, instability_score, risk_flags):
    """
    Human-readable insights for psychiatrists
    """
    summary = {
        "instability_score": instability_score,
        "stability_level": (
            "High Risk" if instability_score > 0.4 else
            "Moderate" if instability_score > 0.25 else
            "Stable"
        ),
        "risk_flags": risk_flags,
        "note": (
            "This analysis provides behavioral indicators only "
            "and is not a medical diagnosis."
        )
    }

    return summary
