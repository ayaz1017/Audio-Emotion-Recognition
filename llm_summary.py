from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
from openai import RateLimitError, OpenAIError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fallback_summary(
    patient_id,
    dominant_emotion,
    instability,
    speech,
    risks,
    trend
):
    text = f"""
Clinical Session Summary (Rule-Based Fallback)

Patient ID: {patient_id}

Observed dominant emotion: {dominant_emotion}
Emotional instability score: {instability}

Speech characteristics:
- Speech rate: {speech['speech_rate']}
- Silence ratio: {speech['silence_ratio']}
- RMS energy: {speech['rms_energy']}
- Pitch variability: {speech['pitch_variability']}

Observed trend:
{trend}

Risk indicators:
{', '.join(risks) if risks else 'No significant behavioral risks observed.'}

Note:
This summary was generated without an LLM due to API limitations.
It is intended for observational support only and does not provide diagnosis.
"""
    return text.strip()


def generate_clinical_summary(
    patient_id,
    dominant_emotion,
    instability,
    speech,
    risks,
    trend,
    history
):
    prompt = f"""
You are a psychiatric clinical documentation assistant.
Generate a professional, non-diagnostic session summary.

Patient ID: {patient_id}
Dominant emotion: {dominant_emotion}
Emotional instability score: {instability}

Speech:
- Rate: {speech['speech_rate']}
- Silence ratio: {speech['silence_ratio']}
- Energy: {speech['rms_energy']}
- Pitch variability: {speech['pitch_variability']}

Trend:
{trend}

Risks:
{', '.join(risks) if risks else 'None'}

Rules:
- Do NOT diagnose
- Use clinical, cautious language
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You write psychiatric session notes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        # 🔴 QUOTA EXHAUSTED
        return fallback_summary(
            patient_id,
            dominant_emotion,
            instability,
            speech,
            risks,
            trend
        )

    except OpenAIError as e:
        # 🔴 ANY OTHER OPENAI FAILURE
        return fallback_summary(
            patient_id,
            dominant_emotion,
            instability,
            speech,
            risks,
            trend
        )