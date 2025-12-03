# =============================================================================
# LLM Explanation Generator for Organ Detection XAI
# Purpose: Turn stored XAI metadata into patient-friendly explanations via Hugging Face
# Environment: Python with requests
# Date: March 2025
# =============================================================================

import json
import os
import requests

XAI_OUTPUT_FILE = "outputs/xai_output.json"
API_URL = os.getenv("HF_CHAT_API_URL", "https://router.huggingface.co/v1/chat/completions")
ROUTER_MODEL = os.getenv("HF_ROUTER_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
HUGGING_FACE_API_TOKEN = os.getenv("HF_TOKEN")

if not HUGGING_FACE_API_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is required to call Hugging Face APIs.")


def build_prompts(class_name: str, organ: str, confidence: float, xai_insights: str):
    organ_label = organ or "the affected area"
    pretty_class = class_name.replace("_", " ").title()
    is_normal = any(token in class_name.lower() for token in ["normal", "healthy", "benign"]) or "no tumor" in class_name.lower()

    if is_normal:
        patient_prompt = (
            "You are a renal/oncology nurse explaining AI screening results to a patient. "
            f"Summarize in two short paragraphs (30-40 words each) that the AI is {confidence:.1f}% confident the {organ_label.lower()} looks healthy ({pretty_class}). "
            f"Reference the supporting evidence from Grad-CAM / Saliency / LIME ({xai_insights}). "
            "Offer practical reassurance (hydration, follow-up) and close with a hopeful sentence."
        )
    else:
        patient_prompt = (
            "You are a renal/oncology nurse explaining AI screening results to a patient. "
            f"Summarize in two short paragraphs (30-40 words each) that the AI detected {pretty_class} in the {organ_label.lower()} with {confidence:.1f}% confidence. "
            f"Reference the highlighted region mentioned in the XAI notes ({xai_insights}). "
            "Encourage discussing next steps with a specialist and end with hopeful language."
        )
    return patient_prompt


def get_llm_explanation(class_name: str, organ: str, xai_insights: str, confidence: float) -> str:
    patient_prompt = build_prompts(class_name, organ, confidence, xai_insights)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}",
    }
    payload = {
        "model": ROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You translate medical AI findings into short, reassuring explanations that mention Grad-CAM, Saliency, and LIME when relevant."
                ),
            },
            {"role": "user", "content": patient_prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 300,
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Error calling Hugging Face Router API: {exc}")
        return "Sorry, I couldn’t generate an explanation right now."
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        print(f"Unexpected response structure: {data}")
        return "Sorry, the LLM response was malformed."


def main():
    print("Starting organ explanation generator...")
    try:
        with open(XAI_OUTPUT_FILE, "r") as f:
            xai_entries = json.load(f)
            if not isinstance(xai_entries, list):
                xai_entries = [xai_entries]
    except FileNotFoundError:
        print(f"Error: {XAI_OUTPUT_FILE} not found. Run the test script first.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {XAI_OUTPUT_FILE}.")
        return

    for entry in xai_entries:
        class_name = entry.get("class_name", "Unknown Class")
        organ = entry.get("organ", "the affected area")
        confidence = float(entry.get("confidence", 0.0))
        xai_insights = entry.get("xai_insights", "")
        explanation = get_llm_explanation(class_name, organ, xai_insights, confidence)
        print(f"\nImage {entry.get('image_index', 'N/A')} ({organ}): {class_name}")
        print(explanation)


if __name__ == "__main__":
    main()
# =============================================================================
