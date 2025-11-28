# Cancer Detection XAI

Real-time cancer detection assistant (March 2025) that ingests MRI images, classifies them into eight cancer types, explains *how* the model decided via Grad-CAM/Saliency/LIME, and translates those findings into language tailored for patients and clinical engineers. Everything ships in a responsive PyQt6 GUI backed by TensorFlow/Keras, OpenCV, and Hugging Face LLMs.

## Project Quick Facts

| Item | Details |
| --- | --- |
| **Purpose** | Detect cancer in real time, visualize the evidence, and communicate the result clearly to both patients and medical staff. |
| **Dataset** | Multi Cancer Dataset (folder-per-class MRI images for ALL, Brain, Breast, Cervical, Kidney, Lung & Colon, Lymphoma, Oral). |
| **Tech Stack** | Python 3.11 (3.8+ may work), PyQt6, TensorFlow/Keras, OpenCV, NumPy, LIME, tf-keras-vis, scikit-learn, matplotlib, Hugging Face LLM API. |
| **GUI Features** | Tabs for patients vs. doctors, progress bars, theming, webcam capture, tooltips, privacy note. |
| **Outputs** | `outputs/*.json` logs for real-time inference, metrics, and batch explanations. |

## Feature Overview

| Feature | Description |
| --- | --- |
| **Cancer prediction** | Classifies MRI images into eight cancer types with calibrated confidence scores. |
| **Explainable AI visuals** | Produces Grad-CAM, Saliency Map, and LIME overlays to highlight the critical tissue regions driving the prediction. |
| **Dual-language explanations** | Generates patient-friendly summaries plus technical briefs for clinicians/engineers via Hugging Face chat completions. |
| **Real-time input** | Accepts single image uploads or webcam frames so you can iterate quickly in the GUI. |
| **Responsive PyQt6 GUI** | Tabbed layout, progress indicators, help modal, and light/dark themes optimized for clinical settings. |
| **Testing & training suite** | Scripts to validate an existing .h5 model, train VGG16 from scratch, or batch-produce LLM explanations. |

## Dataset

- Source: [Multi Cancer Dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer), organized with folder-per-class subdirectories such as `data/Multi_Cancer/Breast Cancer/…`.
- Images: MRI slices and microscopy captures, often colorized; this project resizes them to 224×224 RGB and normalizes to [0, 1].
- Classes: Acute Lymphoblastic Leukemia (ALL), Brain, Breast, Cervical, Kidney, Lung & Colon, Lymphoma, Oral.

## System Architecture

1. **Data ingestion** – `scripts/train_model_with_xai.py` samples from the dataset, applies augmentation (rotations, shifts, flips), and splits into train/val/test sets.
2. **Deep learning model** – VGG16 (ImageNet weights) acts as the feature extractor; a custom GAP + Dense + softmax head performs the 8-way classification. Training freezes the convolutional base by default for stability on smaller datasets.
3. **Explainability stack** – tf-keras-vis Grad-CAM, TensorFlow saliency maps, and LIME superpixel perturbations run off the same tensors to give corroborating views.
4. **LLM reasoning** – `_chat_completion` (in `src/cancer_detection_gui.py`) calls Hugging Face’s `/v1/chat/completions` router using the token stored in `.env`. Prompts include predicted class, confidence, and XAI evidence so the LLM crafts two narratives: one patient-friendly, one technical.
5. **Presentation layer** – PyQt6 GUI orchestrates image capture, inference, XAI rendering, LLM calls, and metrics logging to JSON. Tabs separate patient info from engineering insights.

## Deep Learning Architecture Explained

- **What is VGG16?** A 16-weight-layer CNN from Oxford’s Visual Geometry Group with stacked 3×3 convolutions followed by max pooling. It is simple, well-studied, and its ImageNet pretraining offers rich edge/texture detectors even for medical imagery.
- **Why choose it here?**
  - Medical datasets are often small; VGG16’s transferable filters reduce overfitting.
  - Freezing most convolutional blocks keeps inference fast and predictable; only the new dense head is retrained.
  - The architecture is compatible with Grad-CAM, making XAI integration straightforward.
- **Head customization:** We replace VGG16’s classifier with `GlobalAveragePooling2D -> Dense(128, relu) -> Dense(8, softmax)` for the eight cancer classes.
- **Training recipe:** Adam optimizer, sparse categorical cross-entropy, 224×224 inputs, 32-image batches, optional augmentation via `ImageDataGenerator`.

## Explainable AI (XAI) in Detail

| Technique | How it works | Why it helps |
| --- | --- | --- |
| **Grad-CAM** | Computes gradients of the target class w.r.t. the final conv layer, weights the feature maps, and produces a coarse heatmap. | Shows which macroscopic regions (tumor cores, boundaries) influenced the decision. |
| **Saliency Map** | Uses TensorFlow gradient tape to measure pixel-level sensitivity. | Highlights fine structures such as lesion edges, vasculature, or nuclei textures. |
| **LIME** | Perturbs superpixels, observes prediction changes, and overlays the most influential regions. | Validates Grad-CAM via an interpretable local surrogate model, increasing trust. |

Multiple methods help clinicians triangulate the model’s reasoning instead of trusting a single visualization.

## Why Hugging Face?

- **Consistent LLM interface** – The router endpoint keeps prompts stable across machines while letting you choose models (default: `meta-llama/Llama-3.2-3B-Instruct:novita`).
- **Narrative clarity** – Vision models speak probabilities; LLMs translate them into empathetic narratives for patients and technical summaries for engineers.
- **Configurable** – Change `HF_CHAT_MODEL` or point to an on-prem endpoint just by editing `.env`. `_chat_completion` will reuse the same interface.

## Installation

**Prerequisites**
- Python 3.11 recommended (3.8+ may work).
- Git.
- Optional webcam for real-time capture.

**Steps**
1. **Clone the repo**
   ```bash
   git clone https://github.com/fishe47/real-time-cancer-detection-xai.git
   cd real-time-cancer-detection-xai
   ```
2. **Create & activate a virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   # or
   .venv\Scripts\activate           # Windows PowerShell or CMD
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Configure environment variables** (loaded automatically by `python-dotenv`)
   ```bash
   cp .env .env.backup   # optional
   # Edit .env:
   HF_TOKEN=hf_your_token_here          # required for Hugging Face API
   HF_CHAT_API_URL=https://router.huggingface.co/v1/chat/completions
   HF_CHAT_MODEL=meta-llama/Llama-3.2-3B-Instruct:novita
   ```
5. **Add data/model assets**
   - Place `cancer_classifier_xai.h5` in `models/`.
   - (Optional) Populate `data/Multi_Cancer/` with a subset of the dataset if you plan to run tests or retrain.

## Usage

### Run the GUI
```bash
source .venv/bin/activate
python src/cancer_detection_gui.py
```
1. **Patients tab** – Upload an MRI image or capture via webcam, click “Process Image,” and read the patient-friendly summary plus highlight overlays.
2. **Doctors/Engineers tab** – Inspect Grad-CAM, Saliency, LIME panels, raw probabilities, performance metrics, and the technical LLM explanation.
3. **Outputs** – Results persist to `outputs/real_time_xai_output.json` and `outputs/performance_metrics.json`.

### Test the model quickly
```bash
python tests/test_with_h5.py
```
- Uses one image per class (requires `data/Multi_Cancer/` sample and `models/cancer_classifier_xai.h5`).
- Generates `outputs/xai_output.json` and `outputs/performance_metrics.json`, useful for regression checks.

### Generate standalone LLM explanations
```bash
python scripts/llm_explanation_generator.py
```
- Consumes `outputs/xai_output.json` (from the test script) and produces patient-friendly narratives without opening the GUI.

### Train or fine-tune a model
```bash
python scripts/train_model_with_xai.py
```
- Samples each class (`SAMPLE_SIZE_PER_CLASS` default 50), augments, trains the VGG16-based model, visualizes XAI on a test sample, and saves to `models/cancer_classifier_xai.h5`. Use caution if you already have a tuned model.

## How the Pipeline Works (Step-by-Step)
1. **Image acquisition** – GUI loads from disk or webcam.
2. **Preprocessing** – Resize to 224×224, convert BGR→RGB, scale to [0,1].
3. **Prediction** – Forward pass through VGG16-based model to obtain softmax probabilities.
4. **XAI generation** – Grad-CAM, Saliency Map, and LIME run on the same tensor batch.
5. **LLM prompting** – `_chat_completion` sends structured prompts containing class name, confidence, and XAI notes; returns two narratives.
6. **Display/logging** – GUI renders heatmaps + explanations and writes JSON outputs for auditing or downstream scripts.

## Project Structure

## Configuration Reference

| File | Key settings |
| --- | --- |
| `src/cancer_detection_gui.py` | `MODEL_PATH`, `OUTPUT_FILE`, `PERFORMANCE_METRICS_FILE`, `HF_TOKEN` (via `.env`), `HF_CHAT_API_URL`, `HF_CHAT_MODEL`. |
| `tests/test_with_h5.py` | `DATA_DIR`, `MODEL_PATH`. |
| `scripts/llm_explanation_generator.py` | `XAI_OUTPUT_FILE`, `OUTPUT_FILE`, Hugging Face token/env vars. |
| `scripts/train_model_with_xai.py` | `DATA_DIR`, `SAMPLE_SIZE_PER_CLASS`, `IMG_SIZE`, training hyperparameters. |

## Use Cases

- **Clinical decision support** – Triage cases with immediate visual + textual evidence before escalating to oncologists.
- **Research & education** – Demonstrate how XAI + LLMs can make medical AI interpretable for posters, lectures, or papers.
- **Model QA/auditing** – Generate overlays and metrics per release to satisfy regulatory or internal review requirements.
- **Dataset annotation helper** – Spot mislabeled samples by cross-referencing predicted class, heatmaps, and explanations.

## Operational Notes

- `.env` is ignored; never commit Hugging Face tokens. Rotate immediately if leaked.
- Large folders (`data/`, `models/`, `outputs/`) are local by default; sync only what’s safe to share.
- macOS/Apple Silicon can leverage `tensorflow-metal` (already in requirements). For NVIDIA GPUs install CUDA/cuDNN as needed.
- Replace the LLM endpoint with on-prem inference if your environment forbids outbound API calls.

## Acknowledgments

- **Dataset** – [Multi Cancer Dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer).
- **Tools** – TensorFlow, Keras, PyQt6, LIME, tf-keras-vis, scikit-learn, Hugging Face, OpenCV, matplotlib.
- **Community** – Thanks to the open-source contributors who maintain these libraries and datasets.
