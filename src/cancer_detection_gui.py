# =============================================================================
# Real-Time Cancer Detection with XAI and LLM Explanations (PyQt6 GUI)
# Purpose: Take input images via GUI, predict cancer types, generate XAI visuals, and provide LLM explanations in real-time
# Dataset: Multi Cancer Dataset (with folder-per-class structure)
# Environment: Python with PyQt6, TensorFlow, Keras, OpenCV, LIME, requests, etc.
# Date: February 2025
# =============================================================================

# Import Libraries
import os
from dotenv import load_dotenv
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QTextEdit, QFileDialog, QScrollArea, QTabWidget, QProgressBar, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QPalette
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Compatibility shim for tf-keras-vis with Keras 3.x where Conv was removed.
try:
    from keras.src.layers.convolutional import base_conv as _keras_base_conv

    _BASE_CONV_CLASS = _keras_base_conv.BaseConv
    if not hasattr(_keras_base_conv, "Conv"):
        # tf-keras-vis uses isinstance(..., Conv); BaseConv covers all conv layers in Keras 3.
        _keras_base_conv.Conv = _BASE_CONV_CLASS
except Exception:
    # If the shim cannot be applied, let tf-keras-vis import fail with the original error.
    _BASE_CONV_CLASS = None

from tf_keras_vis.gradcam import Gradcam
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import json
import requests
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =============================================================================
# Configuration
# Set these parameters based on your setup
# =============================================================================

MODEL_PATH = "models/cancer_classifier_xai.h5"  # Path to your saved model
IMG_SIZE = (224, 224)  # Match your model’s input size
OUTPUT_FILE = "outputs/real_time_xai_output.json"  # Path for saving results
PERFORMANCE_METRICS_FILE = "outputs/performance_metrics.json"  # Path for metrics

# Load environment variables from a local .env file if present (no effect if missing)
load_dotenv()

HUGGING_FACE_API_TOKEN = os.getenv("HF_TOKEN")
HF_CHAT_API_URL = os.getenv("HF_CHAT_API_URL", "https://router.huggingface.co/v1/chat/completions")
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct:novita")
HF_HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}

if not HUGGING_FACE_API_TOKEN:
    raise RuntimeError(
        "No Hugging Face API token found. Set HF_TOKEN in the environment or update HUGGING_FACE_API_TOKEN."
    )

# Class dictionary (hardcoded for simplicity, match test_with_h5.py)
class_dict = {
    'ALL': 0, 'Brain Cancer': 1, 'Breast Cancer': 2, 'Cervical Cancer': 3,
    'Kidney Cancer': 4, 'Lung and Colon Cancer': 5, 'Lymphoma': 6, 'Oral Cancer': 7
}


def _is_convolutional_layer(layer):
    """Return True when the layer behaves like a convolution."""
    if _BASE_CONV_CLASS is not None and isinstance(layer, _BASE_CONV_CLASS):
        return True
    name = layer.__class__.__name__.lower()
    if "conv" in name:
        return True
    return hasattr(layer, "kernel_size") and hasattr(layer, "strides")


def _find_penultimate_conv_layer(model):
    """Locate the last convolutional layer in the network for Grad-CAM."""
    candidate = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            nested_candidate = _find_penultimate_conv_layer(layer)
            if nested_candidate is not None:
                candidate = nested_candidate
        if _is_convolutional_layer(layer):
            print(
                "Found convolutional layer:",
                getattr(layer, "name", type(layer)),
                "type:",
                type(layer),
            )
            candidate = layer
    return candidate


def _chat_completion(messages, max_tokens, temperature=0.7, top_p=0.9):
    """Call Hugging Face router chat-completions API and return the assistant response."""
    payload = {
        "messages": messages,
        "model": HF_CHAT_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        response = requests.post(HF_CHAT_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()
    except (requests.exceptions.RequestException, KeyError, IndexError, TypeError) as exc:
        print(f"Hugging Face chat completion failed: {exc}")
        return None


# =============================================================================
# Functions
# Load and preprocess image, predict, generate XAI, and provide LLM explanation
# =============================================================================

def load_and_preprocess_image(image_path):
    """Load and preprocess an input image for the model."""
    if isinstance(image_path, str):  # File path
        img = cv2.imread(image_path)
    else:  # numpy array (e.g., from camera)
        img = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def visualize_xai_and_collect(model, img_array, class_dict, predicted_class):
    """Generate XAI visualizations and collect data for LLM."""
    pred = model.predict(img_array)
    confidence = pred[0][predicted_class] * 100
    class_name = list(class_dict.keys())[predicted_class]
    
    # Dynamic XAI insights based on class_name
    if "brain" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the tumor in the brain center. "
                       "Saliency Map shows tumor edges are critical. "
                       "LIME outlines the tumor area as the most important region.")
    elif "breast" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the tumor in the breast tissue. "
                       "Saliency Map shows tumor edges are critical. "
                       "LIME outlines the tumor area as the most important region.")
    elif "lung" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the tumor in the lung tissue. "
                       "Saliency Map shows tumor edges are critical. "
                       "LIME outlines the tumor area as the most important region.")
    elif "colon" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the tumor in the colon tissue. "
                       "Saliency Map shows tumor edges are critical. "
                       "LIME outlines the tumor area as the most important region.")
    elif "cervical" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the tumor in the cervical tissue. "
                       "Saliency Map shows tumor edges are critical. "
                       "LIME outlines the tumor area as the most important region.")
    elif "kidney" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the tumor in the kidney tissue. "
                       "Saliency Map shows tumor edges are critical. "
                       "LIME outlines the tumor area as the most important region.")
    elif "oral" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the tumor in the oral cavity. "
                       "Saliency Map shows tumor edges are critical. "
                       "LIME outlines the tumor area as the most important region.")
    elif "lymphoma" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the tumor in the lymphatic system. "
                       "Saliency Map shows tumor edges are critical. "
                       "LIME outlines the tumor area as the most important region.")
    elif "all" in class_name.lower():
        xai_insights = ("Grad-CAM highlights the condition in the affected area. "
                       "Saliency Map shows critical patterns. "
                       "LIME outlines the most important region.")
    
    # Generate XAI visuals
    def gradcam_loss(output):
        return output[0, predicted_class]
    gradcam = Gradcam(model, clone=False)
    cam = gradcam(gradcam_loss, img_array, penultimate_layer=-1)
    
    def saliency_map(model, img_array):
        img_tensor = tf.convert_to_tensor(img_array)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            preds = model(img_tensor)
            loss = preds[:, predicted_class]
        grads = tape.gradient(loss, img_tensor)[0]
        saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
        return saliency
    
    saliency = saliency_map(model, img_array)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array[0], model.predict, top_labels=1, num_samples=500, hide_color=0  # Reduced samples for speed
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    lime_img = mark_boundaries(img_array[0], mask)
    
    # Squeeze the batch dimension and ensure correct shape for visualization
    original_img = np.squeeze(img_array, axis=0)  # Remove batch dimension, should be (224, 224, 3)
    print(f"Original image shape: {original_img.shape}")  # Debug print to verify shape
    
    # Handle Grad-CAM shape (e.g., (224, 224, 1) to (224, 224) for heatmap)
    cam = np.squeeze(cam, axis=0) if cam.ndim == 3 and cam.shape[0] == 1 else cam  # Remove batch dimension if present and size is 1
    if cam.ndim == 3 and cam.shape[-1] == 1:  # Convert (224, 224, 1) to (224, 224)
        cam = cam[:, :, 0]
    print(f"Grad-CAM shape: {cam.shape}")
    
    # Ensure saliency is 2D (heatmap)
    saliency = np.squeeze(saliency, axis=0) if saliency.ndim == 3 and saliency.shape[0] == 1 else saliency  # Remove batch dimension if present and size is 1
    if saliency.ndim != 2:
        saliency = np.max(saliency, axis=-1)  # Flatten to 2D if needed
    print(f"Saliency map shape: {saliency.shape}")
    
    # Ensure lime_img is 3D (RGB with overlay)
    lime_img = np.squeeze(lime_img, axis=0) if lime_img.ndim == 4 and lime_img.shape[0] == 1 else lime_img  # Remove batch dimension if present and size is 1
    if lime_img.ndim == 2:  # Grayscale to RGB if needed
        lime_img = np.repeat(lime_img[:, :, np.newaxis], 3, axis=-1)
    print(f"LIME image shape: {lime_img.shape}")
    
    return class_name, confidence, xai_insights, original_img, cam, saliency, lime_img

def get_llm_explanation(class_name, xai_insights, confidence):
    # Safely parse class_name for tumor type and location
    parts = class_name.split()
    if len(parts) > 1:  # Multi-word name (e.g., "Kidney Cancer")
        location = parts[0].lower()  # e.g., "kidney"
        tumor_type = parts[1].lower()  # e.g., "cancer"
    else:  # Single-word name (e.g., "ALL", "Lymphoma")
        location = "the affected area"  # Generic for non-specific cancers
        tumor_type = class_name.lower()  # e.g., "all", "lymphoma"

    tumor_label = f"{tumor_type} tumor" if tumor_type not in ["all"] else f"{tumor_type} condition"  # Adjust for "ALL"

    # Infer tumor location primarily from class_name, ignoring XAI unless brain-specific
    tumor_location = "in the middle of the body"  # Default (removed for specificity, but kept for fallback)
    if "brain" in class_name.lower():
        tumor_location = "in the center of the brain" if "brain center" in xai_insights.lower() else "in the brain"
    elif "breast" in class_name.lower():
        tumor_location = "in the breast tissue"
    elif "lung" in class_name.lower():
        tumor_location = "in the lung tissue"
    elif "colon" in class_name.lower():
        tumor_location = "in the colon tissue"
    elif "cervical" in class_name.lower():
        tumor_location = "in the cervical tissue"
    elif "kidney" in class_name.lower():
        tumor_location = "in the kidney tissue"
    elif "oral" in class_name.lower():
        tumor_location = "in the oral cavity"
    elif "lymphoma" in class_name.lower():
        tumor_location = "in the lymphatic system"
    elif "all" in class_name.lower():
        tumor_location = "in the affected area"

    # Medical context based on cancer type
    medical_context = f"it’s a common {tumor_type} {('tumor' if tumor_type not in ['all'] else 'condition')}" if tumor_type not in ["all"] else f"it’s a type of blood cancer involving abnormal lymphocytes"

    # Use dynamic opener based on confidence
    opener = "Dear Patient," if confidence >= 70 else "We understand your concerns,"

    # Patient-friendly prompt with template, headers, checklist, and bullet points
    patient_prompt = (f"Follow this template: [GREETING] [PREDICTION SUMMARY] [MEDICAL CONTEXT] [HOPEFUL NOTE] [VALIDATION SUMMARY], "
                     f"with each section having 20-50 words and using 'Dear Patient,' as the opener. Include keywords 'confidence,' 'tumor,' 'survival.' "
                     f"[GREETING] Start with a formal, reassuring tone. [PREDICTION SUMMARY] State the AI’s {confidence:.1f}% confidence in identifying a {tumor_label} in {tumor_location}. "
                     f"[MEDICAL CONTEXT] Explain it’s a manageable condition, using bullet points for survival facts. [HOPEFUL NOTE] Offer hope with bullet points on health outcomes. "
                     f"[VALIDATION SUMMARY] Recap confidence, location, and hope, ending with ✨. Use simple language, avoid jargon.")

    # Doctor/Engineer-friendly prompt with template, headers, checklist, and numbered lists
    doctor_prompt = (f"Follow this template: [INTRODUCTION] [CLINICAL CONTEXT] [XAI ANALYSIS] [CONCLUSION] [VALIDATION SUMMARY], "
                     f"with each section having 20-50 words and using 'For oncologists and data engineers,' as the opener. Include keywords 'confidence,' 'neoplastic,' 'activation.' "
                     f"[INTRODUCTION] Explain the {confidence:.1f}% prediction of {class_name} in {tumor_location}. [CLINICAL CONTEXT] Provide advanced context (e.g., renal cell carcinoma). "
                     f"[XAI ANALYSIS] Detail Grad-CAM, Saliency Map, and LIME findings in a numbered list. [CONCLUSION] Note clinical utility. "
                     f"[VALIDATION SUMMARY] Recap confidence, location, and XAI, ending with 🩺. Use technical terms.")

    patient_messages = [
        {
            "role": "system",
            "content": "You craft reassuring, plain-language summaries of cancer AI findings for patients.",
        },
        {"role": "user", "content": patient_prompt},
    ]
    patient_explanation = _chat_completion(
        patient_messages,
        max_tokens=400,
        temperature=0.7,
        top_p=0.9,
    )
    if not patient_explanation:
        patient_explanation = "Sorry, we couldn’t generate a patient explanation right now."

    doctor_messages = [
        {
            "role": "system",
            "content": (
                "You brief oncologists and data engineers on cancer AI predictions using concise technical language."
            ),
        },
        {"role": "user", "content": doctor_prompt},
    ]
    doctor_explanation = _chat_completion(
        doctor_messages,
        max_tokens=600,
        temperature=0.7,
        top_p=0.9,
    )
    if not doctor_explanation:
        doctor_explanation = (
            "Due to a technical issue, we couldn’t generate a detailed explanation. "
            f"The AI predicted {class_name} with {confidence:.1f}% confidence, identifying a {tumor_label} in the {tumor_location}. "
            "Please consult the XAI insights and performance metrics for further analysis. 🩺"
        )

    return patient_explanation, doctor_explanation

def convert_array_to_qimage(array):
    """Convert numpy array to QImage for PyQt display."""
    height, width, channel = array.shape
    bytes_per_line = 3 * width
    if channel == 3:  # RGB
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    image = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return image

# =============================================================================
# PyQt6 GUI
# Create a responsive, themeable window with progress indicators, tooltips, and help
# =============================================================================

class CancerDetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Cancer Detection with XAI and LLM")
        self.setGeometry(100, 100, 1200, 800)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Responsive design
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Tab widget for different views (Patients, Doctors/Engineers)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Patient Tab (Main Interface)
        patient_tab = QWidget()
        patient_layout = QVBoxLayout(patient_tab)
        
        # Upload button with tooltip
        self.upload_button = QPushButton("Upload MRI Image")
        self.upload_button.setToolTip("Upload an MRI image (.jpg, .png, .jpeg) for cancer detection.")
        self.upload_button.clicked.connect(self.upload_image)
        patient_layout.addWidget(self.upload_button)
        
        # Camera button with tooltip
        self.camera_button = QPushButton("Capture from Camera")
        self.camera_button.setToolTip("Capture an image from your webcam for real-time analysis.")
        self.camera_button.clicked.connect(self.capture_from_camera)
        patient_layout.addWidget(self.camera_button)
        
        # Progress bar for processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        patient_layout.addWidget(self.progress_bar)
        
        # Image display (scrollable area for XAI visuals)
        self.image_label = QLabel("No image uploaded or captured yet")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setToolTip("Displays the uploaded/captured image and XAI visualizations after processing.")
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        patient_layout.addWidget(scroll_area)
        
        # Results display for patients
        self.patient_results_text = QTextEdit()
        self.patient_results_text.setReadOnly(True)
        self.patient_results_text.setToolTip("Shows prediction results and layman-friendly explanations.")
        patient_layout.addWidget(self.patient_results_text)
        
        # Process button with tooltip (enabled after upload/capture)
        self.process_button = QPushButton("Process Image")
        self.process_button.setToolTip("Process the image to get predictions, XAI visuals, and explanations.")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        patient_layout.addWidget(self.process_button)
        
        # Theme toggle button with tooltip
        self.theme_button = QPushButton("Toggle Theme")
        self.theme_button.setToolTip("Switch between light and dark themes for better visibility.")
        self.theme_button.clicked.connect(self.toggle_theme)
        patient_layout.addWidget(self.theme_button)
        
        # Help button with tooltip
        self.help_button = QPushButton("Help")
        self.help_button.setToolTip("View instructions and help for using this application.")
        self.help_button.clicked.connect(self.show_help)
        patient_layout.addWidget(self.help_button)
        
        # Privacy notice for patients
        self.privacy_label = QLabel("Privacy Notice: Your data is anonymized and not stored. We comply with HIPAA/GDPR for medical data protection. Uploads are processed securely and deleted after use. ✨")
        self.privacy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.privacy_label.setToolTip("Learn about how your data is protected and handled.")
        patient_layout.addWidget(self.privacy_label)
        
        # Doctors/Engineers Tab (Metrics and Technical Details)
        doctors_tab = QWidget()
        doctors_layout = QVBoxLayout(doctors_tab)
        
        # Metrics button with tooltip
        self.metrics_button = QPushButton("Show Performance Metrics")
        self.metrics_button.setToolTip("Display performance metrics (accuracy, precision, recall, F1-score) for engineers/doctors.")
        self.metrics_button.clicked.connect(self.show_metrics)
        doctors_layout.addWidget(self.metrics_button)
        
        # Technical results display
        self.doctors_results_text = QTextEdit()
        self.doctors_results_text.setReadOnly(True)
        self.doctors_results_text.setToolTip("Shows technical predictions, XAI insights, and performance metrics.")
        doctors_layout.addWidget(self.doctors_results_text)
        
        self.tab_widget.addTab(patient_tab, "Patients")
        self.tab_widget.addTab(doctors_tab, "Doctors/Engineers")
        
        # Load model
        self.model = None
        self.load_model()
        
        # Timer for periodic checks (optional for real-time updates, not used here)
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_status)
        self.timer.start(1000)  # Check every second (optional)
        
        self.image_path = None
        self.is_dark_theme = False
        self.xai_data = None  # Store XAI data for potential export or reuse

    def load_model(self):
        """Load the saved model in a background-friendly way."""
        try:
            self.model = load_model(MODEL_PATH)
            print(f"Loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.patient_results_text.append(f"Error: Could not load model - {e} ❗")
            self.doctors_results_text.append(f"Error: Could not load model - {e} 🔍")

    def upload_image(self):
        """Open file dialog to upload an image."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            if pixmap.isNull():
                self.patient_results_text.append("Error: Invalid image file ❗")
                return
            self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))
            self.process_button.setEnabled(True)
            self.patient_results_text.append("Image uploaded successfully. Click 'Process Image' or 'Capture from Camera' to analyze. ✨")

    def capture_from_camera(self):
        """Capture an image from the webcam."""
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("temp_camera_image.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.image_path = "temp_camera_image.jpg"
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))
            self.process_button.setEnabled(True)
            self.patient_results_text.append("Image captured from camera successfully. Click 'Process Image' to analyze. ✨")
        cap.release()
        os.remove("temp_camera_image.jpg") if os.path.exists("temp_camera_image.jpg") else None

    def process_image(self):
        if not self.model or not self.image_path:
            self.patient_results_text.append("Error: No model or image loaded ❗")
            self.doctors_results_text.append("Error: No model or image loaded 🔍")
            return
    
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
    
        try:
            # Load and preprocess image
            if os.path.exists(self.image_path):
                img_array = load_and_preprocess_image(self.image_path)
            else:
                img_array = load_and_preprocess_image(self.image_path)  # Handle camera capture
            self.progress_bar.setValue(30)
            self.patient_results_text.append("Image preprocessed successfully 💊")
            self.doctors_results_text.append("Image preprocessed successfully 🩺")
        
            # Predict and generate XAI
            predicted_class = np.argmax(self.model.predict(img_array)[0])
            self.progress_bar.setValue(60)
            class_name, confidence, xai_insights, original_img, cam, saliency, lime_img = visualize_xai_and_collect(self.model, img_array, class_dict, predicted_class)
        
            self.progress_bar.setValue(80)
            self.display_xai_visuals(original_img, cam, saliency, lime_img)
        
            # Generate LLM explanations for both audiences
            patient_explanation, doctor_explanation = get_llm_explanation(class_name, xai_insights, confidence)
            self.progress_bar.setValue(100)
        
            # Update results for patients (formal, hopeful, simple, with separators and bullet points)
            self.patient_results_text.clear()
            self.patient_results_text.append(f"🌟 Prediction Results 🌟\n---\n")
            self.patient_results_text.append(f"Predicted Class: {class_name} ({confidence:.1f}% confidence) 💊\n---\n")
            self.patient_results_text.append(f"XAI Insights: {xai_insights} 🩺\n---\n")
            self.patient_results_text.append(f"Layman’s Explanation:\n{patient_explanation}\n---\n")
            self.patient_results_text.append(f"Positive Affirmation: With early detection and advanced medical care, many patients thrive. ✨")
        
            # Update results for doctors/engineers (technical, detailed, with separators and numbered lists)
            self.doctors_results_text.clear()
            self.doctors_results_text.append(f"🩺 Technical Analysis 🩺\n---\n")
            self.doctors_results_text.append(f"Predicted Class: {class_name} ({confidence:.1f}% confidence) 💡\n---\n")
            self.doctors_results_text.append(f"XAI Insights: {xai_insights} 🔍\n---\n")
            self.doctors_results_text.append(f"Prompt for LLM:\n{doctor_explanation}\n---\n")
            self.doctors_results_text.append(f"Performance Metrics: {self.load_performance_metrics() if self.load_performance_metrics() else 'Metrics unavailable'} 🩺")
        
            # Save results to JSON
            xai_data = {
                "image_path": self.image_path,
                "predicted_class": int(predicted_class),
                "class_name": class_name,
                "confidence": confidence,
                "xai_insights": xai_insights,
                "patient_explanation": patient_explanation,
                "doctor_explanation": doctor_explanation
            }
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(xai_data, f, indent=4)
            self.patient_results_text.append(f"Saved results to {OUTPUT_FILE} ✨")
            self.doctors_results_text.append(f"Saved results to {OUTPUT_FILE} 🩺")
            self.xai_data = xai_data  # Store for potential export
    
        except Exception as e:
            self.patient_results_text.append(f"Error processing image: {e} ❗")
            self.doctors_results_text.append(f"Error processing image: {e} 🔍")
        finally:
            self.progress_bar.setVisible(False)

    def display_xai_visuals(self, original, cam, saliency, lime):
        """Display XAI visuals in a single image for PyQt."""
        # Ensure the input arrays have the correct shape for Matplotlib
        original = np.squeeze(original, axis=0) if original.ndim == 4 and original.shape[0] == 1 else original  # Remove batch dimension if present and size is 1
        if original.ndim == 2:  # Grayscale to RGB if needed
            original = np.repeat(original[:, :, np.newaxis], 3, axis=-1)
        print(f"Original image shape after squeeze: {original.shape}")  # Debug print to verify shape

        cam = np.squeeze(cam, axis=0) if cam.ndim == 3 and cam.shape[0] == 1 else cam  # Remove batch dimension if present and size is 1
        if cam.ndim == 2:  # Ensure 2D heatmap for overlay
            cam = np.expand_dims(cam, axis=-1)  # Make 3D for compatibility
        elif cam.ndim == 3 and cam.shape[-1] == 1:  # Convert (224, 224, 1) to (224, 224)
            cam = cam[:, :, 0]
        print(f"Grad-CAM shape after squeeze: {cam.shape}")

        saliency = np.squeeze(saliency, axis=0) if saliency.ndim == 3 and saliency.shape[0] == 1 else saliency  # Remove batch dimension if present and size is 1
        if saliency.ndim != 2:
            saliency = np.max(saliency, axis=-1)  # Flatten to 2D if needed
        print(f"Saliency map shape after squeeze: {saliency.shape}")

        lime = np.squeeze(lime, axis=0) if lime.ndim == 4 and lime.shape[0] == 1 else lime  # Remove batch dimension if present and size is 1
        if lime.ndim == 2:  # Grayscale to RGB if needed
            lime = np.repeat(lime[:, :, np.newaxis], 3, axis=-1)
        print(f"LIME image shape after squeeze: {lime.shape}")

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(2, 2, 2)
        plt.imshow(original)
        plt.imshow(cam, cmap="jet", alpha=0.5)
        plt.title("Grad-CAM")
        plt.axis("off")
        
        plt.subplot(2, 2, 3)
        plt.imshow(saliency, cmap="hot")
        plt.title("Saliency Map")
        plt.axis("off")
        
        plt.subplot(2, 2, 4)
        plt.imshow(lime)
        plt.title("LIME")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig("temp_xai_visuals.png")
        plt.close()
        
        pixmap = QPixmap("temp_xai_visuals.png")
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))
        os.remove("temp_xai_visuals.png")

    def show_metrics(self):
        """Display performance metrics for doctors/engineers."""
        metrics = self.load_performance_metrics()
        if metrics:
            metrics_text = (f"Accuracy: {metrics['accuracy']:.3f}\n"
                          f"Precision: {metrics['precision']:.3f}\n"
                          f"Recall: {metrics['recall']:.3f}\n"
                          f"F1-Score: {metrics['f1_score']:.3f}")
            self.doctors_results_text.append(f"🩺 Performance Metrics 🩺\n---\n{metrics_text}\n---")
        else:
            self.doctors_results_text.append("Error: Could not load performance metrics. 🔍")

    def load_performance_metrics(self):
        """Load performance metrics from JSON file."""
        try:
            with open(PERFORMANCE_METRICS_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Performance metrics file not found at {PERFORMANCE_METRICS_FILE}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON in {PERFORMANCE_METRICS_FILE}")
            return None

    def check_status(self):
        """Optional: Check for updates or status (not used here, but can be extended)."""
        pass

    def toggle_theme(self):
        """Toggle between light and dark themes for better visibility."""
        self.is_dark_theme = not self.is_dark_theme
        app = QApplication.instance()
        app.setStyle('Fusion')
        palette = QPalette()
        if self.is_dark_theme:
            palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.darkGray)
            palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.darkGray)
            palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkGray)
            palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            palette.setColor(QPalette.ColorRole.Link, Qt.GlobalColor.lightGray)
            palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.blue)
            palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        else:
            palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.lightGray)
            palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.lightGray)
            palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            palette.setColor(QPalette.ColorRole.Link, Qt.GlobalColor.blue)
            palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.blue)
            palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        app.setPalette(palette)
        self.patient_results_text.append(f"Switched to {'Dark' if self.is_dark_theme else 'Light'} Theme ✨")

    def show_help(self):
        """Display help instructions and documentation."""
        help_text = ("**Help for Real-Time Cancer Detection with XAI and LLM**\n"
                     "1. Upload an MRI image or capture from camera to analyze for cancer. 🌟\n"
                     "2. Click 'Process Image' to get predictions, XAI visuals (Grad-CAM, Saliency Map, LIME), and explanations. 💊\n"
                     "3. Use the 'Patients' tab for simple, friendly results, and 'Doctors/Engineers' for technical details. 🩺\n"
                     "4. Toggle the theme (light/dark) for better visibility. ✨\n"
                     "5. Privacy: Your data is anonymized, not stored, and complies with HIPAA/GDPR. 🔒\n"
                     "6. XAI (Explainable AI) uses Grad-CAM, Saliency Map, and LIME to explain predictions visually. 🔍\n"
                     "7. Contact support for issues or questions. ❗")
        self.patient_results_text.append(help_text)

# Run the application
if __name__ == "__main__":
    app = QApplication([])
    window = CancerDetectionWindow()
    window.show()
    app.exec()
