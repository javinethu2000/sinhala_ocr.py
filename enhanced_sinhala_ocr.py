"""
Enhanced Sinhala OCR System - Google Lens-like functionality
Supports image upload, text extraction, and copying functionality
"""

import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
import tensorflow as tf
from keras import layers, models
from keras.optimizers import SGD
from keras.utils import to_categorical
from flask import Flask, request, jsonify, send_file, render_template
import io
import base64
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
IMG_SIZE = 28
CHANNELS = 1
MODEL_NAME = "sinhala_complete_cnn.h5"  # Use the most comprehensive model
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "gif"}

# Load existing label mapping
def load_label_mapping():
    """Load the existing comprehensive label mapping"""
    try:
        # Try to load from the existing mapping
        from sinhala_ocr import label_to_unicode, unicode_to_label
        return label_to_unicode, unicode_to_label
    except ImportError:
        # Fallback to basic mapping
        label_to_unicode = {
            "a": "අ", "aa": "ආ", "ae": "ඇ", "aae": "ඈ", "i": "ඉ", "ii": "ඊ",
            "u": "උ", "uu": "ඌ", "ri": "ඍ", "e": "එ", "ee": "ඒ", "ai": "ඓ",
            "o": "ඔ", "oo": "ඕ", "au": "ඖ", "ka": "ක", "kaa": "කා", "ki": "කි"
            # Add more as needed
        }
        unicode_to_label = {v: k for k, v in label_to_unicode.items()}
        return label_to_unicode, unicode_to_label

label_to_unicode, unicode_to_label = load_label_mapping()

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def is_allowed_filename(filename: str) -> bool:
    """Check if filename has allowed extension"""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXT

def preprocess_image_for_model(img: np.ndarray, img_size=IMG_SIZE) -> np.ndarray:
    """
    Preprocess image for model inference
    Input: grayscale (H,W) or color (H,W,3)
    Output: normalized (img_size,img_size,1) float32
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize while maintaining aspect ratio
    h, w = img.shape
    if h != img_size or w != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # Normalize
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def load_model_and_labels(model_path: str = MODEL_NAME):
    """Load trained model and corresponding labels"""
    model_paths = [
        'sinhala_complete_cnn.h5', 
        'sinhala_essential_cnn.h5', 
        'sinhala_cnn.h5', 
        model_path
    ]
    
    for path in model_paths:
        try:
            if os.path.exists(path):
                logger.info(f"Loading model: {path}")
                model = models.load_model(path)
                labels_path = path + ".labels.json"
                
                if os.path.exists(labels_path):
                    with open(labels_path, "r", encoding="utf-8") as f:
                        labels = json.load(f)
                    logger.info(f"✅ Loaded {len(labels)} characters from {path}")
                    return model, labels
                else:
                    logger.warning(f"⚠️ Labels file not found: {labels_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load {path}: {e}")
            continue
    
    raise FileNotFoundError("No valid model found")

def segment_characters_from_image(image: np.ndarray, 
                                min_area=80, 
                                debug=False) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Advanced character segmentation with improved accuracy
    Returns list of (cropped_image, bbox) tuples
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding for better binarization
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # Morphological operations to connect character parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small noise
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_opening, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Filter based on size and aspect ratio
        if area < min_area or w < 5 or h < 5:
            continue
        
        # Filter out very wide or very tall rectangles (likely noise)
        aspect_ratio = w / h
        if aspect_ratio > 3 or aspect_ratio < 0.1:
            continue
            
        boxes.append((x, y, w, h))
    
    # Sort boxes by reading order (top-to-bottom, left-to-right)
    boxes = sorted(boxes, key=lambda b: (b[1] // 20, b[0]))
    
    results = []
    for (x, y, w, h) in boxes:
        # Add padding around characters
        pad = 5
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad)
        y1 = min(gray.shape[0], y + h + pad)
        
        crop = gray[y0:y1, x0:x1]
        results.append((crop, (x0, y0, x1 - x0, y1 - y0)))
    
    if debug:
        logger.info(f"Found {len(results)} character segments")
    
    return results

def predict_on_image(image: np.ndarray, 
                    model, 
                    label_list: List[str], 
                    confidence_threshold: float = 0.1) -> Dict:
    """
    Enhanced prediction with confidence filtering and detailed results
    """
    try:
        # Determine model input size
        try:
            if hasattr(model, 'input_shape'):
                expected_img_size = model.input_shape[1]
            else:
                expected_img_size = 64  # Default fallback
        except:
            expected_img_size = 64
            
        logger.info(f"Using input size: {expected_img_size}x{expected_img_size}")
        
        # Segment characters
        segs = segment_characters_from_image(image, debug=True)
        
        if not segs:
            return {
                "text": "",
                "characters": [],
                "confidence": 0.0,
                "character_count": 0
            }
        
        characters = []
        full_text = ""
        total_confidence = 0.0
        
        for i, (crop, bbox) in enumerate(segs):
            try:
                # Preprocess
                xcrop = preprocess_image_for_model(crop, img_size=expected_img_size)
                xcrop = np.expand_dims(xcrop, axis=0)
                
                # Predict
                preds = model.predict(xcrop, verbose=0)
                idx = np.argmax(preds[0])
                prob = float(preds[0][idx])
                
                if prob < confidence_threshold:
                    continue
                
                label = label_list[idx]
                unicode_char = label  # Labels are already Unicode characters
                
                character_info = {
                    "char": unicode_char,
                    "confidence": prob,
                    "bbox": bbox,
                    "label": label,
                    "position": i
                }
                
                characters.append(character_info)
                full_text += unicode_char
                total_confidence += prob
                
            except Exception as e:
                logger.error(f"Error processing character {i}: {e}")
                continue
        
        # Sort by position for reading order
        characters.sort(key=lambda x: (x["bbox"][1] // 20, x["bbox"][0]))
        full_text = "".join([char["char"] for char in characters])
        
        avg_confidence = total_confidence / len(characters) if characters else 0.0
        
        return {
            "text": full_text,
            "characters": characters,
            "confidence": avg_confidence,
            "character_count": len(characters)
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {
            "text": "",
            "characters": [],
            "confidence": 0.0,
            "character_count": 0,
            "error": str(e)
        }

def create_annotated_image(image: np.ndarray, predictions: List[Dict]) -> str:
    """Create annotated image with bounding boxes and characters"""
    annotated = image.copy()
    
    for char_info in predictions:
        x, y, w, h = char_info["bbox"]
        conf = char_info["confidence"]
        char = char_info["char"]
        
        # Color based on confidence
        if conf > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif conf > 0.5:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Draw character and confidence
        text = f"{char} ({conf:.2f})"
        cv2.putText(annotated, text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Convert to base64
    _, buf = cv2.imencode('.png', annotated)
    return base64.b64encode(buf).decode('utf-8')

def create_enhanced_app(model_path: str = MODEL_NAME):
    """Create enhanced Flask app with improved UI and functionality"""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    model, labels = None, None
    
    @app.before_request
    def load_model_once():
        nonlocal model, labels
        if model is None:
            try:
                model, labels = load_model_and_labels(model_path)
                logger.info("Model & labels loaded for Flask app.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                model, labels = None, None
    
    @app.route("/", methods=["GET"])
    def home():
        return render_template("enhanced_index.html")
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model_loaded": model is not None,
            "labels_count": len(labels) if labels else 0,
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route("/api/analyze", methods=["POST"])
    def analyze_image():
        """Enhanced image analysis endpoint"""
        nonlocal model, labels
        
        if model is None or labels is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        try:
            # Handle file upload
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
                
            file = request.files['file']
            if file.filename == '' or not file.filename:
                return jsonify({"error": "No file selected"}), 400
            
            if not is_allowed_filename(file.filename):
                return jsonify({"error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXT)}"}), 400
            
            # Read and decode image
            in_memory = file.read()
            nparr = np.frombuffer(in_memory, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({"error": "Could not decode image"}), 400
            
            # Get confidence threshold from request
            confidence_threshold = float(request.form.get('confidence', 0.1))
            
            # Predict
            results = predict_on_image(img, model, labels, confidence_threshold)
            
            # Create annotated image
            annotated_image = create_annotated_image(img, results.get("characters", []))
            
            # Response
            response = {
                "success": True,
                "results": results,
                "annotated_image": annotated_image,
                "processing_info": {
                    "image_size": f"{img.shape[1]}x{img.shape[0]}",
                    "confidence_threshold": confidence_threshold,
                    "model_input_size": f"{model.input_shape[1]}x{model.input_shape[2]}",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in analyze_image: {e}")
            return jsonify({
                "error": "Internal processing error",
                "details": str(e)
            }), 500
    
    @app.route("/api/export", methods=["POST"])
    def export_text():
        """Export extracted text in different formats"""
        try:
            data = request.get_json()
            text = data.get('text', '')
            format_type = data.get('format', 'txt')
            
            if format_type == 'json':
                export_data = {
                    "extracted_text": text,
                    "character_count": len(text),
                    "export_time": datetime.now().isoformat()
                }
                content = json.dumps(export_data, ensure_ascii=False, indent=2)
                mimetype = 'application/json'
                filename = f"sinhala_ocr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                content = text
                mimetype = 'text/plain'
                filename = f"sinhala_ocr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            return app.response_class(
                content,
                mimetype=mimetype,
                headers={"Content-disposition": f"attachment; filename={filename}"}
            )
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Maximum size: 16MB"}), 413
    
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Sinhala OCR System")
    parser.add_argument("--serve", action="store_true", help="Start Flask server")
    parser.add_argument("--model", default=MODEL_NAME, help="Model path")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.serve:
        app = create_enhanced_app(args.model)
        print(f"🚀 Starting Enhanced Sinhala OCR Server")
        print(f"📊 Model: {args.model}")
        print(f"🌐 URL: http://localhost:{args.port}")
        print(f"📝 Features: Google Lens-like image analysis with text extraction and copy functionality")
        
        app.run(
            host="0.0.0.0", 
            port=args.port, 
            debug=args.debug,
            threaded=True
        )
    else:
        parser.print_help()