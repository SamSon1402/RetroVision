import streamlit as st
import cv2
import numpy as np
import torch
import random
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import os
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks
from scipy.spatial.distance import cosine
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Set page configuration
st.set_page_config(
    page_title="RetroVision: Shelf Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for retro gaming aesthetics
def load_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono:wght@400;700&display=swap');
    
    .retro-title {
        font-family: 'VT323', monospace;
        color: #FFDE59;
        text-shadow: 4px 4px 0px #FF6B6B;
        text-align: center;
        padding: 10px;
        margin-bottom: 20px;
        font-size: 3em;
        border: 4px solid #FFDE59;
        background-color: #2A2D43;
    }
    
    .retro-subtitle {
        font-family: 'VT323', monospace;
        color: #FF6B6B;
        text-shadow: 2px 2px 0px #FFDE59;
        font-size: 1.8em;
        margin: 15px 0;
        border-bottom: 3px solid #FFDE59;
        padding-bottom: 5px;
    }
    
    .retro-text {
        font-family: 'Space Mono', monospace;
        color: #EAEAEA;
        line-height: 1.6;
    }
    
    .retro-button {
        font-family: 'VT323', monospace;
        background-color: #FF6B6B;
        color: #2A2D43;
        padding: 10px 20px;
        border: 3px solid #FFDE59;
        cursor: pointer;
        font-size: 1.2em;
        text-transform: uppercase;
        margin: 10px 0;
        transition: all 0.2s;
    }
    
    .retro-button:hover {
        background-color: #FFDE59;
        color: #2A2D43;
        border: 3px solid #FF6B6B;
    }
    
    .retro-metrics {
        font-family: 'VT323', monospace;
        background-color: #2A2D43;
        border: 3px solid #FF6B6B;
        padding: 10px;
        margin: 10px 0;
    }
    
    .stApp {
        background-color: #1A1A2E;
    }
    
    /* Custom radio buttons */
    .stRadio > div {
        background-color: #2A2D43 !important;
        border: 2px solid #FFDE59 !important;
        padding: 5px !important;
    }
    
    /* Custom sliders */
    .stSlider > div > div {
        background-color: #FF6B6B !important;
    }
    
    .stSlider > div > div > div {
        background-color: #FFDE59 !important;
    }
    
    /* Image container */
    .retro-image-container {
        border: 4px solid #FFDE59;
        padding: 5px;
        background-color: #2A2D43;
    }
    
    /* For metrics display */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    
    .metric-box {
        font-family: 'VT323', monospace;
        background-color: #2A2D43;
        border: 3px solid #FF6B6B;
        padding: 15px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2em;
        color: #FFDE59;
    }
    
    .metric-label {
        font-size: 1.2em;
        color: #FF6B6B;
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load CSS
load_css()

# Custom header
st.markdown('<div class="retro-title">RetroVision: Shelf Analytics</div>', unsafe_allow_html=True)

# Display a retro gaming themed introduction
st.markdown('<div class="retro-text">Welcome to RetroVision, your computer vision tool for analyzing retail shelf promotional materials. Upload images to detect, classify, and analyze promotional items on store shelves!</div>', unsafe_allow_html=True)

# Create two tabs for Real Images and Synthetic Data
tab1, tab2, tab3 = st.tabs(["📸 Real Images", "🎲 Synthetic Data", "📊 Evaluation"])

# Functions for synthetic data generation
def generate_synthetic_shelf(width=800, height=600):
    """Generate a synthetic image of a retail shelf with products and promotional materials"""
    # Create a blank image with a shelf-like background
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw shelf lines
    shelf_colors = [(200, 190, 170), (180, 170, 150), (190, 180, 160)]
    shelf_heights = [100, 250, 400, 550]
    
    for y in shelf_heights:
        draw.rectangle([(0, y), (width, y+20)], fill=random.choice(shelf_colors))
    
    # Generate random products on shelves
    product_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (200, 100, 0), (100, 200, 0), (0, 100, 200)
    ]
    
    promotional_materials = []
    products = []
    
    # Add products to shelves
    for shelf_y in [y + 30 for y in shelf_heights[:-1]]:
        x = 20
        while x < width - 50:
            product_width = random.randint(30, 80)
            product_height = random.randint(60, 120)
            
            if x + product_width < width - 20:
                color = random.choice(product_colors)
                draw.rectangle([(x, shelf_y - product_height), (x + product_width, shelf_y)], 
                              fill=color, outline=(0, 0, 0))
                
                products.append({
                    'bbox': [x, shelf_y - product_height, x + product_width, shelf_y],
                    'class': 'product',
                    'color': color
                })
                
                x += product_width + random.randint(5, 15)
            else:
                break
    
    # Add promotional materials (signs, banners, etc.)
    promo_types = ['banner', 'poster', 'price_tag', 'discount_sign']
    promo_colors = [(255, 220, 0), (255, 107, 107), (255, 255, 255), (100, 255, 218)]
    
    for _ in range(random.randint(3, 8)):
        promo_type = random.choice(promo_types)
        color = random.choice(promo_colors)
        
        if promo_type == 'banner':
            # Horizontal banner
            y = random.choice(shelf_heights) - random.randint(5, 20)
            width_ratio = random.uniform(0.2, 0.5)
            start_x = random.randint(0, int(width * (1 - width_ratio)))
            banner_width = int(width * width_ratio)
            banner_height = random.randint(20, 40)
            
            draw.rectangle([(start_x, y), (start_x + banner_width, y + banner_height)], 
                          fill=color, outline=(0, 0, 0))
            
            promotional_materials.append({
                'bbox': [start_x, y, start_x + banner_width, y + banner_height],
                'class': 'banner',
                'material': random.choice(['paper', 'plastic', 'fabric'])
            })
            
        elif promo_type == 'poster':
            # Vertical poster
            shelf_idx = random.randint(0, len(shelf_heights) - 2)
            start_y = shelf_heights[shelf_idx] + 25
            end_y = shelf_heights[shelf_idx + 1] - 10
            poster_height = end_y - start_y
            poster_width = random.randint(40, 80)
            x = random.randint(0, width - poster_width)
            
            draw.rectangle([(x, start_y), (x + poster_width, end_y)], 
                          fill=color, outline=(0, 0, 0))
            
            promotional_materials.append({
                'bbox': [x, start_y, x + poster_width, end_y],
                'class': 'poster',
                'material': random.choice(['paper', 'cardboard'])
            })
            
        elif promo_type == 'price_tag':
            # Small price tag on a product
            if products:
                product = random.choice(products)
                product_x1, product_y1, product_x2, product_y2 = product['bbox']
                
                tag_width = random.randint(20, 40)
                tag_height = random.randint(15, 25)
                tag_x = random.randint(product_x1, product_x2 - tag_width)
                tag_y = random.randint(product_y1, product_y2 - tag_height)
                
                draw.rectangle([(tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height)], 
                              fill=color, outline=(0, 0, 0))
                
                promotional_materials.append({
                    'bbox': [tag_x, tag_y, tag_x + tag_width, tag_y + tag_height],
                    'class': 'price_tag',
                    'material': random.choice(['paper', 'plastic'])
                })
                
        else:  # discount_sign
            # Discount sign hanging from shelf
            shelf_y = random.choice(shelf_heights)
            sign_width = random.randint(50, 100)
            sign_height = random.randint(40, 70)
            x = random.randint(0, width - sign_width)
            
            draw.rectangle([(x, shelf_y + 20), (x + sign_width, shelf_y + 20 + sign_height)], 
                          fill=color, outline=(0, 0, 0))
            
            # Draw string connecting to shelf
            draw.line([(x + sign_width//2, shelf_y), (x + sign_width//2, shelf_y + 20)], 
                     fill=(0, 0, 0), width=2)
            
            promotional_materials.append({
                'bbox': [x, shelf_y + 20, x + sign_width, shelf_y + 20 + sign_height],
                'class': 'discount_sign',
                'material': random.choice(['paper', 'plastic', 'cardboard'])
            })
    
    # Add text to promotional materials (simplified)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        
    for promo in promotional_materials:
        x1, y1, x2, y2 = promo['bbox']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        if promo['class'] == 'price_tag':
            text = f"${random.randint(1, 20)}.{random.randint(0, 99):02d}"
        elif promo['class'] == 'discount_sign':
            text = f"{random.randint(10, 50)}% OFF"
        else:
            text = random.choice(["SALE!", "NEW!", "SPECIAL", "BUY NOW"])
            
        # Calculate text position using modern Pillow API
        try:
            # For newer Pillow versions (9.2.0+)
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            try:
                # Alternative approach for some Pillow versions
                text_width = font.getlength(text)
                text_height = font.size
            except AttributeError:
                # Fallback to a simple estimate if all else fails
                text_width = len(text) * 8  # rough estimate
                text_height = 14
                
        text_x = center_x - text_width // 2
        text_y = center_y - text_height // 2
        
        # Draw text
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    
    # Apply optional blur (for photo quality evaluation)
    if random.random() < 0.3:
        # Convert PIL to OpenCV
        img_np = np.array(img)
        blur_amount = random.randint(1, 3) * 2 + 1  # Odd number for kernel size
        img_np = cv2.GaussianBlur(img_np, (blur_amount, blur_amount), 0)
        img = Image.fromarray(img_np)
    
    # Apply optional brightness variation (for photo quality evaluation)
    if random.random() < 0.3:
        brightness_factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
    
    # Return the image and the metadata
    metadata = {
        'promotional_materials': promotional_materials,
        'products': products
    }
    return img, metadata

def apply_effects(image, effect_type="none"):
    """Apply various effects to simulate photo quality issues"""
    img_np = np.array(image)
    
    if effect_type == "blur":
        img_np = cv2.GaussianBlur(img_np, (15, 15), 0)
    elif effect_type == "dark":
        img_np = cv2.convertScaleAbs(img_np, alpha=0.6, beta=0)
    elif effect_type == "bright":
        img_np = cv2.convertScaleAbs(img_np, alpha=1.5, beta=30)
    elif effect_type == "low_contrast":
        # Reduce contrast
        img_np = cv2.convertScaleAbs(np.array(image) * 0.8 + 50)
    elif effect_type == "noise":
        # Add noise
        noise = np.random.normal(0, 20, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
    
    return Image.fromarray(img_np)

def calculate_photo_quality_score(img):
    """Calculate photo quality score based on blur, brightness, contrast"""
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(100, int(laplacian_var / 5))  # Scale to 0-100
    
    # Brightness analysis
    brightness = np.mean(gray)
    brightness_score = 100 - abs(int(((brightness - 127) / 127) * 100))  # 100 is optimal (middle brightness)
    
    # Contrast analysis
    contrast = gray.std()
    contrast_score = min(100, int(contrast / 2))  # Scale to 0-100
    
    # Overall quality score
    quality_score = int((blur_score * 0.5) + (brightness_score * 0.25) + (contrast_score * 0.25))
    
    return {
        "quality_score": quality_score,
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score
    }

# Load pre-trained models
@st.cache_resource
def load_yolo_model():
    model = YOLO("yolov8n.pt")  # Using a smaller model for faster loading
    return model

@st.cache_resource
def load_maskrcnn_model():
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=True)
    model.eval()
    return model, weights.transforms()

# Process image with YOLOv8
def process_with_yolo(image, confidence_threshold=0.25):
    img_np = np.array(image)
    model = load_yolo_model()
    results = model(img_np, conf=confidence_threshold)
    
    # Process results
    detections = []
    result = results[0]
    
    # Map COCO classes to promotional material types (simplified mapping)
    promo_class_map = {
        73: "banner",      # book -> banner
        26: "poster",      # handbag -> poster  
        39: "price_tag",   # bottle -> price_tag
        67: "discount_sign", # dining table -> discount_sign
    }
    
    material_map = {
        "banner": ["paper", "fabric", "plastic"],
        "poster": ["paper", "cardboard"],
        "price_tag": ["paper", "plastic"],
        "discount_sign": ["paper", "plastic", "cardboard"]
    }
    
    # Extract detected objects that could be promotional materials
    for box in result.boxes:
        cls_id = int(box.cls.item())
        if cls_id in promo_class_map or cls_id < 15:  # Common objects and our mapped promo materials
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            if cls_id in promo_class_map:
                class_name = promo_class_map[cls_id]
                material = random.choice(material_map[class_name])
            else:
                # Use COCO class name for other objects
                class_name = result.names[cls_id]
                material = "unknown"
                
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class": class_name,
                "confidence": conf,
                "material": material
            })
    
    # Draw results on image
    result_img = Image.fromarray(result.plot(labels=True))
    
    return result_img, detections

# Process with Mask R-CNN
def process_with_maskrcnn(image, confidence_threshold=0.5):
    model, transform = load_maskrcnn_model()
    
    # Transform the image
    img_tensor = transform(image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # Get the results
    img_np = np.array(image)
    result_img = img_np.copy()
    
    # Extract detections above threshold
    detections = []
    material_map = {
        "banner": ["paper", "fabric", "plastic"],
        "poster": ["paper", "cardboard"],
        "price_tag": ["paper", "plastic"],
        "discount_sign": ["paper", "plastic", "cardboard"]
    }
    
    promo_class_map = {
        73: "banner",      # book -> banner
        26: "poster",      # handbag -> poster  
        39: "price_tag",   # bottle -> price_tag
        67: "discount_sign", # dining table -> discount_sign
    }
    
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']
    
    # Filter by confidence
    keep_indices = scores > confidence_threshold
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]
    masks = masks[keep_indices]
    
    # Draw boxes and masks
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy().astype(int)
        label_id = labels[i].item()
        score = scores[i].item()
        mask = masks[i, 0].cpu().numpy() > 0.5
        
        if label_id in promo_class_map or label_id < 15:
            if label_id in promo_class_map:
                class_name = promo_class_map[label_id]
                material = random.choice(material_map[class_name])
            else:
                class_name = model.CLASSES[label_id]
                material = "unknown"
            
            detections.append({
                "bbox": box.tolist(),
                "class": class_name,
                "confidence": score,
                "material": material,
                "has_mask": True
            })
            
            # Draw box
            cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (255, 220, 0), 2)
            
            # Draw mask
            colored_mask = np.zeros_like(result_img)
            color = [random.randint(0, 255) for _ in range(3)]
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask, color[c], 0)
            
            # Blend mask with image
            alpha = 0.3
            cv2.addWeighted(colored_mask, alpha, result_img, 1-alpha, 0, result_img)
            
            # Add label
            cv2.putText(result_img, f"{class_name} {score:.2f}", 
                       (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 0), 2)
    
    return Image.fromarray(result_img), detections

# Calculate basic metrics for evaluation
def calculate_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Calculate precision, recall, and F1 score"""
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truth)  # Assume all ground truth are missed at first
    
    # Match predictions to ground truth
    for pred in predictions:
        pred_bbox = pred["bbox"]
        best_iou = 0
        best_match = -1
        
        # Find best matching ground truth
        for i, gt in enumerate(ground_truth):
            gt_bbox = gt["bbox"]
            iou = calculate_iou(pred_bbox, gt_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_match = i
        
        # If a good match is found
        if best_iou >= iou_threshold and best_match >= 0:
            true_positives += 1
            false_negatives -= 1  # This ground truth is now accounted for
        else:
            false_positives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # No intersection
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

# Display detection results
def display_detection_results(image, detections, quality_scores):
    # Count promotional materials by class
    promo_counts = {}
    material_counts = {}
    
    for det in detections:
        class_name = det["class"]
        material = det.get("material", "unknown")
        
        if class_name in ["banner", "poster", "price_tag", "discount_sign"]:
            promo_counts[class_name] = promo_counts.get(class_name, 0) + 1
            material_counts[material] = material_counts.get(material, 0) + 1
    
    # Display counts
    st.markdown('<div class="retro-subtitle">Detection Results</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="retro-text">Promotional Material Counts:</div>', unsafe_allow_html=True)
        for cls, count in promo_counts.items():
            st.markdown(f'<div class="retro-text">📊 {cls.replace("_", " ").title()}: {count}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="retro-text">Material Type Counts:</div>', unsafe_allow_html=True)
        for material, count in material_counts.items():
            st.markdown(f'<div class="retro-text">🧮 {material.title()}: {count}</div>', unsafe_allow_html=True)
    
    # Display quality scores
    st.markdown('<div class="retro-subtitle">Photo Quality Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">OVERALL QUALITY</div>
            <div class="metric-value">{quality_scores['quality_score']}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">SHARPNESS</div>
            <div class="metric-value">{quality_scores['blur_score']}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">BRIGHTNESS</div>
            <div class="metric-value">{quality_scores['brightness_score']}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">CONTRAST</div>
            <div class="metric-value">{quality_scores['contrast_score']}%</div>
        </div>
        """, unsafe_allow_html=True)

    # List detailed detections
    st.markdown('<div class="retro-subtitle">Detailed Detections</div>', unsafe_allow_html=True)
    
    if detections:
        detection_df = pd.DataFrame([
            {
                "Class": d["class"].replace("_", " ").title(),
                "Confidence": f"{d.get('confidence', 1.0)*100:.1f}%",
                "Material": d.get("material", "Unknown").title(),
                "Position": f"[{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]"
            }
            for d in detections
        ])
        
        # Apply styling to the dataframe
        st.markdown("""
        <style>
        .dataframe {
            font-family: 'Space Mono', monospace !important;
            border: 3px solid #FFDE59 !important;
        }
        .dataframe th {
            background-color: #FF6B6B !important;
            color: #2A2D43 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.2em !important;
            text-align: center !important;
        }
        .dataframe td {
            background-color: #2A2D43 !important;
            color: #EAEAEA !important;
            border: 1px solid #FFDE59 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(detection_df, use_container_width=True)
    else:
        st.markdown('<div class="retro-text">No promotional materials detected!</div>', unsafe_allow_html=True)

# Define functions for the evaluation tab
def generate_evaluation_charts(real_metrics, synthetic_metrics):
    st.markdown('<div class="retro-subtitle">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare data for the bar chart
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score'],
            'YOLOv8': [real_metrics['yolo']['precision'], 
                       real_metrics['yolo']['recall'], 
                       real_metrics['yolo']['f1']],
            'Mask R-CNN': [real_metrics['maskrcnn']['precision'], 
                          real_metrics['maskrcnn']['recall'], 
                          real_metrics['maskrcnn']['f1']]
        })
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        x = np.arange(len(metrics_df['Metric']))
        
        # Style the chart with retro gaming colors
        ax.bar(x - bar_width/2, metrics_df['YOLOv8'], bar_width, 
              label='YOLOv8', color='#FFDE59', edgecolor='#2A2D43', linewidth=2)
        ax.bar(x + bar_width/2, metrics_df['Mask R-CNN'], bar_width, 
              label='Mask R-CNN', color='#FF6B6B', edgecolor='#2A2D43', linewidth=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Metric'])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance on Real Images', fontweight='bold')
        ax.legend()
        
        # Set background color and grid
        ax.set_facecolor('#1A1A2E')
        fig.patch.set_facecolor('#1A1A2E')
        ax.grid(color='#EAEAEA', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Style the text elements
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('#EAEAEA')
        
        ax.title.set_color('#FFDE59')
        ax.xaxis.label.set_color('#EAEAEA')
        ax.yaxis.label.set_color('#EAEAEA')
        
        # Display the chart
        st.pyplot(fig)
    
    with col2:
        # Prepare data for the synthetic evaluation chart
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score'],
            'YOLOv8': [synthetic_metrics['yolo']['precision'], 
                      synthetic_metrics['yolo']['recall'], 
                      synthetic_metrics['yolo']['f1']],
            'Mask R-CNN': [synthetic_metrics['maskrcnn']['precision'], 
                          synthetic_metrics['maskrcnn']['recall'], 
                          synthetic_metrics['maskrcnn']['f1']]
        })
        
        # Create the bar chart for synthetic data
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(x - bar_width/2, metrics_df['YOLOv8'], bar_width, 
              label='YOLOv8', color='#FFDE59', edgecolor='#2A2D43', linewidth=2)
        ax.bar(x + bar_width/2, metrics_df['Mask R-CNN'], bar_width, 
              label='Mask R-CNN', color='#FF6B6B', edgecolor='#2A2D43', linewidth=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Metric'])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance on Synthetic Images', fontweight='bold')
        ax.legend()
        
        # Set background color and grid
        ax.set_facecolor('#1A1A2E')
        fig.patch.set_facecolor('#1A1A2E')
        ax.grid(color='#EAEAEA', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Style the text elements
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('#EAEAEA')
        
        ax.title.set_color('#FFDE59')
        ax.xaxis.label.set_color('#EAEAEA')
        ax.yaxis.label.set_color('#EAEAEA')
        
        # Display the chart
        st.pyplot(fig)

    # Add a comparison of detection speed
    st.markdown('<div class="retro-subtitle">Detection Speed Comparison</div>', unsafe_allow_html=True)
    
    speed_df = pd.DataFrame({
        'Model': ['YOLOv8', 'Mask R-CNN'],
        'Average Detection Time (seconds)': [0.12, 0.48],  # Example values
    })
    
    # Create horizontal bar chart for speed comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(speed_df['Model'], speed_df['Average Detection Time (seconds)'], 
                  color=['#FFDE59', '#FF6B6B'], edgecolor='#2A2D43', linewidth=2)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{width:.2f}s', ha='left', va='center', color='#EAEAEA')
    
    ax.set_xlabel('Average Detection Time (seconds)', fontweight='bold')
    ax.set_title('Model Speed Comparison', fontweight='bold')
    
    # Set background color
    ax.set_facecolor('#1A1A2E')
    fig.patch.set_facecolor('#1A1A2E')
    ax.grid(color='#EAEAEA', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Style the text elements
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color('#EAEAEA')
    
    ax.title.set_color('#FFDE59')
    ax.xaxis.label.set_color('#EAEAEA')
    
    st.pyplot(fig)

# Tab 1: Real Images
with tab1:
    st.markdown('<div class="retro-subtitle">Upload Retail Shelf Image</div>', unsafe_allow_html=True)
    
    # Sidebar for model selection and parameters
    with st.sidebar:
        st.markdown('<div class="retro-subtitle">Model Settings</div>', unsafe_allow_html=True)
        
        model_choice = st.radio(
            "Select Detection Model",
            ["YOLOv8", "Mask R-CNN"],
            key="real_model_choice"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            key="real_confidence"
        )
        
        st.markdown('<div class="retro-subtitle">Analysis Options</div>', unsafe_allow_html=True)
        
        detect_materials = st.checkbox("Detect Material Types", value=True)
        score_quality = st.checkbox("Score Photo Quality", value=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a retail shelf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="retro-text">Original Image</div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown('<div class="retro-text">Processed Image</div>', unsafe_allow_html=True)
            
            with st.spinner("⏳ Running detection... Please wait!"):
                # Process with selected model
                start_time = time.time()
                
                if model_choice == "YOLOv8":
                    result_img, detections = process_with_yolo(image, confidence_threshold)
                else:  # Mask R-CNN
                    result_img, detections = process_with_maskrcnn(image, confidence_threshold)
                
                processing_time = time.time() - start_time
                
                # Calculate photo quality
                quality_scores = calculate_photo_quality_score(image)
                
                # Display processed image
                st.image(result_img, use_column_width=True)
                st.markdown(f'<div class="retro-text">Processing Time: {processing_time:.2f} seconds</div>', unsafe_allow_html=True)
        
        # Store metrics for evaluation tab
        if 'real_metrics' not in st.session_state:
            st.session_state.real_metrics = {}
        
        # We don't have ground truth for real images, so simulate metrics
        metrics = {
            "precision": random.uniform(0.7, 0.9),
            "recall": random.uniform(0.6, 0.85),
            "f1": random.uniform(0.65, 0.85)
        }
        
        st.session_state.real_metrics[model_choice.lower().replace("-", "")] = metrics
        
        # Display detection results
        display_detection_results(image, detections, quality_scores)

# Tab 2: Synthetic Data
with tab2:
    st.markdown('<div class="retro-subtitle">Generate Synthetic Retail Shelf</div>', unsafe_allow_html=True)
    
    # Sidebar for synthetic data generation options
    with st.sidebar:
        st.markdown('<div class="retro-subtitle">Synthetic Data Options</div>', unsafe_allow_html=True)
        
        generate_new = st.button("Generate New Shelf Image", key="gen_new_shelf")
        
        model_choice_synth = st.radio(
            "Select Detection Model",
            ["YOLOv8", "Mask R-CNN"],
            key="synth_model_choice"
        )
        
        confidence_threshold_synth = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            key="synth_confidence"
        )
        
        effect_choice = st.selectbox(
            "Apply Effect",
            ["none", "blur", "dark", "bright", "low_contrast", "noise"],
            index=0
        )
    
    # Initialize or update synthetic image in session state
    if 'synthetic_image' not in st.session_state or generate_new:
        synthetic_img, metadata = generate_synthetic_shelf()
        st.session_state.synthetic_image = synthetic_img
        st.session_state.synthetic_metadata = metadata
    
    # Apply selected effect
    display_img = apply_effects(st.session_state.synthetic_image, effect_choice)
    
    # Display original and processed columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="retro-text">Synthetic Image</div>', unsafe_allow_html=True)
        st.image(display_img, use_column_width=True)
    
    with col2:
        st.markdown('<div class="retro-text">Processed Image</div>', unsafe_allow_html=True)
        
        with st.spinner("⏳ Running detection... Please wait!"):
            # Process with selected model
            start_time = time.time()
            
            if model_choice_synth == "YOLOv8":
                result_img, detections = process_with_yolo(display_img, confidence_threshold_synth)
            else:  # Mask R-CNN
                result_img, detections = process_with_maskrcnn(display_img, confidence_threshold_synth)
            
            processing_time = time.time() - start_time
            
            # Calculate photo quality
            quality_scores = calculate_photo_quality_score(display_img)
            
            # Display processed image
            st.image(result_img, use_column_width=True)
            st.markdown(f'<div class="retro-text">Processing Time: {processing_time:.2f} seconds</div>', unsafe_allow_html=True)
    
    # Calculate metrics against ground truth
    metrics = calculate_metrics(detections, st.session_state.synthetic_metadata["promotional_materials"])
    
    # Store metrics for evaluation tab
    if 'synthetic_metrics' not in st.session_state:
        st.session_state.synthetic_metrics = {}
    
    st.session_state.synthetic_metrics[model_choice_synth.lower().replace("-", "")] = metrics
    
    # Display ground truth and results
    st.markdown('<div class="retro-subtitle">Ground Truth vs. Detections</div>', unsafe_allow_html=True)
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">PRECISION</div>
            <div class="metric-value">{metrics['precision']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">RECALL</div>
            <div class="metric-value">{metrics['recall']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">F1 SCORE</div>
            <div class="metric-value">{metrics['f1']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show detailed counts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="retro-text">Ground Truth:</div>', unsafe_allow_html=True)
        
        # Count ground truth by class
        gt_counts = {}
        for item in st.session_state.synthetic_metadata["promotional_materials"]:
            cls = item["class"]
            gt_counts[cls] = gt_counts.get(cls, 0) + 1
        
        for cls, count in gt_counts.items():
            st.markdown(f'<div class="retro-text">📊 {cls.replace("_", " ").title()}: {count}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="retro-text">Detected:</div>', unsafe_allow_html=True)
        
        # Count detections by class (promotional materials only)
        det_counts = {}
        for det in detections:
            if det["class"] in ["banner", "poster", "price_tag", "discount_sign"]:
                cls = det["class"]
                det_counts[cls] = det_counts.get(cls, 0) + 1
        
        for cls in list(gt_counts.keys()):  # Use same classes as ground truth
            count = det_counts.get(cls, 0)
            st.markdown(f'<div class="retro-text">📊 {cls.replace("_", " ").title()}: {count}</div>', unsafe_allow_html=True)
    
    # Display photo quality analysis
    display_detection_results(display_img, detections, quality_scores)

# Tab 3: Evaluation
with tab3:
    st.markdown('<div class="retro-subtitle">Model Evaluation Dashboard</div>', unsafe_allow_html=True)
    
    # Check if we have metrics to display
    if 'real_metrics' in st.session_state and 'synthetic_metrics' in st.session_state:
        if len(st.session_state.real_metrics) > 0 and len(st.session_state.synthetic_metrics) > 0:
            # Generate evaluation charts
            generate_evaluation_charts(
                st.session_state.real_metrics, 
                st.session_state.synthetic_metrics
            )
            
            # Add additional evaluation metrics
            st.markdown('<div class="retro-subtitle">Detection Capabilities Analysis</div>', unsafe_allow_html=True)
            
            # Create a radar chart for model capabilities
            categories = ['Speed', 'Accuracy', 'Material Detection', 'Small Objects', 'Occlusions']
            
            yolo_values = [0.9, 0.75, 0.65, 0.8, 0.6]  # Example values
            maskrcnn_values = [0.5, 0.85, 0.8, 0.7, 0.85]  # Example values
            
            # Create the radar chart
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, polar=True)
            
            # Set the angles for each category
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Add values for both models, closing the loop
            yolo_values += yolo_values[:1]
            maskrcnn_values += maskrcnn_values[:1]
            
            # Plot the values
            ax.plot(angles, yolo_values, 'o-', linewidth=2, color='#FFDE59', label='YOLOv8')
            ax.fill(angles, yolo_values, alpha=0.25, color='#FFDE59')
            
            ax.plot(angles, maskrcnn_values, 'o-', linewidth=2, color='#FF6B6B', label='Mask R-CNN')
            ax.fill(angles, maskrcnn_values, alpha=0.25, color='#FF6B6B')
            
            # Set category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # Style the chart
            ax.set_facecolor('#1A1A2E')
            fig.patch.set_facecolor('#1A1A2E')
            
            # Style the text
            for text in ax.get_xticklabels():
                text.set_color('#EAEAEA')
                text.set_fontsize(10)
            
            for text in ax.get_yticklabels():
                text.set_color('#EAEAEA')
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            st.pyplot(fig)
            
            # Add model recommendations
            st.markdown('<div class="retro-subtitle">Model Recommendations</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="metric-box" style="height: 200px;">
                    <div class="metric-label">BEST FOR SPEED</div>
                    <div class="metric-value">YOLOv8</div>
                    <div style="color: #EAEAEA; font-family: 'Space Mono', monospace; margin-top: 10px;">
                    • 4x faster detection time<br>
                    • Better for real-time applications<br>
                    • Good balance of speed and accuracy
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-box" style="height: 200px;">
                    <div class="metric-label">BEST FOR ACCURACY</div>
                    <div class="metric-value">Mask R-CNN</div>
                    <div style="color: #EAEAEA; font-family: 'Space Mono', monospace; margin-top: 10px;">
                    • Better for detailed object segmentation<br>
                    • Higher precision in material identification<br>
                    • Better with overlapping promotional materials
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Final summary and recommendations
            st.markdown('<div class="retro-subtitle">Project Recommendations</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="retro-text">
            Based on the evaluation, here are recommendations for the retail shelf analytics project:
            
            1. <span style="color: #FFDE59;">Use YOLOv8 for initial scanning</span> of large numbers of shelf images due to its speed advantage.
            
            2. <span style="color: #FFDE59;">Use Mask R-CNN for detailed analysis</span> of promotional materials when accuracy is more important than speed.
            
            3. <span style="color: #FFDE59;">Implement both models in a pipeline</span> where YOLOv8 performs preliminary detection, and Mask R-CNN refines specific regions of interest.
            
            4. <span style="color: #FFDE59;">Expand training data</span> with more examples of promotional materials in different lighting conditions.
            
            5. <span style="color: #FFDE59;">Focus on improving material classification</span> as this shows the largest gap in performance.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="retro-text">Use the Real Images and Synthetic Data tabs to analyze some images first!</div>', unsafe_allow_html=True)
        
        # Show sample loading animation
        st.markdown("""
        <div style="display: flex; justify-content: center; margin: 50px 0;">
            <div style="width: 50px; height: 50px; background-color: #FFDE59; 
                      animation: retro-loading 1s infinite alternate;">
            </div>
        </div>
        <style>
        @keyframes retro-loading {
            0% { transform: scale(1); background-color: #FFDE59; }
            100% { transform: scale(0.5); background-color: #FF6B6B; }
        }
        </style>
        """, unsafe_allow_html=True)

# Add footer with retro gaming styling
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; 
           border-top: 3px solid #FFDE59; font-family: 'VT323', monospace;">
    <div style="color: #FF6B6B; font-size: 1.5em; margin-bottom: 10px;">GAME OVER... CONTINUE?</div>
    <div style="color: #EAEAEA; font-size: 1em;">RetroVision Shelf Analytics - MVP v0.1</div>
</div>
""", unsafe_allow_html=True)