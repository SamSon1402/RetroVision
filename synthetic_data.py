import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import random
import json
import os

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

# Generate and save a batch of synthetic images
def generate_dataset(num_images=10, output_dir="synthetic_data"):
    """Generate a dataset of synthetic shelf images with annotations"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create images subdirectory
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    # Create annotations subdirectory
    annotations_dir = os.path.join(output_dir, "annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    
    for i in range(num_images):
        # Generate synthetic image and metadata
        img, metadata = generate_synthetic_shelf()
        
        # Apply random effect
        effect = random.choice(["none", "blur", "dark", "bright", "low_contrast", "noise"])
        img_with_effect = apply_effects(img, effect)
        
        # Save image
        image_filename = f"shelf_{i+1:03d}.png"
        img_with_effect.save(os.path.join(images_dir, image_filename))
        
        # Save annotation
        annotation_filename = f"shelf_{i+1:03d}.json"
        
        # Add effect information and quality score to metadata
        quality_scores = calculate_photo_quality_score(img_with_effect)
        metadata["effect"] = effect
        metadata["quality_scores"] = quality_scores
        
        with open(os.path.join(annotations_dir, annotation_filename), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Generated {num_images} synthetic shelf images in {output_dir}")

if __name__ == "__main__":
    # Generate a small dataset when run directly
    generate_dataset(num_images=5)