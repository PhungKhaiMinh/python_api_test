"""
Image Processor Module  
Handles image processing operations using PaddleOCR-based algorithms
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance


def detect_hmi_screen(image):
    """
    Detect and extract HMI screen from image
    Uses PaddleOCR-based HMI detection algorithm
    
    Args:
        image: OpenCV image (numpy array BGR)
        
    Returns:
        tuple: (extracted_screen, processing_time)
    """
    import time
    start_time = time.time()
    
    try:
        if image is None or len(image.shape) != 3:
            return None, time.time() - start_time
        
        # Use PaddleOCR-based HMI detection algorithm
        from .paddleocr_engine import detect_hmi_screen_paddle
        hmi_screen, proc_time = detect_hmi_screen_paddle(image)
        
        if hmi_screen is not None and hmi_screen.size > 0:
            return hmi_screen, time.time() - start_time
        
        return None, time.time() - start_time
            
    except Exception as e:
        print(f"[ERROR] detect_hmi_screen: {e}")
        return None, time.time() - start_time


def detect_hmi_screen_paddle(image):
    """
    Wrapper for PaddleOCR-based HMI detection
    
    Returns:
        tuple: (extracted_screen, processing_time)
    """
    try:
        from .paddleocr_engine import detect_hmi_screen_paddle as _detect_hmi
        return _detect_hmi(image)
    except Exception as e:
        print(f"[ERROR] detect_hmi_screen_paddle: {e}")
        return None, 0


def preprocess_hmi_image(image, roi_coordinates, original_filename):
    """
    Preprocess HMI image and extract ROIs
    
    Returns:
        list: List of preprocessed ROIs
    """
    results = []
    
    try:
        img_height, img_width = image.shape[:2]
        
        for i, coords in enumerate(roi_coordinates):
            if len(coords) != 4:
                continue
            
            # Convert coordinates
            is_normalized = any(isinstance(value, float) and 0 <= value <= 1 for value in coords)
            
            if is_normalized:
                x1, y1, x2, y2 = coords
                x1, x2 = int(x1 * img_width), int(x2 * img_width)
                y1, y2 = int(y1 * img_height), int(y2 * img_height)
            else:
                x1, y1, x2, y2 = [int(float(c)) for c in coords]
            
            # Ensure correct order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Validate coordinates
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
                continue
            
            # Crop ROI
            roi = image[y1:y2, x1:x2]
            
            if roi.size > 0:
                results.append({
                    "roi_index": i,
                    "roi_image": roi,
                    "coordinates": (x1, y1, x2, y2)
                })
    
    except Exception as e:
        print(f"[ERROR] preprocess_hmi_image: {e}")
    
    return results


def preprocess_roi_for_ocr(roi, roi_index=0, original_filename="", roi_name=None):
    """
    Preprocess ROI for better OCR
    """
    try:
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Resize if ROI is too small
        h, w = gray.shape
        if h < 30 or w < 30:
            scale = max(30/h, 30/w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return morph
        
    except Exception as e:
        print(f"[ERROR] preprocess_roi_for_ocr: {e}")
        return roi


def check_image_quality(image):
    """
    Check image quality
    
    Returns:
        dict: Image quality information
    """
    result = {
        'is_good_quality': True,
        'issues': [],
        'blurriness': 0,
        'brightness': 0,
        'contrast': 0
    }
    
    if image is None or image.size == 0:
        result['is_good_quality'] = False
        result['issues'].append('empty_image')
        return result
    
    # Convert to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Check brightness
    brightness = np.mean(gray)
    result['brightness'] = brightness
    
    # Check contrast
    contrast = np.std(gray)
    result['contrast'] = contrast
    
    # Check blurriness using Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result['blurriness'] = laplacian_var
    
    blur_threshold = 7.0
    if laplacian_var < blur_threshold:
        result['is_good_quality'] = False
        result['issues'].append('blurry')
        return result
    
    if brightness > 220:
        result['is_good_quality'] = False
        result['issues'].append('too_bright')
        return result
    
    if contrast < 16:
        result['is_good_quality'] = False
        result['issues'].append('low_contrast')
        return result
    
    return result


def enhance_image_quality(image, quality_info):
    """Enhance image quality based on check results"""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    enhanced = gray.copy()
    
    # Handle blurry images
    if 'blurry' in quality_info['issues']:
        kernel = np.array([[-1, -1, -1],
                           [-1, 10, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Handle too bright images
    if 'too_bright' in quality_info['issues']:
        alpha = 0.9
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=0)
    
    return enhanced


def enhance_image(image):
    """General image enhancement"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def adaptive_edge_detection(image):
    """Adaptive edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Auto threshold Canny
    median = np.median(blurred)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    
    edges = cv2.Canny(blurred, lower, upper)
    
    return edges


def extract_content_region(img):
    """Extract main content region from image"""
    return img


def fine_tune_hmi_screen(image, roi_coords):
    """Fine-tune HMI screen detection"""
    return image


# Placeholder for backward compatibility
def init_cv2_detectors(sift_det=None, flann_match=None):
    """Initialize CV2 detectors - deprecated, kept for backward compatibility"""
    pass


class ImageAligner:
    """
    Image alignment class - simplified for PaddleOCR workflow
    The main alignment is now done in paddleocr_engine
    """
    
    def __init__(self, template_img=None, source_img=None):
        self.template_img = template_img.copy() if template_img is not None else None
        self.source_img = source_img.copy() if source_img is not None else None
        self.warped_img = None
        
    def align_images(self):
        """Return source image as-is since alignment is now handled differently"""
        return self.source_img.copy() if self.source_img is not None else None
    
    def get_homography_matrix(self):
        return None
    
    def transform_roi_coordinates(self, roi_coordinates):
        return roi_coordinates


def preprocess_hmi_image_with_alignment(image, template_path, roi_coordinates, original_filename):
    """Preprocess with alignment - simplified version"""
    return preprocess_hmi_image(image, roi_coordinates, original_filename)
