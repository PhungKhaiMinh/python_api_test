"""
PaddleOCR Engine Module
Handles PaddleOCR initialization, image processing and text extraction
Replaces EasyOCR with optimized PaddleOCR implementation
"""

import os
import sys
import io
import warnings
import contextlib
import cv2
import numpy as np
import re
from math import sqrt, atan2, degrees
from PIL import Image, ImageEnhance
import time

# ============================================================
# ENVIRONMENT CONFIGURATION - Must be set BEFORE importing PaddleOCR
# ============================================================
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['GLOG_minloglevel'] = '3'
os.environ['FLAGS_call_stack_level'] = '0'
os.environ['PADDLE_PDX_SILENT_MODE'] = '1'

warnings.filterwarnings('ignore')


@contextlib.contextmanager
def suppress_output():
    """Temporarily suppress stdout and stderr to hide unnecessary messages"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# Fix encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass


# ============================================================
# GLOBAL VARIABLES
# ============================================================
_paddle_ocr_instance = None
_paddle_ocr_initialized = False
IOU_THRESHOLD = 0.01  # IoU 1% threshold
HAS_PADDLEOCR = False  # Will be set after initialization

# GPU accelerator reference (will be set by app)
_gpu_accelerator = None


def load_roi_info():
    """Load ROI info from cache or file"""
    try:
        from .cache_manager import get_roi_info_cached
        return get_roi_info_cached()
    except Exception as e:
        print(f"[ERROR] Failed to load ROI info: {e}")
        return {}


def init_paddleocr_globals(gpu_acc=None):
    """Initialize global PaddleOCR variables from app"""
    global _gpu_accelerator
    _gpu_accelerator = gpu_acc


def get_paddleocr_instance():
    """Get or create PaddleOCR instance (singleton pattern for speed)"""
    global _paddle_ocr_instance, _paddle_ocr_initialized, HAS_PADDLEOCR
    
    if _paddle_ocr_instance is None:
        print("[*] Initializing PaddleOCR reader...")
        
        try:
            # Import PaddleOCR AFTER environment variables are set
            with suppress_output():
                from paddleocr import PaddleOCR
            
            # Initialize with optimized parameters
            with suppress_output():
                _paddle_ocr_instance = PaddleOCR(
                    lang='en',
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_thresh=0.15,
                    text_det_box_thresh=0.25,
                    text_det_unclip_ratio=2.2,
                    text_rec_score_thresh=0.0,
                    text_det_limit_side_len=512,
                    text_det_limit_type='max',
                )
            
            _paddle_ocr_initialized = True
            HAS_PADDLEOCR = True
            print("[OK] PaddleOCR initialized successfully")
            
        except ImportError as e:
            print(f"[ERROR] PaddleOCR not installed: {e}")
            print("Please run: pip install paddleocr paddlepaddle")
            _paddle_ocr_instance = None
            _paddle_ocr_initialized = False
            HAS_PADDLEOCR = False
        except Exception as e:
            print(f"[ERROR] Failed to initialize PaddleOCR: {e}")
            _paddle_ocr_instance = None
            _paddle_ocr_initialized = False
            HAS_PADDLEOCR = False
    
    return _paddle_ocr_instance


def is_paddleocr_available():
    """Check if PaddleOCR is available and initialized"""
    return _paddle_ocr_initialized and _paddle_ocr_instance is not None


# ============================================================
# HMI DETECTION FUNCTIONS
# ============================================================

def enhance_image_for_hmi(image):
    """Enhance image quality before edge detection"""
    # Convert from OpenCV (BGR) to PIL (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Increase contrast with PIL
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_pil = enhancer.enhance(2)
    
    # Convert back to OpenCV format
    enhanced_image = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
    
    # Continue with standard image processing
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(11, 11))
    enhanced = clahe.apply(gray)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=0)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return blurred, enhanced


def adaptive_edge_detection_hmi(image):
    """Edge detection with multiple methods combined"""
    median_val = np.median(image)
    lower = int(max(0, (1.0 - 0.33) * median_val))
    upper = int(min(255, (1.0 + 0.33) * median_val))
    canny_edges = cv2.Canny(image, lower, upper)
    
    # Use larger kernel for Sobel filter
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, sobel_edges = cv2.threshold(sobel_edges, 80, 255, cv2.THRESH_BINARY)
    
    # Combine both edge detection methods
    combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
    
    # Dilate to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=2)
    final_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    
    return final_edges


def process_lines(lines, img_shape, min_length=20, max_lines_per_direction=30):
    """Process and group lines by horizontal/vertical direction"""
    if lines is None:
        return [], []
    
    horizontal_lines = []
    vertical_lines = []
    all_h_lines = []
    all_v_lines = []
    
    height, width = img_shape[:2]
    min_length = max(min_length, int(min(height, width) * 0.02))
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if length < min_length:
            continue
        
        if x2 != x1:
            angle = degrees(atan2(y2-y1, x2-x1))
        else:
            angle = 90
        
        if abs(angle) < 40 or abs(angle) > 140:
            all_h_lines.append([x1, y1, x2, y2, angle, length])
        elif abs(angle - 90) < 40 or abs(angle + 90) < 40:
            all_v_lines.append([x1, y1, x2, y2, angle, length])
    
    # Sort by length
    all_h_lines.sort(key=lambda x: x[5], reverse=True)
    all_v_lines.sort(key=lambda x: x[5], reverse=True)
    
    min_lines = min(4, len(all_h_lines))
    horizontal_lines = [line[:5] for line in all_h_lines[:max(min_lines, max_lines_per_direction)]]
    
    min_lines = min(4, len(all_v_lines))
    vertical_lines = [line[:5] for line in all_v_lines[:max(min_lines, max_lines_per_direction)]]
    
    return horizontal_lines, vertical_lines


def find_rectangle_from_lines(horizontal_lines, vertical_lines, img_shape):
    """Find rectangle from classified horizontal and vertical lines"""
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None
    
    top_line = min(horizontal_lines, key=lambda line: min(line[1], line[3]))
    bottom_line = max(horizontal_lines, key=lambda line: max(line[1], line[3]))
    left_line = min(vertical_lines, key=lambda line: min(line[0], line[2]))
    right_line = max(vertical_lines, key=lambda line: max(line[0], line[2]))
    
    top_y = min(top_line[1], top_line[3])
    bottom_y = max(bottom_line[1], bottom_line[3])
    left_x = min(left_line[0], left_line[2])
    right_x = max(right_line[0], right_line[2])
    
    height, width = img_shape[:2]
    
    if left_x < 0: left_x = 0
    if top_y < 0: top_y = 0
    if right_x >= width: right_x = width - 1
    if bottom_y >= height: bottom_y = height - 1
    
    rect_width = right_x - left_x
    rect_height = bottom_y - top_y
    
    if rect_width < 20 or rect_height < 20:
        return None
    
    aspect_ratio = max(rect_width, rect_height) / (min(rect_width, rect_height) + 1e-6)
    if aspect_ratio > 5:
        return None
    
    area = rect_width * rect_height
    total_area = height * width
    if area < 0.01 * total_area or area > 0.9 * total_area:
        return None
    
    top_left = (int(left_x), int(top_y))
    top_right = (int(right_x), int(top_y))
    bottom_right = (int(right_x), int(bottom_y))
    bottom_left = (int(left_x), int(bottom_y))
    
    return (top_left, top_right, bottom_right, bottom_left, area)


def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_content_region(img):
    """Extract content region from image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced_contrast = cv2.convertScaleAbs(gray, alpha=1.3, beta=5)
    blurred = cv2.GaussianBlur(enhanced_contrast, (3, 3), 0)
    
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobel_x, sobel_y)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, gradient_thresh = cv2.threshold(gradient_mag, 20, 255, cv2.THRESH_BINARY)
    
    vertical_kernel = np.ones((11, 3), np.uint8)
    gradient_dilated = cv2.dilate(gradient_thresh, vertical_kernel, iterations=3)
    horizontal_kernel = np.ones((3, 9), np.uint8)
    gradient_dilated = cv2.dilate(gradient_dilated, horizontal_kernel, iterations=2)
    
    kernel = np.ones((5, 5), np.uint8)
    gradient_final = cv2.morphologyEx(gradient_dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(gradient_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        enhanced_for_threshold = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        _, thresh = cv2.threshold(enhanced_for_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = img.shape[0] * img.shape[1] * 0.005
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    mask = np.zeros_like(gray)
    if large_contours:
        largest_contour = max(large_contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        return mask, largest_contour
    
    mask.fill(255)
    return mask, None


def fine_tune_hmi_screen(image, roi_coords):
    """Fine-tune detected HMI screen region"""
    x_min, y_min, x_max, y_max = roi_coords
    roi = image[y_min:y_max, x_min:x_max]
    
    content_mask, largest_contour = extract_content_region(roi)
    
    if largest_contour is None:
        return roi, roi_coords
    
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < 0.1 * roi.shape[0] * roi.shape[1]:
        return roi, roi_coords
    
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = np.array(box, dtype=np.int32)
    
    points = approx.reshape(-1, 2)
    points = order_points(points)
    
    width_a = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    width_b = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    height_b = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    if max_width < 10 or max_height < 10:
        return roi, roi_coords
    
    dst_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    src_points = points.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(roi, M, (max_width, max_height))
    
    new_roi_coords = (x_min, y_min, x_min + warped.shape[1], y_min + warped.shape[0])
    
    return warped, new_roi_coords


def detect_hmi_screen_paddle(image):
    """
    Detect and extract HMI screen from image using PaddleOCR algorithm
    
    Args:
        image: OpenCV image (numpy array BGR)
        
    Returns:
        tuple: (extracted_screen, processing_time)
    """
    start_time = time.time()
    
    try:
        if image is None or len(image.shape) != 3:
            return None, time.time() - start_time
        
        # Step 1: Enhance image
        enhanced_img, enhanced_clahe = enhance_image_for_hmi(image)
        
        # Step 2: Edge detection
        edges = adaptive_edge_detection_hmi(enhanced_clahe)
        
        # Step 3: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = image.shape[0] * image.shape[1] * 0.001
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        contour_mask = np.zeros_like(edges)
        cv2.drawContours(contour_mask, large_contours, -1, 255, 2)
        
        # Step 4: Detect lines
        lines = cv2.HoughLinesP(contour_mask, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=30)
        
        if lines is None or len(lines) < 2:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=10, maxLineGap=40)
            if lines is None or len(lines) < 2:
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=50)
        
        if lines is None:
            return None, time.time() - start_time
        
        # Step 5: Classify horizontal/vertical lines
        horizontal_lines, vertical_lines = process_lines(lines, image.shape, min_length=20)
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None, time.time() - start_time
        
        # Step 6: Find rectangle from lines
        largest_rectangle = find_rectangle_from_lines(horizontal_lines, vertical_lines, image.shape)
        
        if largest_rectangle is None:
            return None, time.time() - start_time
        
        # Step 7: Extract HMI region
        top_left, top_right, bottom_right, bottom_left, _ = largest_rectangle
        
        x_min = min(top_left[0], bottom_left[0])
        y_min = min(top_left[1], top_right[1])
        x_max = max(top_right[0], bottom_right[0])
        y_max = max(bottom_left[1], bottom_right[1])
        
        # Validate bounds
        if x_min < 0: x_min = 0
        if y_min < 0: y_min = 0
        if x_max >= image.shape[1]: x_max = image.shape[1] - 1
        if y_max >= image.shape[0]: y_max = image.shape[0] - 1
        
        if x_max > x_min and y_max > y_min:
            roi_coords = (x_min, y_min, x_max, y_max)
            
            # Fine-tune and warp HMI region
            warped_roi, refined_coords = fine_tune_hmi_screen(image, roi_coords)
            
            processing_time = time.time() - start_time
            print(f"[OK] HMI screen extracted in {processing_time:.2f}s")
            return warped_roi, processing_time
        
        return None, time.time() - start_time
        
    except Exception as e:
        print(f"[ERROR] HMI detection failed: {e}")
        return None, time.time() - start_time


# ============================================================
# OCR FUNCTIONS
# ============================================================

def read_image_with_paddleocr(image_input):
    """
    Read text from image using PaddleOCR
    
    Args:
        image_input: Can be file path or numpy array (OpenCV image)
    
    Returns:
        tuple: (results, img_width, img_height)
    """
    ocr = get_paddleocr_instance()
    
    if ocr is None:
        print("[ERROR] PaddleOCR not available")
        return None, 0, 0
    
    start_time = time.time()
    
    try:
        if isinstance(image_input, np.ndarray):
            img_height, img_width = image_input.shape[:2]
            # Save to temp file for PaddleOCR
            temp_path = "_temp_paddle_ocr.jpg"
            cv2.imwrite(temp_path, image_input)
            results = ocr.predict(temp_path)
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            img = cv2.imread(image_input)
            if img is not None:
                img_height, img_width = img.shape[:2]
            else:
                img_height, img_width = 1, 1
            results = ocr.predict(image_input)
        
        elapsed = time.time() - start_time
        print(f"[OK] PaddleOCR completed in {elapsed:.2f}s")
        
        return results, img_width, img_height
        
    except Exception as e:
        print(f"[ERROR] PaddleOCR failed: {e}")
        return None, 0, 0


def extract_ocr_data(results):
    """Extract data from PaddleOCR results"""
    all_data = []
    
    if not results:
        return all_data
    
    for result in results:
        if hasattr(result, 'json') and result.json:
            json_data = result.json
            res = json_data.get('res', json_data)
            
            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            polys = res.get('rec_polys', res.get('dt_polys', []))
            
            for i in range(len(texts)):
                data = {
                    'text': texts[i] if i < len(texts) else '',
                    'confidence': scores[i] if i < len(scores) else 0.0,
                    'bbox': polys[i] if i < len(polys) else []
                }
                all_data.append(data)
    
    return all_data


# ============================================================
# TEXT NORMALIZATION AND MATCHING FUNCTIONS
# ============================================================

def normalize_text(text):
    """
    Normalize text for fuzzy matching
    - Remove : and extra whitespace
    - Convert to uppercase
    - Normalize "ST14 - LEAK" → "ST14-LEAK"
    """
    if not text:
        return ""
    normalized = text.strip().upper()
    normalized = normalized.rstrip(':')
    normalized = re.sub(r'\s*-\s*', '-', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def fuzzy_match(text1, text2, threshold=0.75):
    """
    Fuzzy compare between 2 strings, returns True if similarity >= threshold
    Uses Levenshtein distance algorithm
    """
    s1 = normalize_text(text1)
    s2 = normalize_text(text2)
    
    if not s1 or not s2:
        return False
    
    len1, len2 = len(s1), len(s2)
    min_len = min(len1, len2)
    max_len = max(len1, len2)
    
    # Check if one string contains the other
    if s1 in s2 or s2 in s1:
        length_ratio = min_len / max_len
        if length_ratio >= 0.7:
            return True
    
    # If lengths differ too much (>40%), no match
    if abs(len1 - len2) > max_len * 0.4:
        return False
    
    # Calculate Levenshtein distance
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    distance = dp[len1][len2]
    similarity = 1 - (distance / max_len)
    
    return similarity >= threshold


# ============================================================
# SCREEN MATCHING FUNCTIONS
# ============================================================

def find_matching_screen(ocr_data, roi_info, selected_area=None, selected_machine=None, debug=False):
    """
    Find best matching screen and sub_page based on Special_rois
    
    Structure: machines > machine_type > machine_code > screens > screen_name > sub_pages > sub_page
    
    Args:
        ocr_data: Extracted OCR data
        roi_info: ROI data from roi_info.json
        selected_area: Selected area/machine_type (F1, F4, ...) - if None will search all
        selected_machine: Selected machine code - if None will search all in area
        debug: Show debug info
    
    Returns:
        tuple: (machine_type, machine_code, screen_name, sub_page, sub_page_data, match_count, match_percentage)
    """
    if not roi_info or 'machines' not in roi_info:
        return None, None, None, None, None, 0, 0
    
    # Get all normalized OCR texts
    ocr_texts = [normalize_text(item['text']) for item in ocr_data]
    
    if debug:
        print(f"\n   [DEBUG] OCR detected {len(ocr_texts)} text items")
        if selected_area and selected_machine:
            print(f"   [DEBUG] Filter by: {selected_area}/{selected_machine}")
    
    best_match = None
    best_match_count = 0
    best_match_percentage = 0
    all_matches = []
    
    # Structure: machines > machine_type (F1) > machine_code (IE-F1-CWA01) > screens > screen_name > sub_pages
    for machine_type, machine_type_data in roi_info['machines'].items():
        # Filter by area (machine_type)
        if selected_area and machine_type != selected_area:
            continue
        
        if not isinstance(machine_type_data, dict):
            continue
        
        # Iterate through machine_codes under this machine_type
        for machine_code, machine_data in machine_type_data.items():
            # Skip non-machine entries like "screens" key in old structure
            if machine_code == "screens":
                continue
            
            # Filter by machine_code
            if selected_machine and machine_code != selected_machine:
                continue
            
            if not isinstance(machine_data, dict) or 'screens' not in machine_data:
                continue
            
            for screen_name, screen_data in machine_data['screens'].items():
                # Check structure with sub_pages
                if isinstance(screen_data, dict) and 'sub_pages' in screen_data:
                    for sub_page, sub_page_data in screen_data['sub_pages'].items():
                        # Look for Special_rois (capital S)
                        special_rois = sub_page_data.get('Special_rois', sub_page_data.get('special_rois', []))
                        if not special_rois:
                            continue
                        
                        match_count = 0
                        matched_rois = []
                        
                        for special_roi in special_rois:
                            special_roi_normalized = normalize_text(special_roi)
                            
                            for ocr_text in ocr_texts:
                                if fuzzy_match(special_roi_normalized, ocr_text):
                                    match_count += 1
                                    matched_rois.append(special_roi)
                                    break
                        
                        match_percentage = (match_count / len(special_rois)) * 100 if len(special_rois) > 0 else 0
                        
                        all_matches.append({
                            'machine_type': machine_type,
                            'machine': machine_code,
                            'screen': screen_name,
                            'sub_page': sub_page,
                            'special_rois': special_rois,
                            'match_count': match_count,
                            'match_percentage': match_percentage,
                            'matched_rois': matched_rois
                        })
                        
                        if match_count > best_match_count or (match_count == best_match_count and match_percentage > best_match_percentage):
                            best_match_count = match_count
                            best_match_percentage = match_percentage
                            best_match = (machine_type, machine_code, screen_name, sub_page, sub_page_data)
                else:
                    # Old structure (no sub_pages) - screen_data might be a list
                    special_rois = []
                    if isinstance(screen_data, dict):
                        special_rois = screen_data.get('Special_rois', screen_data.get('special_rois', []))
                    
                    if not special_rois:
                        continue
                    
                    match_count = 0
                    matched_rois = []
                    
                    for special_roi in special_rois:
                        special_roi_normalized = normalize_text(special_roi)
                        
                        for ocr_text in ocr_texts:
                            if fuzzy_match(special_roi_normalized, ocr_text):
                                match_count += 1
                                matched_rois.append(special_roi)
                                break
                    
                    match_percentage = (match_count / len(special_rois)) * 100 if len(special_rois) > 0 else 0
                    
                    all_matches.append({
                        'machine_type': machine_type,
                        'machine': machine_code,
                        'screen': screen_name,
                        'sub_page': '1',
                        'special_rois': special_rois,
                        'match_count': match_count,
                        'match_percentage': match_percentage,
                        'matched_rois': matched_rois
                    })
                    
                    if match_count > best_match_count or (match_count == best_match_count and match_percentage > best_match_percentage):
                        best_match_count = match_count
                        best_match_percentage = match_percentage
                        best_match = (machine_type, machine_code, screen_name, "1", screen_data)
    
    if debug and all_matches:
        print(f"\n   [DEBUG] Screen matching results:")
        for m in all_matches:
            status = "✓" if m['match_count'] > 0 else "✗"
            print(f"      {status} {m['machine_type']}/{m['machine']}/{m['screen']}/sub-page {m['sub_page']}: "
                  f"{m['match_count']}/{len(m['special_rois'])} matches ({m['match_percentage']:.0f}%)")
    
    if best_match:
        return best_match[0], best_match[1], best_match[2], best_match[3], best_match[4], best_match_count, best_match_percentage
    
    return None, None, None, None, None, 0, 0


# ============================================================
# IoU FUNCTIONS
# ============================================================

def polygon_to_normalized_bbox(polygon, img_width, img_height):
    """
    Convert polygon from PaddleOCR to normalized bounding box [x1, y1, x2, y2]
    polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (4 corners of text box)
    """
    if not polygon or len(polygon) < 4:
        return None
    
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    norm_x1 = x_min / img_width
    norm_y1 = y_min / img_height
    norm_x2 = x_max / img_width
    norm_y2 = y_max / img_height
    
    return [norm_x1, norm_y1, norm_x2, norm_y2]


def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between 2 bounding boxes
    box format: [x1, y1, x2, y2] (normalized)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


def filter_ocr_by_roi(ocr_data, sub_page_data, img_width, img_height):
    """
    Filter OCR results based on IoU with ROIs
    Returns: list of filtered OCR results with matching ROI info
    """
    if not sub_page_data:
        return []
    
    # Support both "Rois" (capital) and "rois" (lowercase)
    rois = sub_page_data.get('Rois', sub_page_data.get('rois', []))
    if not rois:
        return []
    
    filtered_results = []
    
    for ocr_item in ocr_data:
        polygon = ocr_item.get('bbox', [])
        if not polygon:
            continue
        
        ocr_bbox = polygon_to_normalized_bbox(polygon, img_width, img_height)
        if not ocr_bbox:
            continue
        
        best_iou = 0
        best_roi_name = None
        best_roi_coords = None
        
        for roi in rois:
            roi_coords = roi.get('coordinates', [])
            if len(roi_coords) != 4:
                continue
            
            iou = calculate_iou(ocr_bbox, roi_coords)
            
            if iou > best_iou:
                best_iou = iou
                best_roi_name = roi.get('name', 'Unknown')
                best_roi_coords = roi_coords
        
        if best_iou >= IOU_THRESHOLD:
            filtered_results.append({
                'text': ocr_item['text'],
                'confidence': ocr_item['confidence'],
                'bbox': polygon,
                'normalized_bbox': ocr_bbox,
                'matched_roi': best_roi_name,
                'roi_coords': best_roi_coords,
                'iou': best_iou
            })
    
    return filtered_results


# ============================================================
# POST-PROCESSING FUNCTIONS
# ============================================================

def post_process_ocr_text(text, allowed_values=None):
    """Post-process OCR text - convert common mistakes"""
    if not text:
        return text
    
    # Convert single 'O' to '0'
    if len(text) == 1 and text.upper() == 'O':
        return '0'
    
    # Handle common character confusions
    if len(text) >= 2:
        chars_to_check = '01OUouIilC'
        suspicious_count = sum(1 for char in text if char in chars_to_check)
        
        if suspicious_count >= 2 and suspicious_count / len(text) >= 0.3:
            upper_text = text.upper()
            upper_no_dot = upper_text.replace('.', '')
            
            if re.search(r'[IUO0Q]{2}', upper_no_dot):
                temp_text = upper_text.replace('U', '0').replace('I', '1').replace('O', '0').replace('C', '0').replace('Q', '0')
                if temp_text.replace('.', '').replace('-', '').isdigit():
                    return temp_text
            
            digit_like_count = sum(1 for char in upper_text if char in 'OUICL')
            if digit_like_count / len(text) >= 0.7:
                cleaned = upper_text
                cleaned = cleaned.replace('O', '0').replace('U', '0').replace('Q', '0')
                cleaned = cleaned.replace('I', '1').replace('L', '1')
                cleaned = cleaned.replace('C', '0').replace('D', '0')
                cleaned = cleaned.replace(' ', '')
                
                if cleaned.replace('.', '').replace('-', '').isdigit():
                    return cleaned
    
    return text

