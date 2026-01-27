"""
PaddleOCR Engine Module - OPTIMIZED FOR SPEED
Handles PaddleOCR initialization, image processing and text extraction
Uses GPU acceleration and parallel processing for maximum performance
"""

import os
import sys
import io
import warnings
import contextlib
import cv2
import numpy as np
import re
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Disable oneDNN to fix PIR attribute conversion error
# This fixes: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]
os.environ['FLAGS_onednn'] = '0'  # Disable oneDNN
os.environ['FLAGS_use_mkldnn'] = '0'  # Disable MKLDNN (Intel MKL-DNN)

# GPU Memory optimization
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'  # Use 50% GPU memory
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'  # Immediate tensor cleanup

# Disable torch if it causes conflicts (PaddleOCR doesn't need torch)
# Unset torch-related environment variables that might interfere
torch_conflict_vars = [k for k in os.environ.keys() if 'torch' in k.lower() or 'pytorch' in k.lower()]
for var in torch_conflict_vars:
    if var.startswith('TORCH_') or var.startswith('PYTORCH_'):
        try:
            del os.environ[var]
        except:
            pass

# Add NVIDIA CUDA libraries to PATH for PaddlePaddle GPU
def _add_nvidia_to_path():
    """Add NVIDIA CUDA DLL paths to system PATH for paddlepaddle-gpu"""
    try:
        import site
        site_packages = site.getsitepackages()
        for sp in site_packages:
            # Check both direct nvidia folder and Lib/site-packages/nvidia
            possible_paths = [
                os.path.join(sp, 'nvidia'),
                os.path.join(sp, 'Lib', 'site-packages', 'nvidia'),
            ]
            for nvidia_base in possible_paths:
                if os.path.exists(nvidia_base):
                    # Add all nvidia/*/bin directories to PATH
                    for subdir in os.listdir(nvidia_base):
                        bin_path = os.path.join(nvidia_base, subdir, 'bin')
                        if os.path.exists(bin_path) and bin_path not in os.environ.get('PATH', ''):
                            os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')
    except Exception as e:
        print(f"[WARNING] Failed to add NVIDIA paths: {e}")

_add_nvidia_to_path()

# Pre-import paddle to avoid circular import issues with Flask debug mode
# Set oneDNN flags BEFORE importing paddle
try:
    # Force disable oneDNN before paddle import
    os.environ['FLAGS_onednn'] = '0'
    os.environ['FLAGS_use_mkldnn'] = '0'
    import paddle
    # Try to disable oneDNN in paddle if possible
    try:
        paddle.set_flags({'FLAGS_onednn': False})
    except:
        pass
except Exception as _paddle_import_error:
    paddle = None
    print(f"[WARNING] Paddle pre-import failed: {_paddle_import_error}")

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
USE_GPU = False  # Will be detected during initialization

# GPU accelerator reference (will be set by app)
_gpu_accelerator = None

# Thread pool for parallel OCR processing
_ocr_executor = None
MAX_OCR_WORKERS = 4  # Number of parallel OCR workers

# Thread safety and error tracking
_paddle_ocr_lock = threading.Lock()
_paddle_ocr_fail_count = 0
_paddle_ocr_max_fails = 3  # Reset instance after 3 consecutive failures

# Performance tracking
_perf_stats = {
    'ocr_calls': 0,
    'total_time': 0,
    'avg_time': 0
}


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


def reset_paddleocr_instance():
    """Reset PaddleOCR instance - useful when instance becomes corrupted"""
    global _paddle_ocr_instance, _paddle_ocr_initialized, HAS_PADDLEOCR, _paddle_ocr_fail_count
    
    with _paddle_ocr_lock:
        print("[*] Resetting PaddleOCR instance...")
        _paddle_ocr_instance = None
        _paddle_ocr_initialized = False
        _paddle_ocr_fail_count = 0
        
        # Try to clear GPU cache if available
        try:
            import gc
            gc.collect()
            # Try to clear Paddle CUDA cache if available
            try:
                if paddle is not None and paddle.device.is_compiled_with_cuda():
                    # Paddle doesn't have direct empty_cache, just garbage collect
                    print("[OK] GPU memory cleanup triggered")
            except:
                pass
        except:
            pass
        
        print("[OK] PaddleOCR instance reset completed")


def _detect_gpu_availability():
    """Detect if GPU is available for PaddlePaddle"""
    global USE_GPU
    try:
        # Use paddle module that was pre-imported at module level
        if paddle is not None:
            USE_GPU = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            if USE_GPU:
                gpu_name = "Unknown"
                try:
                    gpu_name = paddle.device.cuda.get_device_name(0)
                except:
                    pass
                print(f"[OK] PaddlePaddle GPU detected: {gpu_name}")
            return USE_GPU
        else:
            print("[WARNING] Paddle not imported, skipping GPU detection")
            USE_GPU = False
            return False
    except Exception as e:
        print(f"[WARNING] GPU detection failed: {e}")
        USE_GPU = False
        return False


def get_paddleocr_instance():
    """Get or create PaddleOCR instance (singleton pattern for speed)"""
    import sys
    global _paddle_ocr_instance, _paddle_ocr_initialized, HAS_PADDLEOCR, USE_GPU, _ocr_executor
    
    with _paddle_ocr_lock:
        # Return existing instance if already initialized
        if _paddle_ocr_instance is not None:
            print(f"[DEBUG] Returning existing PaddleOCR instance", flush=True)
            return _paddle_ocr_instance
        
        # Skip if we know it won't work (already failed permanently)
        # BUT allow retry if it's a DLL/torch error (might be fixable)
        if _paddle_ocr_initialized and _paddle_ocr_instance is None:
            # Allow one retry attempt for DLL errors
            if not hasattr(get_paddleocr_instance, '_retry_attempted'):
                print(f"[WARNING] PaddleOCR initialization previously failed, attempting retry...", flush=True)
                get_paddleocr_instance._retry_attempted = True
                # Reset initialization flag to allow retry
                _paddle_ocr_initialized = False
            else:
                print(f"[ERROR] PaddleOCR initialization previously failed, returning None", flush=True)
                sys.stdout.flush()
                return None
            
        print("[*] Initializing PaddleOCR reader (OPTIMIZED)...")
        
        try:
            # Detect GPU first
            _detect_gpu_availability()
            
            # Try to import PaddleOCR - handle DLL errors
            try:
                with suppress_output():
                    from paddleocr import PaddleOCR
            except Exception as import_error:
                error_str = str(import_error)
                if "torch" in error_str.lower() or "dll" in error_str.lower() or "WinError" in error_str:
                    print(f"[WARNING] PaddleOCR import failed due to DLL/torch conflict: {error_str}")
                    print("[*] Attempting to fix by unsetting torch-related environment variables...")
                    # Unset torch-related env vars that might cause conflicts
                    torch_vars = [k for k in os.environ.keys() if 'torch' in k.lower() or 'pytorch' in k.lower()]
                    for var in torch_vars:
                        if var in os.environ:
                            del os.environ[var]
                    # Retry import
                    try:
                        with suppress_output():
                            from paddleocr import PaddleOCR
                        print("[OK] PaddleOCR import succeeded after fixing environment")
                    except Exception as retry_error:
                        print(f"[ERROR] PaddleOCR import still failed after fix: {retry_error}")
                        raise import_error
                else:
                    raise import_error
            
            # ============================================================
            # OPTIMIZED PARAMETERS FOR SPEED
            # ============================================================
            # - text_det_limit_side_len: Larger = more accurate but slower
            #   960 is good balance for HMI screens (typically 800x600 to 1920x1080)
            # - text_det_db_score_mode: 'fast' for speed, 'slow' for accuracy
            # - use_angle_cls: False to skip text orientation classification
            # - text_det_thresh: Lower = detect more text but more noise
            # - text_det_box_thresh: Filter weak detections
            # ============================================================
            
            # Try initialization with error handling
            init_success = False
            last_error = None
            
            for attempt in range(2):  # Try twice
                try:
                    with suppress_output():
                        _paddle_ocr_instance = PaddleOCR(
                            lang='en',
                            # Disable unnecessary preprocessing
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False,
                            # Detection parameters - MATCHING OLD WORKING VERSION
                            text_det_thresh=0.15,          # Lower threshold to detect more text
                            text_det_box_thresh=0.25,      # Filter weak boxes
                            text_det_unclip_ratio=1.5,      # Standard ratio (old: 2.2, but using 1.5 for better accuracy)
                            text_det_limit_side_len=512,   # Standard size (old working version)
                            text_det_limit_type='max',
                            # Recognition parameters
                            text_rec_score_thresh=0.0,     # Keep all results (old working version)
                        )
                    init_success = True
                    break
                except Exception as init_error:
                    last_error = init_error
                    error_str = str(init_error)
                    if attempt == 0 and ("torch" in error_str.lower() or "dll" in error_str.lower() or "WinError" in error_str):
                        print(f"[WARNING] PaddleOCR initialization attempt {attempt + 1} failed: {error_str}")
                        print("[*] Retrying with environment cleanup...")
                        # Additional cleanup
                        import gc
                        gc.collect()
                        time.sleep(0.5)  # Brief pause
                    else:
                        raise init_error
            
            if not init_success:
                raise last_error
            
            _paddle_ocr_initialized = True
            HAS_PADDLEOCR = True
            
            # Initialize thread pool for parallel processing
            if _ocr_executor is None:
                _ocr_executor = ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS, thread_name_prefix="OCR")
            
            gpu_status = "GPU" if USE_GPU else "CPU"
            print(f"[OK] PaddleOCR initialized successfully ({gpu_status} mode)")
            
            # Warm up PaddleOCR instance
            try:
                print("[*] Warming up PaddleOCR instance...")
                warmup_start = time.time()
                
                # Create a small dummy image (white 100x100 image with some text-like patterns)
                dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                # Add some black lines to simulate text
                cv2.rectangle(dummy_image, (10, 40), (90, 60), (0, 0, 0), -1)
                
                # Use temp file for warmup (required by some PaddleOCR versions)
                temp_path = f"_temp_paddle_warmup_{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(temp_path, dummy_image)
                # Support both PaddleOCR 2.x and 3.x
                if hasattr(_paddle_ocr_instance, 'predict'):
                    _paddle_ocr_instance.predict(temp_path)
                else:
                    _paddle_ocr_instance.ocr(temp_path, cls=False)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                warmup_time = time.time() - warmup_start
                print(f"[OK] PaddleOCR warm-up completed in {warmup_time:.2f}s")
            except Exception as warmup_error:
                print(f"[WARNING] PaddleOCR warm-up failed (non-critical): {warmup_error}")
            
        except ImportError as e:
            print(f"[ERROR] PaddleOCR not installed: {e}", flush=True)
            print("Please run: pip install paddleocr paddlepaddle-gpu", flush=True)
            import traceback
            traceback.print_exc()
            _paddle_ocr_instance = None
            _paddle_ocr_initialized = True  # Mark as initialized (failed)
            HAS_PADDLEOCR = False
        except Exception as e:
            error_str = str(e)
            print(f"[ERROR] Failed to initialize PaddleOCR: {error_str}", flush=True)
            import traceback
            traceback.print_exc()
            
            # If it's a DLL/torch error, suggest solutions
            if "torch" in error_str.lower() or "dll" in error_str.lower() or "WinError" in error_str:
                print("\n[SUGGESTION] PaddleOCR failed due to DLL/torch conflict. Try:", flush=True)
                print("  1. Uninstall torch if not needed: pip uninstall torch", flush=True)
                print("  2. Or reinstall paddlepaddle-gpu: pip uninstall paddlepaddle-gpu && pip install paddlepaddle-gpu", flush=True)
                print("  3. Or use CPU version: pip uninstall paddlepaddle-gpu && pip install paddlepaddle", flush=True)
            
            _paddle_ocr_instance = None
            _paddle_ocr_initialized = True  # Mark as initialized (failed)
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
    Detect and extract HMI screen from image - OPTIMIZED FOR SPEED
    
    Optimizations:
    - Downscale image for faster processing
    - Optimized edge detection parameters
    - Early termination for invalid cases
    
    Args:
        image: OpenCV image (numpy array BGR)
        
    Returns:
        tuple: (extracted_screen, processing_time)
    """
    start_time = time.time()
    
    try:
        if image is None or len(image.shape) != 3:
            return None, time.time() - start_time
        
        original_shape = image.shape
        
        # OPTIMIZATION: Downscale for faster processing if image is large
        max_process_size = 800
        scale = 1.0
        if max(image.shape[:2]) > max_process_size:
            scale = max_process_size / max(image.shape[:2])
            process_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            process_image = image
        
        # Step 1: Enhance image
        enhanced_img, enhanced_clahe = enhance_image_for_hmi(process_image)
        
        # Step 2: Edge detection
        edges = adaptive_edge_detection_hmi(enhanced_clahe)
        
        # Step 3: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = process_image.shape[0] * process_image.shape[1] * 0.001
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        if not large_contours:
            return None, time.time() - start_time
        
        contour_mask = np.zeros_like(edges)
        cv2.drawContours(contour_mask, large_contours, -1, 255, 2)
        
        # Step 4: Detect lines - optimized thresholds
        lines = cv2.HoughLinesP(contour_mask, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=30)
        
        if lines is None or len(lines) < 2:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=10, maxLineGap=40)
            if lines is None or len(lines) < 2:
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=50)
        
        if lines is None:
            return None, time.time() - start_time
        
        # Step 5: Classify horizontal/vertical lines
        horizontal_lines, vertical_lines = process_lines(lines, process_image.shape, min_length=20)
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None, time.time() - start_time
        
        # Step 6: Find rectangle from lines
        largest_rectangle = find_rectangle_from_lines(horizontal_lines, vertical_lines, process_image.shape)
        
        if largest_rectangle is None:
            return None, time.time() - start_time
        
        # Step 7: Extract HMI region (scale back coordinates if needed)
        top_left, top_right, bottom_right, bottom_left, _ = largest_rectangle
        
        # Scale coordinates back to original image size
        if scale != 1.0:
            inv_scale = 1.0 / scale
            x_min = int(min(top_left[0], bottom_left[0]) * inv_scale)
            y_min = int(min(top_left[1], top_right[1]) * inv_scale)
            x_max = int(max(top_right[0], bottom_right[0]) * inv_scale)
            y_max = int(max(bottom_left[1], bottom_right[1]) * inv_scale)
        else:
            x_min = min(top_left[0], bottom_left[0])
            y_min = min(top_left[1], top_right[1])
            x_max = max(top_right[0], bottom_right[0])
            y_max = max(bottom_left[1], bottom_right[1])
        
        # Validate bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(original_shape[1] - 1, x_max)
        y_max = min(original_shape[0] - 1, y_max)
        
        if x_max > x_min and y_max > y_min:
            roi_coords = (x_min, y_min, x_max, y_max)
            
            # Fine-tune and warp HMI region from ORIGINAL image
            warped_roi, refined_coords = fine_tune_hmi_screen(image, roi_coords)
            
            processing_time = time.time() - start_time
            print(f"[OK] HMI extracted: {warped_roi.shape[1]}x{warped_roi.shape[0]} in {processing_time:.2f}s")
            return warped_roi, processing_time
        
        return None, time.time() - start_time
        
    except Exception as e:
        print(f"[ERROR] HMI detection failed: {e}")
        return None, time.time() - start_time


# ============================================================
# OCR FUNCTIONS
# ============================================================

def read_image_with_paddleocr(image_input, max_retries=2):
    """
    Read text from image using PaddleOCR with retry mechanism and auto-recovery
    
    Args:
        image_input: Can be file path or numpy array (OpenCV image)
        max_retries: Maximum number of retry attempts if OCR fails
    
    Returns:
        tuple: (results, img_width, img_height)
    """
    global _paddle_ocr_fail_count
    
    ocr = get_paddleocr_instance()
    
    if ocr is None:
        print("[ERROR] PaddleOCR not available")
        return None, 0, 0
    
    start_time = time.time()
    temp_path = None
    
    # Generate unique temp file name to avoid conflicts in concurrent requests
    unique_id = str(uuid.uuid4())[:8]
    temp_path = f"_temp_paddle_ocr_{unique_id}.jpg"
    
    try:
        # IMPORTANT: Do NOT resize the input image - this affects OCR accuracy
        # The image must be processed at its original size to maintain accuracy
        
        if isinstance(image_input, np.ndarray):
            # Use image as-is without any resizing
            img_height, img_width = image_input.shape[:2]
            # Save to temp file for PaddleOCR
            cv2.imwrite(temp_path, image_input)
        else:
            img = cv2.imread(image_input)
            if img is not None:
                # Use image as-is without any resizing
                img_height, img_width = img.shape[:2]
                # Save to temp and read to keep current predict flow
                cv2.imwrite(temp_path, img)
            else:
                img_height, img_width = 1, 1
                # If image read failed, try direct prediction
                if hasattr(ocr, 'predict'):
                    results = ocr.predict(image_input)
                else:
                    results = ocr.ocr(image_input, cls=False)
                return results, img_width, img_height
        
        # Auto-detect PaddleOCR version and use appropriate method
        # PaddleOCR 2.x: ocr() method
        # PaddleOCR 3.x: predict() method
        use_predict = hasattr(ocr, 'predict')
        use_ocr = hasattr(ocr, 'ocr')
        
        if not use_predict and not use_ocr:
            print("[ERROR] PaddleOCR instance has neither predict() nor ocr() method")
            return None, img_width, img_height
        
        # Retry mechanism for OCR prediction with auto-recovery
        results = None
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if use_predict:
                    # PaddleOCR 3.x
                    results = ocr.predict(temp_path)
                else:
                    # PaddleOCR 2.x
                    results = ocr.ocr(temp_path, cls=False)
                
                # If successful, reset fail count and break out of retry loop
                if results is not None:
                    with _paddle_ocr_lock:
                        _paddle_ocr_fail_count = 0
                    break
                    
            except Exception as e:
                last_error = e
                with _paddle_ocr_lock:
                    _paddle_ocr_fail_count += 1
                
                if attempt < max_retries:
                    wait_time = 0.1 * (attempt + 1)  # Exponential backoff
                    print(f"[WARNING] PaddleOCR attempt {attempt + 1} failed: {e}")
                    print(f"[*] Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] PaddleOCR failed after {max_retries + 1} attempts: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # If we've failed too many times, reset the instance
                    should_reset = False
                    with _paddle_ocr_lock:
                        if _paddle_ocr_fail_count >= _paddle_ocr_max_fails:
                            should_reset = True
                    
                    if should_reset:
                        print(f"[WARNING] PaddleOCR has failed {_paddle_ocr_fail_count} times, resetting instance...")
                        reset_paddleocr_instance()
                        # Try one more time with new instance
                        try:
                            ocr = get_paddleocr_instance()
                            if ocr is not None:
                                print("[*] Retrying with reset PaddleOCR instance...")
                                if hasattr(ocr, 'predict'):
                                    results = ocr.predict(temp_path)
                                else:
                                    results = ocr.ocr(temp_path, cls=False)
                                if results is not None:
                                    with _paddle_ocr_lock:
                                        _paddle_ocr_fail_count = 0
                                    print("[OK] PaddleOCR recovered after reset")
                        except Exception as recovery_error:
                            print(f"[ERROR] PaddleOCR recovery failed: {recovery_error}")
        
        elapsed = time.time() - start_time
        
        # Validate results
        if results is None:
            print(f"[WARNING] PaddleOCR returned None results after {max_retries + 1} attempts")
            if last_error:
                print(f"[ERROR] Last error: {last_error}")
            return None, img_width, img_height
        
        # Check if results is empty or invalid
        if isinstance(results, list) and len(results) == 0:
            print(f"[WARNING] PaddleOCR returned empty results")
        elif not isinstance(results, list):
            print(f"[WARNING] PaddleOCR returned unexpected result type: {type(results)}")
        
        print(f"[OK] PaddleOCR completed in {elapsed:.2f}s")
        
        return results, img_width, img_height
        
    except Exception as e:
        print(f"[ERROR] PaddleOCR failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0
        
    finally:
        # Always clean up temp file, even if there was an error
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                print(f"[WARNING] Failed to cleanup temp file {temp_path}: {cleanup_error}")


def read_image_with_paddleocr_batch(images, max_workers=4):
    """
    Process multiple images in parallel using PaddleOCR
    
    Args:
        images: List of numpy arrays (OpenCV images)
        max_workers: Number of parallel workers
    
    Returns:
        List of (results, img_width, img_height) tuples
    """
    if not images:
        return []
    
    if len(images) == 1:
        return [read_image_with_paddleocr(images[0])]
    
    results = []
    
    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_image_with_paddleocr, img) for img in images]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Batch OCR error: {e}")
                results.append((None, 0, 0))
    
    return results


def get_ocr_performance_stats():
    """Get OCR performance statistics"""
    with _paddle_ocr_lock:
        return _perf_stats.copy()


def extract_ocr_data(results):
    """Extract data from PaddleOCR results (supports both 2.x and 3.x formats)"""
    all_data = []
    
    if not results:
        print("[DEBUG] extract_ocr_data: results is None or empty", flush=True)
        return all_data
    
    if not isinstance(results, list):
        print(f"[DEBUG] extract_ocr_data: results is not a list, type: {type(results)}", flush=True)
        return all_data
    
    # Check if this is PaddleOCR 2.x format: results[0] is a list of [box, (text, score)] items
    # Format: [[[box, (text, score)], [box, (text, score)], ...]]
    if (len(results) > 0 and isinstance(results[0], list) and 
        len(results[0]) > 0 and
        isinstance(results[0][0], list) and
        len(results[0][0]) == 2):
        # PaddleOCR 2.x format: results[0] contains list of [box, (text, score)] items
        print("[DEBUG] extract_ocr_data: Detected PaddleOCR 2.x format (per-item list)", flush=True)
        try:
            items = results[0]  # List of [box, (text, score)] items
            print(f"[DEBUG] extract_ocr_data: Processing {len(items)} items from results[0]", flush=True)
            
            for idx, item in enumerate(items):
                try:
                    if not isinstance(item, list) or len(item) < 2:
                        continue
                    
                    polygon = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_data = item[1]  # (text, score) or [text, score]
                    
                    if isinstance(text_data, (list, tuple)) and len(text_data) >= 2:
                        text = str(text_data[0])
                        score = float(text_data[1]) if len(text_data) > 1 else 0.0
                    else:
                        text = str(text_data) if text_data else ''
                        score = 0.0
                    
                    data = {
                        'text': text,
                        'confidence': score,
                        'bbox': polygon
                    }
                    all_data.append(data)
                    
                    # Debug: Show first item
                    if idx == 0:
                        print(f"[DEBUG] extract_ocr_data: First item - text='{text[:30]}', score={score:.2f}, bbox_len={len(polygon) if polygon else 0}", flush=True)
                except Exception as e:
                    print(f"[WARNING] extract_ocr_data: Error processing item[{idx}]: {e}", flush=True)
                    continue
        except Exception as e:
            print(f"[WARNING] extract_ocr_data: Error processing PaddleOCR 2.x format: {e}", flush=True)
            import traceback
            traceback.print_exc()
    # Check if this is PaddleOCR 2.x old format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, score)], ...]
    elif len(results) > 0 and isinstance(results[0], list) and len(results[0]) == 2:
        # PaddleOCR 2.x old format (per-item format)
        print("[DEBUG] extract_ocr_data: Detected PaddleOCR 2.x old format (per-item)", flush=True)
        for idx, item in enumerate(results):
            try:
                if len(item) == 2:
                    polygon = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_data = item[1]  # (text, score) or [text, score]
                    
                    if isinstance(text_data, (list, tuple)) and len(text_data) >= 2:
                        text = str(text_data[0])
                        score = float(text_data[1]) if len(text_data) > 1 else 0.0
                    else:
                        text = str(text_data) if text_data else ''
                        score = 0.0
                    
                    data = {
                        'text': text,
                        'confidence': score,
                        'bbox': polygon
                    }
                    all_data.append(data)
                    
                    # Debug: Show first item
                    if idx == 0:
                        print(f"[DEBUG] extract_ocr_data: First item - text='{text[:30]}', score={score:.2f}, bbox_len={len(polygon) if polygon else 0}", flush=True)
            except Exception as e:
                print(f"[WARNING] extract_ocr_data: Error processing item[{idx}]: {e}", flush=True)
                continue
    else:
        # PaddleOCR 3.x format: List of PredictResult objects with .json attribute
        print("[DEBUG] extract_ocr_data: Detected PaddleOCR 3.x format", flush=True)
        for idx, result in enumerate(results):
            try:
                if hasattr(result, 'json') and result.json:
                    json_data = result.json
                    res = json_data.get('res', json_data)
                    
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    # PaddleOCR 3.x: dt_polys is detection polygons (text box positions)
                    # rec_polys might be recognition polygons (after recognition, may be empty)
                    # Use dt_polys first as it's more reliable for bounding boxes
                    dt_polys = res.get('dt_polys', [])
                    rec_polys = res.get('rec_polys', [])
                    
                    # Choose which polygons to use
                    # dt_polys should match texts count (detection -> recognition pipeline)
                    if len(dt_polys) == len(texts) and len(dt_polys) > 0:
                        polys = dt_polys
                    elif len(rec_polys) == len(texts) and len(rec_polys) > 0:
                        polys = rec_polys
                    elif len(dt_polys) > 0:
                        polys = dt_polys
                    elif len(rec_polys) > 0:
                        polys = rec_polys
                    else:
                        polys = []
                    
                    if not texts:
                        continue
                    
                    # Debug: Show polygon info
                    if idx == 0:
                        print(f"[DEBUG] extract_ocr_data: texts={len(texts)}, dt_polys={len(dt_polys)}, rec_polys={len(rec_polys)}, using polys={len(polys)}", flush=True)
                        if len(polys) > 0:
                            print(f"[DEBUG] extract_ocr_data: Sample polygon format: type={type(polys[0])}, value={polys[0] if len(str(polys[0])) < 100 else str(polys[0])[:100]}", flush=True)
                    
                    # Ensure polys length matches texts length
                    if len(polys) != len(texts):
                        print(f"[WARNING] extract_ocr_data: Mismatch - texts={len(texts)}, polys={len(polys)}", flush=True)
                        # Use minimum length to avoid index errors
                        max_len = min(len(texts), len(polys)) if polys else len(texts)
                    else:
                        max_len = len(texts)
                    
                    for i in range(max_len):
                        bbox = polys[i] if i < len(polys) else []
                        # Debug: Show first bbox
                        if idx == 0 and i == 0 and bbox:
                            print(f"[DEBUG] extract_ocr_data: First bbox: type={type(bbox)}, len={len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'}, value={bbox[:2] if isinstance(bbox, (list, tuple)) and len(bbox) > 0 else bbox}", flush=True)
                        
                        data = {
                            'text': texts[i] if i < len(texts) else '',
                            'confidence': scores[i] if i < len(scores) else 0.0,
                            'bbox': bbox
                        }
                        all_data.append(data)
                else:
                    print(f"[DEBUG] extract_ocr_data: result[{idx}] has no json attribute or json is None", flush=True)
            except Exception as e:
                print(f"[WARNING] extract_ocr_data: Error processing result[{idx}]: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
    
    print(f"[DEBUG] extract_ocr_data: Extracted {len(all_data)} text items from {len(results)} results", flush=True)
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
    
    # If machine_code is provided, get machine_type from machine_code to improve matching
    # This handles cases where area="F4" but machine_type="F41" or "F42"
    target_machine_types = None
    if selected_machine:
        try:
            from .config_manager import get_machine_type
            machine_type_from_code = get_machine_type(selected_machine)
            if machine_type_from_code:
                target_machine_types = [machine_type_from_code]
                if debug:
                    print(f"   [DEBUG] Machine type from machine_code: {machine_type_from_code}")
        except Exception as e:
            if debug:
                print(f"   [DEBUG] Could not get machine_type from machine_code: {e}")
    
    # If area is provided but no machine_type found, try to expand area to machine_types
    # Example: area="F4" -> search for "F41", "F42"
    if selected_area and not target_machine_types:
        # Check if area matches any machine_type directly
        if selected_area in roi_info['machines']:
            target_machine_types = [selected_area]
        else:
            # Try to find machine_types that start with the area code
            # Example: "F4" -> ["F41", "F42"]
            target_machine_types = [mt for mt in roi_info['machines'].keys() if mt.startswith(selected_area)]
            if not target_machine_types:
                # Fallback: use area as-is
                target_machine_types = [selected_area]
    
    if debug:
        print(f"\n   [DEBUG] OCR detected {len(ocr_texts)} text items")
        if selected_area and selected_machine:
            print(f"   [DEBUG] Filter by: area={selected_area}, machine={selected_machine}")
        if target_machine_types:
            print(f"   [DEBUG] Target machine_types: {target_machine_types}")
    
    best_match = None
    best_match_count = 0
    best_match_percentage = 0
    all_matches = []
    
    # Structure: machines > machine_type (F1) > machine_code (IE-F1-CWA01) > screens > screen_name > sub_pages
    for machine_type, machine_type_data in roi_info['machines'].items():
        # Filter by machine_type (use target_machine_types if available, otherwise use selected_area)
        if target_machine_types and machine_type not in target_machine_types:
            continue
        elif selected_area and not target_machine_types and machine_type != selected_area:
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
                        
                        if debug:
                            print(f"   [DEBUG] Checking screen: {machine_type}/{machine_code}/{screen_name}/sub-page {sub_page}")
                            print(f"   [DEBUG] Special_rois ({len(special_rois)}): {special_rois[:5]}...")  # Show first 5
                            print(f"   [DEBUG] OCR texts sample ({min(10, len(ocr_texts))}): {ocr_texts[:10]}")  # Show first 10
                        
                        for special_roi in special_rois:
                            special_roi_normalized = normalize_text(special_roi)
                            
                            for ocr_text in ocr_texts:
                                if fuzzy_match(special_roi_normalized, ocr_text):
                                    match_count += 1
                                    matched_rois.append(special_roi)
                                    if debug:
                                        print(f"   [DEBUG] ✓ Matched: '{special_roi}' <-> '{ocr_text}'")
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
    
    try:
        # Handle different polygon formats
        # Format 1: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Format 2: numpy array
        # Format 3: list of tuples
        
        # Convert to list if needed
        if hasattr(polygon, 'tolist'):
            polygon = polygon.tolist()
        
        # Extract x and y coordinates
        xs = []
        ys = []
        for p in polygon:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            else:
                # Invalid format
                return None
        
        if not xs or not ys:
            return None
        
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        
        # Validate coordinates
        if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
            return None
        
        norm_x1 = x_min / img_width
        norm_y1 = y_min / img_height
        norm_x2 = x_max / img_width
        norm_y2 = y_max / img_height
        
        return [norm_x1, norm_y1, norm_x2, norm_y2]
    except Exception as e:
        print(f"[WARNING] polygon_to_normalized_bbox: Error converting polygon: {e}, polygon type: {type(polygon)}", flush=True)
        return None


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
    Handles cases where one OCR text matches multiple ROIs (e.g., "Complete Complete Complete" spanning multiple columns)
    Returns: list of filtered OCR results with matching ROI info
    """
    if not sub_page_data:
        print(f"[DEBUG] filter_ocr_by_roi: sub_page_data is None or empty", flush=True)
        return []
    
    # Support both "Rois" (capital) and "rois" (lowercase)
    rois = sub_page_data.get('Rois', sub_page_data.get('rois', []))
    if not rois:
        print(f"[DEBUG] filter_ocr_by_roi: No ROIs found in sub_page_data", flush=True)
        return []
    
    print(f"[DEBUG] filter_ocr_by_roi: Processing {len(ocr_data)} OCR items, {len(rois)} ROIs, image size: {img_width}x{img_height}", flush=True)
    
    filtered_results = []
    no_bbox_count = 0
    no_normalized_count = 0
    no_match_count = 0
    
    for ocr_item in ocr_data:
        polygon = ocr_item.get('bbox', [])
        if not polygon:
            no_bbox_count += 1
            if no_bbox_count <= 3:
                print(f"[DEBUG] filter_ocr_by_roi: OCR item has no bbox: text='{ocr_item.get('text', '')[:30]}'", flush=True)
            continue
        
        # Debug: Show first polygon format
        if no_bbox_count == 0 and no_normalized_count == 0 and no_match_count == 0:
            print(f"[DEBUG] filter_ocr_by_roi: First polygon: type={type(polygon)}, len={len(polygon) if isinstance(polygon, (list, tuple)) else 'N/A'}, sample={polygon[:1] if isinstance(polygon, (list, tuple)) and len(polygon) > 0 else polygon}", flush=True)
        
        ocr_bbox = polygon_to_normalized_bbox(polygon, img_width, img_height)
        if not ocr_bbox:
            no_normalized_count += 1
            if no_normalized_count <= 3:
                print(f"[DEBUG] filter_ocr_by_roi: Failed to normalize bbox: polygon type={type(polygon)}, len={len(polygon) if isinstance(polygon, (list, tuple)) else 'N/A'}, value={str(polygon)[:100]}", flush=True)
            continue
        
        # Find ALL ROIs that match this OCR text (IoU >= threshold)
        matching_rois = []
        for roi in rois:
            roi_coords = roi.get('coordinates', [])
            if len(roi_coords) != 4:
                continue
            
            iou = calculate_iou(ocr_bbox, roi_coords)
            
            if iou >= IOU_THRESHOLD:
                matching_rois.append({
                    'roi': roi,
                    'roi_name': roi.get('name', 'Unknown'),
                    'roi_coords': roi_coords,
                    'iou': iou
                })
        
        if not matching_rois:
            no_match_count += 1
            if no_match_count <= 3:
                # Find best IoU for debugging
                best_iou = 0.0
                for roi in rois:
                    roi_coords = roi.get('coordinates', [])
                    if len(roi_coords) == 4:
                        iou = calculate_iou(ocr_bbox, roi_coords)
                        if iou > best_iou:
                            best_iou = iou
                print(f"[DEBUG] filter_ocr_by_roi: No match for text='{ocr_item.get('text', '')[:30]}', best_iou={best_iou:.4f}, ocr_bbox={ocr_bbox}", flush=True)
            continue
        
        # If only one ROI matches, use it directly
        if len(matching_rois) == 1:
            match = matching_rois[0]
            allowed_values = match['roi'].get('allowed_values', [])
            filtered_results.append({
                'text': ocr_item['text'],
                'confidence': ocr_item['confidence'],
                'bbox': polygon,
                'normalized_bbox': ocr_bbox,
                'matched_roi': match['roi_name'],
                'roi_coords': match['roi_coords'],
                'allowed_values': allowed_values,
                'iou': match['iou']
            })
        else:
            # Multiple ROIs match - need to split text or assign to each ROI
            # This happens when OCR reads a long text spanning multiple columns
            ocr_text = ocr_item['text']
            words = ocr_text.split()
            
            # Check if all matching ROIs have the same allowed_values
            # If yes, we can assign the same text to all (after processing)
            all_same_allowed = True
            first_allowed = matching_rois[0]['roi'].get('allowed_values', [])
            for match in matching_rois[1:]:
                if match['roi'].get('allowed_values', []) != first_allowed:
                    all_same_allowed = False
                    break
            
            # Strategy 1: If text contains repeated words and all ROIs have same allowed_values
            # Split text into words and assign one word per ROI (left to right)
            if len(set(word.upper() for word in words)) == 1 and all_same_allowed and first_allowed:
                # All words are the same (e.g., "Complete Complete Complete")
                # Assign one word to each ROI
                for i, match in enumerate(matching_rois):
                    if i < len(words):
                        # Use the word at position i
                        word_text = words[i]
                    else:
                        # If more ROIs than words, use the first word
                        word_text = words[0] if words else ocr_text
                    
                    allowed_values = match['roi'].get('allowed_values', [])
                    filtered_results.append({
                        'text': word_text,
                        'confidence': ocr_item['confidence'],
                        'bbox': polygon,
                        'normalized_bbox': ocr_bbox,
                        'matched_roi': match['roi_name'],
                        'roi_coords': match['roi_coords'],
                        'allowed_values': allowed_values,
                        'iou': match['iou']
                    })
            else:
                # Strategy 2: Text contains different words (e.g., "Complete Incomplete Complete")
                # Try to match words with ROIs based on position
                # Sort ROIs by X coordinate (left to right)
                sorted_matches = sorted(matching_rois, key=lambda m: m['roi_coords'][0])
                
                # Assign words to ROIs (left to right)
                for i, match in enumerate(sorted_matches):
                    if i < len(words):
                        word_text = words[i]
                    else:
                        # If more ROIs than words, use the last word
                        word_text = words[-1] if words else ocr_text
                    
                    allowed_values = match['roi'].get('allowed_values', [])
                    filtered_results.append({
                        'text': word_text,
                        'confidence': ocr_item['confidence'],
                        'bbox': polygon,
                        'normalized_bbox': ocr_bbox,
                        'matched_roi': match['roi_name'],
                        'roi_coords': match['roi_coords'],
                        'allowed_values': allowed_values,
                        'iou': match['iou']
                    })
    
    print(f"[DEBUG] filter_ocr_by_roi: Results - matched: {len(filtered_results)}, no_bbox: {no_bbox_count}, no_normalized: {no_normalized_count}, no_match: {no_match_count}", flush=True)
    
    return filtered_results


def filter_ocr_by_roi_parallel(ocr_data, sub_page_data, img_width, img_height, max_workers=4):
    """
    Filter OCR results in parallel for large datasets
    Use this when ocr_data has more than 50 items
    """
    if len(ocr_data) < 50:
        return filter_ocr_by_roi(ocr_data, sub_page_data, img_width, img_height)
    
    # Split data for parallel processing
    chunk_size = max(10, len(ocr_data) // max_workers)
    chunks = [ocr_data[i:i + chunk_size] for i in range(0, len(ocr_data), chunk_size)]
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(filter_ocr_by_roi, chunk, sub_page_data, img_width, img_height)
            for chunk in chunks
        ]
        
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"[ERROR] Parallel filter error: {e}")
    
    return all_results


# ============================================================
# POST-PROCESSING FUNCTIONS
# ============================================================

def match_text_with_allowed_values(text, allowed_values):
    """
    Match OCR text with allowed_values list
    Handles cases like:
    - "Complete Complete Complete Complete Complete" -> "Complete" (if in allowed_values)
    - "Incomplete" -> "Incomplete" (if in allowed_values)
    - "ON" -> "ON" (if in allowed_values)
    - "OFF" -> "OFF" (if in allowed_values)
    
    Args:
        text: OCR text (may contain repeated words)
        allowed_values: List of allowed values (e.g., ["Complete", "Incomplete"])
    
    Returns:
        str: Matched value from allowed_values, or processed text if no match
    """
    if not text or not allowed_values or len(allowed_values) == 0:
        return text
    
    text_clean = text.strip()
    if not text_clean:
        return text
    
    # Normalize text: remove extra spaces, convert to uppercase for comparison
    text_normalized = ' '.join(text_clean.split()).upper()
    
    # First, try exact match (case-insensitive)
    for allowed_value in allowed_values:
        if text_normalized == allowed_value.upper().strip():
            print(f"[DEBUG] Exact match found: '{text}' -> '{allowed_value}'")
            return allowed_value
    
    # If text contains repeated words (e.g., "Complete Complete Complete")
    # Split by spaces and check each word
    words = text_clean.split()
    
    # If all words are the same, use that word
    if len(set(word.upper() for word in words)) == 1:
        single_word = words[0]
        # Check if this word matches any allowed_value
        for allowed_value in allowed_values:
            if single_word.upper() == allowed_value.upper().strip():
                print(f"[DEBUG] Repeated word match: '{text}' -> '{allowed_value}'")
                return allowed_value
    
    # Try to find allowed_value that appears in the text (whole word match)
    for allowed_value in allowed_values:
        allowed_normalized = allowed_value.upper().strip()
        # Check if allowed_value appears as a whole word in text
        pattern = r'\b' + re.escape(allowed_normalized) + r'\b'
        if re.search(pattern, text_normalized, re.IGNORECASE):
            print(f"[DEBUG] Whole word match found: '{text}' -> '{allowed_value}'")
            return allowed_value
    
    # Try fuzzy matching: find the closest allowed_value
    # Simple approach: check if any word in text matches any allowed_value
    for word in words:
        word_upper = word.upper()
        for allowed_value in allowed_values:
            if word_upper == allowed_value.upper().strip():
                print(f"[DEBUG] Word match found: '{text}' -> '{allowed_value}'")
                return allowed_value
    
    # If no match found, return the first word (most likely the actual value)
    # This handles cases where OCR reads multiple times but we want just one
    if len(words) > 0:
        first_word = words[0]
        # Check if first word is close to any allowed_value
        for allowed_value in allowed_values:
            if first_word.upper() == allowed_value.upper().strip():
                print(f"[DEBUG] First word match found: '{text}' -> '{allowed_value}'")
                return allowed_value
    
    # Return original text if no match
    print(f"[DEBUG] No match found for '{text}' with allowed_values {allowed_values}, returning original")
    return text


def extract_number_from_text(text):
    """
    Extract numerical value from text string
    Handles cases like:
    - "Injection maximum pre 94.2 MPa" -> "94.2"
    - "Injection maximum spe 26.8 mm/s" -> "26.8"
    - "-0.0 rpm" -> "-0.0"
    - "0.1 mm/s" -> "0.1"
    - "94.2 MPa" -> "94.2"
    - "3.7 Sec." -> "3.7"
    - "29.4 mm" -> "29.4"
    - "0 %" -> "0"
    - "18.0" -> "18.0" (already a number)
    - "08:51:28" -> "08:51:28" (time format, keep as-is)
    - "08:51:28Sec" -> "08:51:28" (time format with suffix, remove suffix)
    - "16:18:56" -> "16:18:56" (time format, keep as-is)
    - "OFF" -> "OFF" (no number, return as-is)
    
    Args:
        text: Input text string
    
    Returns:
        str: Extracted number as string, or original text if no number found or text is already clean
    """
    if not text:
        return text
    
    text_clean = text.strip()
    
    # Check if text is in time format hh:mm:ss (with optional suffix like "Sec", "Sec.")
    # Pattern: hh:mm:ss or hh:mm:ssSec or hh:mm:ss Sec. etc.
    # First check if text already is pure hh:mm:ss format
    if re.match(r'^\d{1,2}:\d{2}:\d{2}$', text_clean):
        return text_clean
    
    # Then check if text has hh:mm:ss with suffix (Sec, Sec., etc.)
    time_pattern = r'^(\d{1,2}:\d{2}:\d{2})(?:Sec|Sec\.|Sec\s*|\.)?$'
    time_match = re.match(time_pattern, text_clean, re.IGNORECASE)
    if time_match:
        # Extract the time part (hh:mm:ss) and return it
        return time_match.group(1)
    
    # Also check if text contains hh:mm:ss somewhere (e.g., "text 08:51:28 Sec")
    time_in_text_pattern = r'(\d{1,2}:\d{2}:\d{2})'
    time_in_text_match = re.search(time_in_text_pattern, text_clean)
    if time_in_text_match:
        # Extract the time part and check if there's a suffix after it
        time_part = time_in_text_match.group(1)
        time_end_pos = time_in_text_match.end()
        remaining_text = text_clean[time_end_pos:].strip()
        # If remaining is just "Sec", "Sec.", or empty, return time part
        if not remaining_text or re.match(r'^(Sec|Sec\.|Sec\s*|\.)$', remaining_text, re.IGNORECASE):
            return time_part
    
    # If text is already just a number (with optional sign and decimal), return as-is
    # This handles cases like "18.0", "-0.0", "0", "94.2"
    if re.match(r'^-?\d+\.?\d*$', text_clean):
        return text_clean
    
    # Pattern to match signed decimal numbers (including negative numbers)
    # Matches: -0.0, 94.2, 26.8, 0.1, 3.7, etc.
    # Pattern: optional minus sign, digits, optional decimal point and digits
    number_pattern = r'-?\d+\.?\d*'
    
    # Find all number matches
    matches = re.findall(number_pattern, text)
    
    if not matches:
        # No number found, return original text (e.g., "OFF", "ON")
        return text
    
    if len(matches) == 1:
        # Only one number found, return it
        return matches[0]
    
    # Multiple numbers found - need to determine which is the value
    # Strategy: Usually the value appears after the label and before/with the unit
    # Common patterns:
    # - "label 94.2 unit" -> value is in the middle
    # - "94.2 unit" -> value is first
    # - "label value1 value2" -> last value is usually the current value
    
    # Try to find number that appears after common separators or at the end
    # Usually the last number in the string is the value (current reading)
    # But check position to be more accurate
    
    # Split text by spaces to analyze structure
    words = text.split()
    
    # Look for number that appears after text words (label) and before/with unit
    # Units typically: MPa, mm/s, rpm, %, Sec., mm, bar, etc.
    unit_pattern = r'(MPa|mm/s|rpm|%|Sec\.|mm|bar|RPM|mm/s)'
    
    # Find position of unit
    unit_match = re.search(unit_pattern, text, re.IGNORECASE)
    
    if unit_match:
        # Unit found - value is usually just before the unit
        unit_pos = unit_match.start()
        # Find number closest to unit position
        best_match = None
        best_distance = float('inf')
        
        for match in matches:
            match_pos = text.find(match)
            # Prefer number that appears before unit
            if match_pos < unit_pos:
                distance = unit_pos - match_pos
                if distance < best_distance:
                    best_distance = distance
                    best_match = match
        
        if best_match:
            return best_match
    
    # If no unit found or no number before unit, return the last number
    # (most likely the current value)
    return matches[-1]


def split_merged_numbers_by_decimal_places(text, roi_decimal_configs):
    """
    Split merged numbers (e.g., "45.00120.00") into separate numbers based on decimal_places config
    
    Strategy: Parse directly from text string using decimal_places pattern
    - For "45.00120.00" with decimal_places=2:
      - Pattern: digits.dd (where dd = 2 decimal places)
      - Parse: "45.00" (4 digits + dot + 2 decimals) and "120.00" (3 digits + dot + 2 decimals)
      - Assign from left to right based on ROI coordinates
    
    Args:
        text: Merged number string (e.g., "45.00120.00")
        roi_decimal_configs: List of dicts with keys: {'roi_name', 'decimal_places', 'roi_coords'}
                           Sorted by X coordinate (left to right)
    
    Returns:
        List of split numbers matching each ROI's decimal_places, or None if cannot split
    """
    if not text or not roi_decimal_configs or len(roi_decimal_configs) < 2:
        return None
    
    print(f"[DEBUG] split_merged_numbers: Input text='{text}', {len(roi_decimal_configs)} ROIs")
    
    # Clean text - keep only digits, dots, and minus signs
    text_clean = re.sub(r'[^\d.-]', '', str(text))
    if not text_clean:
        print(f"[DEBUG] split_merged_numbers: No digits found in '{text}'")
        return None
    
    is_negative = text_clean.startswith('-')
    if is_negative:
        text_clean = text_clean[1:]
    
    print(f"[DEBUG] split_merged_numbers: Cleaned text='{text_clean}'")
    
    # Strategy: Parse numbers directly from text using decimal_places pattern
    # For "45.00120.00" with decimal_places=2:
    # - Pattern: \d+\.\d{2} (digits + dot + exactly 2 decimals)
    # - Match: "45.00" and "120.00"
    
    split_numbers = []
    remaining_text = text_clean
    
    for i, roi_config in enumerate(roi_decimal_configs):
        decimal_places = roi_config.get('decimal_places')
        if decimal_places is None:
            print(f"[DEBUG] split_merged_numbers: ROI {i+1} has no decimal_places config")
            return None
        
        # Build regex pattern for this ROI's format
        # Pattern: digits + dot + exactly 'decimal_places' decimal digits
        if decimal_places == 0:
            # Integer: just digits (no dot)
            pattern = r'^(\d+)'
        else:
            # Decimal: digits.dd (where dd = decimal_places)
            pattern = rf'^(\d+\.\d{{{decimal_places}}})'
        
        # Try to match pattern from start of remaining_text
        match = re.match(pattern, remaining_text)
        if match:
            number_str = match.group(1)
            remaining_text = remaining_text[len(number_str):]
            
            # Validate number
            try:
                num_value = float(number_str)
                if -1000 <= num_value <= 10000:
                    split_numbers.append(number_str)
                    print(f"[DEBUG] ROI {i+1}/{len(roi_decimal_configs)}: Parsed '{number_str}' (value={num_value}), remaining='{remaining_text}'")
                else:
                    print(f"[DEBUG] ROI {i+1}: Number '{number_str}' out of range")
                    return None
            except:
                print(f"[DEBUG] ROI {i+1}: Failed to parse '{number_str}' as number")
                return None
        else:
            # Pattern not found - try to extract digits and format manually
            # This handles cases where dots are missing (e.g., "450012000")
            print(f"[DEBUG] ROI {i+1}: Pattern not matched, trying manual extraction...")
            
            # Extract digits from remaining text
            digits_match = re.match(r'^(\d+)', remaining_text)
            if not digits_match:
                print(f"[DEBUG] ROI {i+1}: No digits found in '{remaining_text}'")
                return None
            
            digits = digits_match.group(1)
            remaining_text = remaining_text[len(digits):]
            
            # Format according to decimal_places
            if decimal_places == 0:
                number_str = digits
            else:
                # Insert dot before last 'decimal_places' digits
                if len(digits) > decimal_places:
                    insert_pos = len(digits) - decimal_places
                    number_str = f"{digits[:insert_pos]}.{digits[insert_pos:]}"
                else:
                    number_str = f"0.{digits.zfill(decimal_places)}"
            
            # Validate
            try:
                num_value = float(number_str)
                if -1000 <= num_value <= 10000:
                    split_numbers.append(number_str)
                    print(f"[DEBUG] ROI {i+1}/{len(roi_decimal_configs)}: Extracted '{number_str}' (value={num_value}), remaining='{remaining_text}'")
                else:
                    print(f"[DEBUG] ROI {i+1}: Number '{number_str}' out of range")
                    return None
            except:
                print(f"[DEBUG] ROI {i+1}: Failed to parse '{number_str}' as number")
                return None
    
    # Check if we successfully parsed all numbers
    if len(split_numbers) == len(roi_decimal_configs):
        # Apply negative sign to first number if needed
        if is_negative:
            split_numbers[0] = '-' + split_numbers[0]
        print(f"[DEBUG] split_merged_numbers: Successfully split '{text}' into {split_numbers}")
        return split_numbers
    
    print(f"[DEBUG] split_merged_numbers: Failed to split - only parsed {len(split_numbers)}/{len(roi_decimal_configs)} numbers")
    return None
    
    for i, roi_config in enumerate(roi_decimal_configs):
        decimal_places = roi_config.get('decimal_places')
        if decimal_places is None:
            return None  # All ROIs must have decimal_places config
        
        # Estimate digits needed: at least 1 integer digit + decimal_places
        # But we'll try to be smart: look for common patterns
        # For "45.00120.00" with decimal_places=2:
        # - First number: "45.00" = 2 integer + 2 decimal = 4 digits
        # - Second number: "120.00" = 3 integer + 2 decimal = 5 digits
        
        # Try different lengths: start with minimum, increase if needed
        min_digits = max(1, decimal_places + 1)
        max_digits = min(8, len(all_digits) - digit_index)  # Don't exceed remaining digits
        
        best_match = None
        best_length = None
        
        # Try different digit lengths
        # Strategy: For merged numbers like "45.00120.00" → "450012000" (9 digits)
        # With 2 ROIs, each with decimal_places=2:
        # - ROI 1 (left): "45.00" = 4 digits (4500)
        # - ROI 2 (right): "120.00" = 5 digits (12000)
        # Total: 4 + 5 = 9 digits ✓
        # 
        # We want to find the length that:
        # 1. Leaves EXACTLY enough digits for remaining ROIs (most important)
        # 2. Produces a reasonable number value
        # 3. Prefers shorter lengths first (greedy approach from left)
        
        candidates = []
        for length in range(min_digits, max_digits + 1):
            if digit_index + length > len(all_digits):
                break
            
            number_digits = all_digits[digit_index:digit_index + length]
            
            # Format according to decimal_places
            if decimal_places == 0:
                number_str = number_digits
            else:
                # Insert dot before last 'decimal_places' digits
                if len(number_digits) > decimal_places:
                    insert_pos = len(number_digits) - decimal_places
                    number_str = f"{number_digits[:insert_pos]}.{number_digits[insert_pos:]}"
                else:
                    # Not enough digits, pad with zeros
                    number_str = f"0.{number_digits.zfill(decimal_places)}"
            
            # Validate: number should be reasonable (not too large)
            try:
                num_value = float(number_str)
                # Reasonable range for HMI values: -1000 to 10000
                if -1000 <= num_value <= 10000:
                    # Calculate remaining digits for other ROIs
                    remaining_for_others = len(all_digits) - (digit_index + length)
                    min_needed_for_others = sum(max(1, rc.get('decimal_places', 0) + 1) 
                                               for rc in roi_decimal_configs[i+1:])
                    
                    # Score calculation (simplified):
                    # 1. Highest priority: EXACTLY enough digits for others (perfect match)
                    # 2. Second priority: At least enough digits for others
                    # 3. Prefer shorter lengths (greedy from left)
                    # 4. Prefer reasonable values
                    score = 0
                    
                    if remaining_for_others == min_needed_for_others:
                        # Perfect match! This is ideal
                        score += 1000000
                    elif remaining_for_others > min_needed_for_others:
                        # More than enough, but acceptable
                        # Penalty for excess digits (prefer exact match)
                        excess = remaining_for_others - min_needed_for_others
                        score += 100000 - (excess * 100)
                    else:
                        # Not enough digits - reject this candidate
                        continue
                    
                    # Prefer shorter lengths (greedy approach - take from left first)
                    # This ensures we get "45.00" (4 digits) instead of "4500.12" (6 digits)
                    score += (100 - length) * 10  # Shorter = higher score
                    
                    # Prefer reasonable values (1-1000 range)
                    if 1 <= num_value <= 1000:
                        score += 50
                    elif num_value < 1:
                        score -= 30  # Penalty for very small numbers
                    elif num_value > 1000:
                        score -= 10  # Small penalty for large numbers
                    
                    candidates.append({
                        'number_str': number_str,
                        'length': length,
                        'value': num_value,
                        'score': score,
                        'remaining': remaining_for_others,
                        'min_needed': min_needed_for_others
                    })
            except:
                pass
        
        # Pick best candidate (highest score)
        if candidates:
            # Sort by score (descending), then by length (ascending for tie-breaking - prefer shorter)
            candidates.sort(key=lambda x: (-x['score'], x['length']))
            best_match = candidates[0]['number_str']
            best_length = candidates[0]['length']
            print(f"[DEBUG] ROI {i+1}/{len(roi_decimal_configs)}: Selected '{best_match}' (length={best_length}, value={candidates[0]['value']}, score={candidates[0]['score']}, remaining={candidates[0]['remaining']}/{candidates[0]['min_needed']} digits)")
            
            # Debug: show top 3 candidates
            if len(candidates) > 1:
                top_candidates = [(c['number_str'], f"score={c['score']}", f"len={c['length']}", f"val={c['value']}") for c in candidates[:3]]
                print(f"[DEBUG] Top 3 candidates: {top_candidates}")
        else:
            best_match = None
            best_length = None
        
        if best_match:
            split_numbers.append(('-' + best_match) if is_negative and i == 0 else best_match)
            digit_index += best_length
            is_negative = False  # Only first number can be negative
        else:
            # Can't extract number for this ROI
            print(f"[DEBUG] split_merged_numbers: Failed to extract number for ROI {i+1} '{roi_config.get('roi_name', 'Unknown')}'")
            return None
    
    # Check if we used all or most digits (allow small remainder for OCR errors)
    remaining_digits = len(all_digits) - digit_index
    if remaining_digits <= 2:  # Allow up to 2 extra digits (OCR noise)
        print(f"[DEBUG] split_merged_numbers: Successfully split '{text}' into {split_numbers}, used {digit_index}/{len(all_digits)} digits, remaining: {remaining_digits}")
        return split_numbers
    
    print(f"[DEBUG] split_merged_numbers: Failed to split '{text}' - used {digit_index}/{len(all_digits)} digits, remaining: {remaining_digits} (too many)")
    return None


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

