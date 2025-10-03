from flask import Flask, request, jsonify, send_from_directory, abort
import os
from werkzeug.utils import secure_filename
import time
import cv2
import numpy as np
import json
import re
from skimage.filters import threshold_sauvola
import fnmatch
import traceback
import random
from math import sqrt, atan2, degrees
from datetime import datetime
from PIL import Image, ImageEnhance
from difflib import SequenceMatcher
import Levenshtein
import concurrent.futures
from threading import Lock
from smart_detection_functions import auto_detect_machine_and_screen_smart
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import threading

# [*] Import GPU Accelerator và Parallel Processor modules
try:
    from gpu_accelerator import get_gpu_accelerator, is_gpu_available, get_gpu_info
    from parallel_processor import (
        get_ocr_thread_pool, get_image_thread_pool, get_roi_processor,
        parallel_map, get_system_stats, ParallelROIProcessor
    )
    OPTIMIZATION_MODULES_AVAILABLE = True
    print("[OK] GPU Accelerator và Parallel Processor modules loaded successfully")
except ImportError as e:
    OPTIMIZATION_MODULES_AVAILABLE = False
    print(f"[WARNING] Optimization modules not available: {e}")
    print("   Server will run in standard mode without GPU acceleration")

# Thêm try-except khi import EasyOCR
try:
    import easyocr
    HAS_EASYOCR = True
    # Khởi tạo đối tượng reader ở cấp độ global
    reader = easyocr.Reader(['en'], gpu=True)
    print("EasyOCR initialized successfully with GPU")
except ImportError:
    print("EasyOCR not installed. OCR functionality will be limited.")
    HAS_EASYOCR = False
    reader = None

# Khởi tạo các detector OpenCV ở cấp độ global để tối ưu hiệu suất
try:
    # SIFT detector cho image alignment
    sift_detector = cv2.SIFT_create()
    print("SIFT detector initialized successfully")
except Exception as e:
    print(f"Error initializing SIFT detector: {e}")
    sift_detector = None

try:
    # ORB detector cho feature comparison  
    orb_detector = cv2.ORB_create(nfeatures=500)
    print("ORB detector initialized successfully")
except Exception as e:
    print(f"Error initializing ORB detector: {e}")
    orb_detector = None

try:
    # BFMatcher cho ORB matching
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    print("BFMatcher initialized successfully")
except Exception as e:
    print(f"Error initializing BFMatcher: {e}")
    bf_matcher = None

try:
    # FLANN matcher cho SIFT matching
    FLANN_INDEX_KDTREE = 1
    flann_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    flann_search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
    print("FLANN matcher initialized successfully")
except Exception as e:
    print(f"Error initializing FLANN matcher: {e}")
    flann_matcher = None

# Global cache cho config data để tối ưu I/O
_roi_info_cache = None
_roi_info_cache_lock = threading.Lock()
_decimal_places_cache = None
_decimal_places_cache_lock = threading.Lock()
_machine_info_cache = None
_machine_info_cache_lock = threading.Lock()

# Thread pool để xử lý OCR song song - Enhanced với Parallel Processor
if OPTIMIZATION_MODULES_AVAILABLE:
    # Sử dụng adaptive thread pool từ parallel_processor
    _ocr_thread_pool = get_ocr_thread_pool()
    _image_thread_pool = get_image_thread_pool()
    _roi_processor = get_roi_processor()
    print(f"[OK] Enhanced thread pools initialized with GPU acceleration support")
else:
    # Fallback to standard thread pool
    _ocr_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(4, cpu_count()))
    _image_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(4, cpu_count()))
    _roi_processor = None
    print(f"[WARNING] Standard OCR Thread pool initialized with {max(4, cpu_count())} workers")

# GPU Accelerator instance
_gpu_accelerator = None
if OPTIMIZATION_MODULES_AVAILABLE and is_gpu_available():
    _gpu_accelerator = get_gpu_accelerator()
    print(f"[OK] GPU Accelerator ready: {get_gpu_info()}")

# Template image cache để tránh đọc lại ảnh template
_template_image_cache = {}
_template_cache_lock = threading.Lock()

def initialize_all_caches():
    """Khởi tạo tất cả cache ngay khi chương trình bắt đầu để tránh delay lần đầu gọi API"""
    print("\n[*] Initializing all caches at startup...")
    
    try:
        # Cache ROI info
        roi_info = get_roi_info_cached()
        print(f"[OK] ROI info cached: {len(roi_info)} items")
    except Exception as e:
        print(f"[ERROR] Error caching ROI info: {e}")
    
    try:
        # Cache decimal places config
        decimal_config = get_decimal_places_config_cached()
        print(f"[OK] Decimal places config cached: {len(decimal_config)} items")
    except Exception as e:
        print(f"[ERROR] Error caching decimal places config: {e}")
    
    try:
        # Cache machine info
        machine_info = get_machine_info_cached()
        print(f"[OK] Machine info cached: {machine_info}")
    except Exception as e:
        print(f"[ERROR] Error caching machine info: {e}")
    
    try:
        # Pre-cache common template images từ reference_images folder
        reference_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data', 'reference_images')
        if os.path.exists(reference_folder):
            template_files = [f for f in os.listdir(reference_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            cached_count = 0
            for template_file in template_files[:]:  # Cache tối đa 10 template đầu tiên
                template_path = os.path.join(reference_folder, template_file)
                if get_template_image_cached(template_path) is not None:
                    cached_count += 1
            print(f"[OK] Template images pre-cached: {cached_count}/{len(template_files)} files")
        else:
            print("ℹ️  Reference images folder not found - skipping template pre-caching")
    except Exception as e:
        print(f"[ERROR] Error pre-caching template images: {e}")
    
    print("🎯 Cache initialization completed!\n")

def get_template_image_cached(template_path):
    """Cache template images để tránh đọc lại từ disk"""
    if not template_path or not os.path.exists(template_path):
        return None
        
    with _template_cache_lock:
        if template_path not in _template_image_cache:
            try:
                template_img = cv2.imread(template_path)
                if template_img is not None:
                    _template_image_cache[template_path] = template_img
                    print(f"[OK] Template image cached: {os.path.basename(template_path)}")
                return template_img
            except Exception as e:
                print(f"[ERROR] Error caching template image: {e}")
                return None
        
        return _template_image_cache[template_path]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ROI_DATA_FOLDER'] = 'roi_data'
app.config['REFERENCE_IMAGES_FOLDER'] = 'roi_data/reference_images'  # Thư mục chứa ảnh tham chiếu
app.config['OCR_RESULTS_FOLDER'] = 'ocr_results'  # Thư mục lưu kết quả OCR

# Cấu hình thư mục lưu trữ ảnh HMI refined
HMI_REFINED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'hmi_refined')
if not os.path.exists(HMI_REFINED_FOLDER):
    os.makedirs(HMI_REFINED_FOLDER)

# Cho phép CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Route để kiểm tra server
@app.route('/')
def home():
    return jsonify({"status": "Server is running", "endpoints": ["/api/images"]}), 200

# [*] New: GPU & Performance monitoring endpoint
@app.route('/api/performance', methods=['GET'])
def get_performance_stats():
    """
    API endpoint để lấy thông tin GPU và system performance
    
    Returns:
        JSON với thông tin GPU, CPU, Memory, và Thread pools
    """
    try:
        stats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_available": False,
            "optimization_enabled": OPTIMIZATION_MODULES_AVAILABLE
        }
        
        # GPU info
        if OPTIMIZATION_MODULES_AVAILABLE and is_gpu_available():
            stats["gpu_available"] = True
            stats["gpu_info"] = get_gpu_info()
            
            if _gpu_accelerator:
                stats["gpu_memory"] = _gpu_accelerator.get_memory_info()
        
        # System stats
        if OPTIMIZATION_MODULES_AVAILABLE:
            stats["system"] = get_system_stats()
        else:
            stats["system"] = {
                "cpu_count": cpu_count(),
                "note": "Running in standard mode"
            }
        
        # EasyOCR status
        stats["ocr"] = {
            "easyocr_available": HAS_EASYOCR,
            "gpu_enabled": HAS_EASYOCR and hasattr(reader, 'gpu') and reader is not None
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route debug chi tiết
@app.route('/debug')
def debug_info():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": [method for method in rule.methods if method != 'OPTIONS' and method != 'HEAD'],
            "route": str(rule)
        })
    
    return jsonify({
        "server_info": {
            "upload_folder": app.config['UPLOAD_FOLDER'],
            "roi_data_folder": app.config['ROI_DATA_FOLDER'],
            "ocr_results_folder": app.config['OCR_RESULTS_FOLDER'],
            "hmi_refined_folder": app.config['HMI_REFINED_FOLDER'],
            "allowed_extensions": list(app.config['ALLOWED_EXTENSIONS']),
            "max_content_length": app.config['MAX_CONTENT_LENGTH']
        },
        "routes": routes,
        "environment": {
            "host": request.host,
            "remote_addr": request.remote_addr,
            "user_agent": str(request.user_agent)
        }
    })

# Cấu hình thư mục lưu trữ hình ảnh
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Cấu hình thư mục lưu trữ dữ liệu ROI
ROI_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data')
if not os.path.exists(ROI_DATA_FOLDER):
    os.makedirs(ROI_DATA_FOLDER)

# Cấu hình thư mục lưu trữ kết quả OCR
OCR_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_results')
if not os.path.exists(OCR_RESULTS_FOLDER):
    os.makedirs(OCR_RESULTS_FOLDER)

# Cấu hình thư mục lưu trữ ảnh template mẫu
REFERENCE_IMAGES_FOLDER = os.path.join(ROI_DATA_FOLDER, 'reference_images')
if not os.path.exists(REFERENCE_IMAGES_FOLDER):
    os.makedirs(REFERENCE_IMAGES_FOLDER)
    print(f"Đã tạo thư mục reference_images tại {REFERENCE_IMAGES_FOLDER}")
    print("Lưu ý: Tên file tham chiếu nên theo định dạng: template_<machine_type>_<screen_name>.png")
    print("Ví dụ: template_F1_Faults.png, template_F42_Production_Data.png")

# Đảm bảo thư mục uploads tồn tại
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROI_DATA_FOLDER'] = ROI_DATA_FOLDER
app.config['OCR_RESULTS_FOLDER'] = OCR_RESULTS_FOLDER
app.config['REFERENCE_IMAGES_FOLDER'] = REFERENCE_IMAGES_FOLDER  # Thêm cấu hình cho thư mục reference_images
app.config['HMI_REFINED_FOLDER'] = HMI_REFINED_FOLDER  # Thêm cấu hình cho thư mục HMI refined
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn kích thước file 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Cached functions để tối ưu I/O operations
def get_roi_info_cached():
    """Cache ROI info để tránh đọc file JSON nhiều lần"""
    global _roi_info_cache
    
    with _roi_info_cache_lock:
        if _roi_info_cache is None:
            try:
                roi_json_path = 'roi_data/roi_info.json'
                if not os.path.exists(roi_json_path):
                    roi_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data/roi_info.json')
                
                with open(roi_json_path, 'r', encoding='utf-8') as f:
                    _roi_info_cache = json.load(f)
                print("[OK] ROI info cached successfully")
            except Exception as e:
                print(f"[ERROR] Error caching ROI info: {e}")
                _roi_info_cache = {}
        
        return _roi_info_cache

def get_decimal_places_config_cached():
    """Cache decimal places config để tránh đọc file nhiều lần"""
    global _decimal_places_cache
    
    with _decimal_places_cache_lock:
        if _decimal_places_cache is None:
            try:
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data', 'decimal_places.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        _decimal_places_cache = json.load(f)
                else:
                    _decimal_places_cache = {}
                print("[OK] Decimal places config cached successfully")
            except Exception as e:
                print(f"[ERROR] Error caching decimal places config: {e}")
                _decimal_places_cache = {}
        
        return _decimal_places_cache

def get_machine_info_cached():
    """Cache machine info để tránh gọi hàm nặng nhiều lần"""
    global _machine_info_cache
    
    with _machine_info_cache_lock:
        if _machine_info_cache is None:
            try:
                # Đọc từ current machine screen file
                current_machine_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'current_machine_screen.json')
                if os.path.exists(current_machine_file):
                    with open(current_machine_file, 'r', encoding='utf-8') as f:
                        _machine_info_cache = json.load(f)
                else:
                    _machine_info_cache = {"machine_code": "F41", "screen_id": "Main"}
                print("[OK] Machine info cached successfully")
            except Exception as e:
                print(f"[ERROR] Error caching machine info: {e}")
                _machine_info_cache = {"machine_code": "F41", "screen_id": "Main"}
        
        return _machine_info_cache

def process_single_roi_optimized(args):
    """Xử lý OCR cho một ROI đơn lẻ - được tối ưu cho parallel processing với GPU acceleration"""
    (roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id) = args
    
    try:
        # Nếu là trường hợp đặc biệt ON/OFF, phân tích màu sắc thay vì OCR
        if is_special_on_off_case and machine_type != "F1":
            # Tách các kênh màu BGR và tính mean với GPU acceleration
            if _gpu_accelerator:
                # Sử dụng GPU để tính mean nhanh hơn
                b, g, r = cv2.split(roi_image)
                avg_blue = _gpu_accelerator.mean(b)
                avg_red = _gpu_accelerator.mean(r)
            else:
                # Fallback to CPU
                b, g, r = cv2.split(roi_image)
                avg_blue = np.mean(b)
                avg_red = np.mean(r)
            
            # Xác định kết quả dựa trên màu sắc chủ đạo
            if avg_blue > avg_red:
                best_text = "OFF"
            else:
                best_text = "ON"
            
            return {
                "roi_index": roi_name,
                "text": best_text,
                "confidence": 1.0,
                "has_text": True
            }
        
        # Kiểm tra EasyOCR có khả dụng không
        if not HAS_EASYOCR or reader is None:
            return {
                "roi_index": roi_name,
                "text": "OCR_NOT_AVAILABLE",
                "confidence": 0,
                "has_text": False,
                "original_value": ""
            }
        
        # Tiền xử lý ROI nhanh với GPU acceleration
        # Resize nếu ROI quá nhỏ
        height, width = roi_image.shape[:2]
        if height < 30 or width < 30:
            scale_factor = max(30/height, 30/width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            # Sử dụng GPU resize nếu có
            if _gpu_accelerator:
                roi_image = _gpu_accelerator.resize_gpu(roi_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            else:
                roi_image = cv2.resize(roi_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale nếu cần - GPU accelerated
        if len(roi_image.shape) == 3:
            if _gpu_accelerator:
                roi_processed = _gpu_accelerator.cvt_color_gpu(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                roi_processed = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_processed = roi_image.copy()
        
        # Gaussian Blur để cải thiện OCR - GPU accelerated
        if _gpu_accelerator:
            roi_processed = _gpu_accelerator.gaussian_blur_gpu(roi_processed, (3, 3), 0)
        else:
            roi_processed = cv2.GaussianBlur(roi_processed, (3, 3), 0)
        
                # Thực hiện OCR với parameters tối ưu
        ocr_results = reader.readtext(roi_processed, 
                                    allowlist='0123456789.-ABCDEFGHIKLNORTUabcdefghiklnortu', 
                                    detail=1, 
                                    paragraph=False, 
                                    batch_size=1, 
                                    text_threshold=0.4,
                                    link_threshold=0.2, 
                                    low_text=0.3, 
                                    mag_ratio=2, 
                                    slope_ths=0.05,
                                    decoder='beamsearch'
                                    )
        
        if ocr_results and len(ocr_results) > 0:
            # Lấy kết quả có confidence cao nhất
            best_result = max(ocr_results, key=lambda x: x[2])
            best_text = best_result[1]
            best_confidence = best_result[2]
            original_value = best_text
            has_text = True
            
            # Kiểm tra nếu kết quả ban đầu có dấu trừ ở đầu
            has_negative_sign = best_text.startswith('-')
            
            # Post-processing cơ bản - chuyển đổi 'O' thành '0' nếu chỉ 1 ký tự
            if len(best_text) == 1 and best_text.upper() == 'O':
                best_text = '0'
            
            # Kiểm tra và chuyển đổi chuỗi kết quả nếu có dạng số (giống logic cũ)
            if len(best_text) >= 2:
                # Đếm số lượng các ký tự dễ nhầm lẫn
                chars_to_check = '01OUouIilC'
                suspicious_chars_count = sum(1 for char in best_text if char in chars_to_check)
                # Nếu có ít nhất 2 ký tự đáng ngờ và chiếm >= 30% chuỗi
                if suspicious_chars_count >= 2 and suspicious_chars_count / len(best_text) >= 0.3:
                    # Kiểm tra các mẫu đặc biệt, như chuỗi "uuuu" hoặc "iuuu" có thể là "1000"
                    upper_text = best_text.upper()
                    
                    # Trường hợp đặc biệt: chuỗi chứa nhiều U liên tiếp (có thể là số 0 lặp lại)
                    upper_text_no_dot = upper_text.replace('.', '')
                    if re.search(r'[IUO0Q]{2}', upper_text_no_dot):
                        temp_text = upper_text.replace('U', '0').replace('I', '1').replace('O', '0').replace('C','0').replace('Q','0')
                        if temp_text.replace('.', '').isdigit():
                            best_text = temp_text
                            is_text_result = False
                    # Trường hợp đặc biệt khác
                    else:
                        # Kiểm tra xem có ít nhất 60% ký tự là chữ cái đáng ngờ I, U, O
                        digit_like_chars_count = sum(1 for char in upper_text if char in 'OUICL')
                        if digit_like_chars_count / len(best_text) >= 0.7:
                            # Chuyển đổi tất cả ký tự dễ nhầm lẫn thành số tương ứng
                            cleaned_text = upper_text
                            cleaned_text = cleaned_text.replace('O', '0').replace('U', '0').replace('Q', '0')
                            cleaned_text = cleaned_text.replace('I', '1').replace('L', '1')
                            cleaned_text = cleaned_text.replace('C', '0').replace('D', '0')
                            
                            # Loại bỏ khoảng trắng nếu kết quả là số
                            cleaned_text = cleaned_text.replace(' ', '')
                            
                            # Kiểm tra nếu kết quả chỉ chứa chữ số
                            if cleaned_text.isdigit():
                                best_text = cleaned_text
                                # Đánh dấu là kết quả số để không bị xử lý như text
                                is_text_result = False
            
            # Đếm số lượng chữ số và chữ cái (loại trừ số 0 và chữ O)
            digit_count = sum(1 for char in best_text if char.isdigit() and char != '0')
            letter_count = sum(1 for char in best_text if char.isalpha() and char.upper() != 'O')
            
            # Kiểm tra nếu có nhiều chữ cái hơn chữ số
            is_text_result = letter_count > digit_count
            
            # Thêm lại dấu trừ ở đầu nếu kết quả ban đầu có
            if has_negative_sign and not best_text.startswith('-'):
                best_text = '-' + best_text
                
            # Kiểm tra xem ROI có allowed_values không
            has_allowed_values = allowed_values and len(allowed_values) > 0
            
            # Nếu là kết quả chủ yếu là chữ hoặc ROI có allowed_values, xử lý như text
            if has_text and (is_text_result or has_allowed_values):
                best_text = best_text.replace('0', 'O').replace('1', 'I').replace('2', 'Z').replace('3', 'E').replace('4', 'A').replace('5', 'S').replace('6', 'G').replace('7', 'T').replace('8', 'B').replace('9', 'P')
                best_text = best_text.upper()
                
                # Thêm kết quả cho ROI này (không có original_value cho kết quả text)
                if len(best_text) == 1:
                    best_text = best_text.replace('O', '0').replace('I', '1').replace('C','0').replace('S','5').replace('G','6').replace('A','4').replace('H','8').replace('L','1').replace('T','7').replace('U','0').replace('E','3').replace('Z','2').replace('Q','0')
                
                # Sử dụng hàm tối ưu để tìm best match với allowed_values
                if has_allowed_values:
                    best_match, match_score, match_method = find_best_allowed_value_match(
                        best_text, allowed_values, roi_name
                    )
                    
                    if best_match:
                        best_text = best_match
                    else:
                        best_text = allowed_values[0]
                
                return {
                    "roi_index": roi_name,
                    "text": best_text,
                    "confidence": best_confidence,
                    "has_text": has_text,
                    "original_value": original_value
                }
            
            # Nếu kết quả chủ yếu là số, xử lý theo định dạng decimal_places
            is_negative = best_text.startswith('-')
            best_text = best_text.upper()
            best_text = best_text.replace('O', '0').replace('I', '1').replace('C','0').replace('S','5').replace('G','6').replace('B','8').replace('T','7').replace('L','1').replace('H','8').replace('A','4').replace('E','3').replace('Z','2').replace('U','0')
            
            # Xử lý kết quả OCR có khoảng trắng giữa các số (ví dụ: "1 3")
            if ' ' in best_text and all(c.isdigit() or c == ' ' or c == '.' or c == '-' for c in best_text):
                best_text = best_text.replace(' ', '')
            
            if '-' in best_text[1:]:
                best_text = best_text[:-1] + best_text[-1].replace('-', '')
            
            # Kiểm tra lại sau khi đã xóa khoảng trắng - xử lý decimal_places cho số
            if has_text and re.match(r'^-?\d+\.?\d*$', best_text):
                try:
                    # Lấy decimal_places config từ cache
                    decimal_places_config = get_decimal_places_config_cached()
                    
                    # Kiểm tra xem có cấu hình cho ROI này không
                    best_text_clean = best_text[1:] if is_negative else best_text
                    
                    # Áp dụng decimal_places trước khi chuyển sang ROI tiếp theo
                    if (machine_type in decimal_places_config and 
                        screen_id in decimal_places_config[machine_type] and 
                        roi_name in decimal_places_config[machine_type][screen_id]):
                        
                        decimal_places = int(decimal_places_config[machine_type][screen_id][roi_name])
                        
                        # Xử lý các trường hợp khác nhau dựa trên decimal_places
                        if decimal_places == 0:
                            # Nếu decimal_places là 0, giữ lại tất cả các chữ số nhưng bỏ dấu chấm
                            formatted_text = str(best_text_clean).replace('.', '')
                            formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                        else:
                            # Đếm số chữ số thập phân hiện tại
                            current_decimal_places = 0
                            if '.' in best_text_clean:
                                dec_part = best_text_clean.split('.')[1]
                                current_decimal_places = len(dec_part)
                            
                            # Nếu số chữ số thập phân hiện tại bằng đúng số chữ số thập phân cần có
                            if current_decimal_places == decimal_places:
                                # Giữ nguyên số
                                formatted_text = best_text_clean
                                formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                            else:
                                # Xử lý khi có dấu thập phân
                                if '.' in best_text_clean:
                                    int_part, dec_part = best_text_clean.split('.')
                                    
                                    # Kết hợp phần nguyên và phần thập phân thành một chuỗi không có dấu chấm
                                    all_digits = int_part + dec_part
                                    
                                    # Đặt dấu chấm vào vị trí thích hợp theo decimal_places
                                    if decimal_places > 0:
                                        if len(all_digits) <= decimal_places:
                                            # Thêm số 0 phía trước và đặt dấu chấm sau số 0 đầu tiên
                                            padded_str = all_digits.zfill(decimal_places)
                                            formatted_text = f"0.{padded_str}"
                                            formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                        else:
                                            # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                            insert_pos = len(all_digits) - decimal_places
                                            formatted_text = f"{all_digits[:insert_pos]}.{all_digits[insert_pos:]}"
                                            formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                    else:
                                        # Nếu decimal_places = 0, bỏ dấu chấm
                                        formatted_text = all_digits
                                        formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                else:
                                    # Không có dấu chấm (số nguyên)
                                    num_str = str(best_text_clean)
                                    
                                    # Thêm phần thập phân nếu cần
                                    if decimal_places > 0:
                                        # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                        if len(num_str) <= decimal_places:
                                            # Nếu số chữ số ít hơn hoặc bằng decimal_places, thêm số 0 ở đầu
                                            padded_str = num_str.zfill(decimal_places)
                                            formatted_text = f"0.{padded_str}"
                                        else:
                                            # Đặt dấu chấm vào vị trí thích hợp
                                            insert_pos = len(num_str) - decimal_places
                                            formatted_text = f"{num_str[:insert_pos]}.{num_str[insert_pos:]}"
                                        
                                        formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                    else:
                                        # Giữ nguyên số nguyên nếu không cần thập phân
                                        formatted_text = num_str
                                        formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                        
                        # Cập nhật best_text cho các bước xử lý tiếp theo
                        best_text = formatted_text
                    else:
                        # Thêm xử lý đặc biệt cho "Machine OEE" nếu không tìm thấy trong cấu hình
                        if roi_name == "Machine OEE":
                            decimal_places = 2  # Áp dụng 2 chữ số thập phân cho Machine OEE theo yêu cầu
                            
                            # Xử lý định dạng số như các trường hợp khác
                            num_str = str(best_text_clean)
                            if len(num_str) <= decimal_places:
                                padded_str = num_str.zfill(decimal_places)
                                formatted_text = f"0.{padded_str}"
                            else:
                                insert_pos = len(num_str) - decimal_places
                                formatted_text = f"{num_str[:insert_pos]}.{num_str[insert_pos:]}"
                            
                            formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                            best_text = formatted_text
                        else:
                            # Nếu không có cấu hình decimal_places, giữ nguyên giá trị
                            formatted_text = best_text
                            formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                except Exception as e:
                    print(f"Error applying decimal places format for ROI {roi_name}: {str(e)}")
                    formatted_text = best_text
                    formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
            else:
                # Nếu không phải là số, giữ nguyên text
                formatted_text = best_text
            
            # Kiểm tra nếu ROI có chứa "working hours" trong tên 
            if "working hours" in roi_name.lower():
                # Loại bỏ tất cả các ký tự không phải số
                digits_only = re.sub(r'[^0-9]', '', formatted_text)
                
                # Kiểm tra xem chuỗi có đủ số để định dạng không
                if len(digits_only) >= 2:
                    # Thêm dấu ":" sau mỗi 2 ký tự từ phải sang trái
                    result = ""
                    for i in range(len(digits_only) - 1, -1, -1):
                        result = digits_only[i] + result
                        if i > 0 and (len(digits_only) - i) % 2 == 0:
                            result = ":" + result
                    
                    formatted_text = result
            
            # Xử lý dấu "-" không ở vị trí đầu tiên
            if "-" in formatted_text[1:]:
                formatted_text = formatted_text[0] + formatted_text[1:].replace('-', '.')
            
            return {
                "roi_index": roi_name,
                "text": formatted_text,
                "confidence": best_confidence,
                "has_text": True,
                "original_value": original_value
            }
        else:
            return {
                "roi_index": roi_name,
                "text": "",
                "confidence": 0,
                "has_text": False,
                "original_value": ""
            }
            
    except Exception as e:
        print(f"Error processing ROI {roi_name}: {e}")
        return {
            "roi_index": roi_name,
            "text": "ERROR",
            "confidence": 0,
            "has_text": False,
            "original_value": ""
        }

def process_roi_with_retry_logic_optimized(roi_args, original_filename):
    """
    Thêm retry logic cho ROI với confidence thấp hoặc chất lượng ảnh kém
    Dựa trên logic từ app_old.py
    """
    (roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id) = roi_args
    
    # Thực hiện OCR ban đầu
    result = process_single_roi_optimized(roi_args)
    
    # Kiểm tra nếu cần retry
    best_confidence = result.get('confidence', 0)
    
    if best_confidence < 0.3:  # Threshold để retry
        print(f"Confidence is below threshold for ROI {roi_name}. Trying alternative approach with connected component analysis...")
        
        try:
            # Thực hiện phân tích thành phần liên kết như trong app_old.py
            # 1. Chuyển ảnh ROI sang grayscale nếu chưa phải
            if len(roi_image.shape) > 2:
                roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi_image.copy()
            
            # 2. Làm mờ nhẹ để giảm nhiễu bề mặt
            roi_blur = cv2.GaussianBlur(roi_gray, (7,7), 0)
            
            # 3. Áp dụng adaptive threshold để tách nền thay đổi độ sáng
            roi_th = cv2.adaptiveThreshold(
                roi_blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,  # Đảo ngược để số là màu trắng, nền đen
                blockSize=11, C=2
            )
            
            # 4. Sử dụng connected component analysis để lọc dựa trên kích thước
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_th, connectivity=8)
            
            # Tạo một mask trống để giữ lại các số
            digit_mask = np.zeros_like(roi_th)
            
            # Lọc các thành phần dựa trên kích thước
            min_area = 50      # Diện tích tối thiểu của số
            max_area = 2000    # Diện tích tối đa của số
            min_width = 5      # Chiều rộng tối thiểu
            min_height = 10    # Chiều cao tối thiểu
            max_width = 100    # Chiều rộng tối đa
            max_height = 100   # Chiều cao tối đa
            aspect_ratio_min = 0.2  # Tỷ lệ chiều rộng/chiều cao tối thiểu
            aspect_ratio_max = 5.0  # Tỷ lệ chiều rộng/chiều cao tối đa
            
            # Bỏ qua label 0 vì đó là nền (background)
            for label in range(1, num_labels):
                # Lấy thông tin của component
                x, y, w, h, area = stats[label]
                
                # Tính toán tỷ lệ khung hình
                aspect_ratio = w / h if h > 0 else 0
                
                # Kiểm tra các điều kiện để xác định là số
                if (min_area < area < max_area and 
                    min_width < w < max_width and 
                    min_height < h < max_height and
                    aspect_ratio_min < aspect_ratio < aspect_ratio_max):
                    # Đây có khả năng là số, giữ lại trong mask
                    digit_mask[labels == label] = 255
            
            # 5. Thực hiện OCR trên mask đã tạo với thông số tối ưu
            retry_results = reader.readtext(digit_mask, 
                                                      allowlist='0123456789.-ABCDEFGHIKLNORTUabcdefghiklnortu', 
                                                      detail=1, 
                                                      paragraph=False, 
                                                      batch_size=1, 
                                                      text_threshold=0.4,
                                                      link_threshold=0.2, 
                                                      low_text=0.3, 
                                                      mag_ratio=2, 
                                                      slope_ths=0.05,
                                                      decoder='beamsearch'
                                                      )
            
            # Kiểm tra nếu có kết quả OCR
            if retry_results and len(retry_results) > 0:
                # Tìm kết quả có confidence cao nhất
                best_retry_result = max(retry_results, key=lambda x: x[2])
                retry_text = best_retry_result[1]  # Text
                retry_confidence = best_retry_result[2]  # Confidence
                
                # Xử lý retry text tương tự như original
                upper_text = retry_text.upper()
                upper_text_no_dot = upper_text.replace('.', '')
                
                if re.search(r'[IUO0Q]{2}', upper_text_no_dot):
                    temp_text = upper_text.replace('U', '0').replace('I', '1').replace('O', '0').replace('C','0').replace('Q','0')
                    if temp_text.replace('.', '').isdigit():
                        retry_text = temp_text
                else:
                    # Kiểm tra xem có ít nhất 60% ký tự là chữ cái đáng ngờ I, U, O
                    digit_like_chars_count = sum(1 for char in upper_text if char in 'OUICL')
                    if digit_like_chars_count / len(retry_text) >= 0.7:
                        # Chuyển đổi tất cả ký tự dễ nhầm lẫn thành số tương ứng
                        cleaned_text = upper_text
                        cleaned_text = cleaned_text.replace('O', '0').replace('U', '0').replace('Q', '0')
                        cleaned_text = cleaned_text.replace('I', '1').replace('L', '1')
                        cleaned_text = cleaned_text.replace('C', '0').replace('D', '0')
                        
                        # Loại bỏ khoảng trắng nếu kết quả là số
                        cleaned_text = cleaned_text.replace(' ', '')
                        
                        # Kiểm tra nếu kết quả chỉ chứa chữ số
                        if cleaned_text.isdigit():
                            retry_text = cleaned_text
                
                # Kiểm tra với allowed_values trước nếu có
                retry_matched_with_allowed_values = False
                if allowed_values and len(allowed_values) > 0:
                    best_match, match_score, match_method = find_best_allowed_value_match(
                        retry_text, allowed_values, f"{roi_name}_retry"
                    )
                    
                    if best_match:
                        retry_text = best_match
                        retry_matched_with_allowed_values = True
                    else:
                        retry_text = allowed_values[0]
                        retry_matched_with_allowed_values = True
                
                # Nếu chưa match với allowed_values, xử lý format decimal
                if not retry_matched_with_allowed_values:
                    # Chấp nhận retry result nếu confidence > 0.3 hoặc tốt hơn original
                    if retry_confidence > 0.3 or retry_confidence > best_confidence:
                        # Áp dụng định dạng theo decimal_places nếu kết quả là số và có cấu hình
                        formatted_retry_text = retry_text
                        
                        # Xử lý trường hợp retry_text chỉ có các ký tự số nhưng không có dấu chấm thập phân
                        is_numeric = re.match(r'^-?\d+$', retry_text) is not None
                        
                        if (is_numeric or re.match(r'^-?\d+\.?\d*$', retry_text)):
                            try:
                                decimal_places_config = get_decimal_places_config_cached()
                                
                                if (machine_type in decimal_places_config and 
                                    screen_id in decimal_places_config[machine_type] and 
                                    roi_name in decimal_places_config[machine_type][screen_id]):
                                    
                                    is_negative = retry_text.startswith('-')
                                    clean_text = retry_text[1:] if is_negative else retry_text
                                    decimal_places = int(decimal_places_config[machine_type][screen_id][roi_name])
                                    
                                    # Xử lý tương tự như phần xử lý decimal_places ở trên
                                    if decimal_places == 0:
                                        # Nếu decimal_places là 0, bỏ dấu chấm
                                        formatted_retry_text = clean_text.replace('.', '')
                                    else:
                                        # Xử lý vị trí dấu thập phân
                                        if '.' in clean_text:
                                            int_part, dec_part = clean_text.split('.')
                                            # Kết hợp phần nguyên và phần thập phân thành một chuỗi không có dấu chấm
                                            all_digits = int_part + dec_part
                                            
                                            # Đặt dấu chấm vào vị trí thích hợp theo decimal_places
                                            if decimal_places > 0:
                                                if len(all_digits) <= decimal_places:
                                                    # Thêm số 0 phía trước và đặt dấu chấm sau số 0 đầu tiên
                                                    padded_str = all_digits.zfill(decimal_places)
                                                    formatted_retry_text = f"0.{padded_str}"
                                                else:
                                                    # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                                    insert_pos = len(all_digits) - decimal_places
                                                    formatted_retry_text = f"{all_digits[:insert_pos]}.{all_digits[insert_pos:]}"
                                            else:
                                                # Nếu decimal_places = 0, bỏ dấu chấm
                                                formatted_retry_text = all_digits
                                        else:
                                            # Không có dấu chấm (số nguyên)
                                            num_str = clean_text
                                            
                                            # Thêm phần thập phân nếu cần
                                            if decimal_places > 0:
                                                # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                                if len(num_str) <= decimal_places:
                                                    # Nếu số chữ số ít hơn hoặc bằng decimal_places, thêm số 0 ở đầu
                                                    padded_str = num_str.zfill(decimal_places)
                                                    formatted_retry_text = f"0.{padded_str}"
                                                else:
                                                    # Đặt dấu chấm vào vị trí thích hợp
                                                    insert_pos = len(num_str) - decimal_places
                                                    formatted_retry_text = f"{num_str[:insert_pos]}.{num_str[insert_pos:]}"
                                            else:
                                                # Giữ nguyên số nguyên nếu không cần thập phân
                                                formatted_retry_text = num_str
                                    
                                    # Thêm dấu âm nếu cần
                                    if is_negative:
                                        formatted_retry_text = f"-{formatted_retry_text}"
                            except Exception as e:
                                print(f"Error formatting retry text: {str(e)}")
                        
                        # Kiểm tra nếu ROI có chứa "working hours" trong tên 
                        if "working hours" in roi_name.lower() and re.match(r'^\d+\.\d+\.\d+$', formatted_retry_text):
                            # Chuyển đổi từ định dạng số.số.số sang số:số:số
                            formatted_retry_text = formatted_retry_text.replace('.', ':').replace(' ', ':').replace('-', ':')
                        
                        # Xử lý dấu "-" không ở vị trí đầu tiên
                        if "-" in formatted_retry_text[1:]:
                            formatted_retry_text = formatted_retry_text[0] + formatted_retry_text[1:].replace('-', '.')
                        
                        # Cập nhật kết quả với retry result
                        result["text"] = formatted_retry_text.replace('C','0')
                        result["confidence"] = retry_confidence
                        result["has_text"] = True
                        result["original_value"] = retry_text
                        print(f"[OK] Updated result with retry OCR: '{formatted_retry_text}' (confidence: {retry_confidence:.4f})")
                    else:
                        print(f"Keeping original result: retry confidence {retry_confidence:.4f} not meeting threshold 0.3")
                else:
                    # Nếu đã match với allowed_values, cập nhật result
                    result["text"] = retry_text
                    result["confidence"] = retry_confidence
                    result["has_text"] = True
                    result["original_value"] = retry_text
                    print(f"[OK] Updated result with allowed value match: '{retry_text}'")
            else:
                print(f"[ERROR] No retry OCR results found on digit mask")
        
        except Exception as e:
            print(f"Error in retry logic for ROI {roi_name}: {e}")
    
    return result

# Thêm hàm căn chỉnh ảnh từ wrap_perspective.py
class ImageAligner:
    def __init__(self, template_img, source_img):
        """Khởi tạo với ảnh mẫu và ảnh nguồn đã được đọc bởi OpenCV"""
        self.template_img = template_img.copy()
        self.source_img = source_img.copy()
        
        # Store warped image
        self.warped_img = None
        
        # Sử dụng SIFT detector global để tối ưu hiệu suất
        self.detector = sift_detector
        
    def align_images(self):
        """Căn chỉnh ảnh nguồn để khớp với ảnh mẫu bằng feature matching và homography."""
        # Kiểm tra SIFT detector có khả dụng không
        if self.detector is None:
            print("Warning: SIFT detector not available, returning original image")
            return self.source_img.copy()
            
        # Convert images to grayscale for feature detection
        template_gray = cv2.cvtColor(self.template_img, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(self.source_img, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        template_keypoints, template_descriptors = self.detector.detectAndCompute(template_gray, None)
        source_keypoints, source_descriptors = self.detector.detectAndCompute(source_gray, None)
        
        # Kiểm tra FLANN matcher có khả dụng không
        if flann_matcher is None:
            print("Warning: FLANN matcher not available, returning original image")
            return self.source_img.copy()
            
        # Sử dụng FLANN matcher global để tối ưu hiệu suất
        matches = flann_matcher.knnMatch(source_descriptors, template_descriptors, k=2)
        
        # Filter good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches")
        
        if len(good_matches) < 10:
            print("Warning: Not enough good matches found for robust homography estimation")
            return self.source_img.copy()  # Return original if can't align
        
        # Extract location of good matches
        source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches])
        template_points = np.float32([template_keypoints[m.trainIdx].pt for m in good_matches])
        
        # Find homography matrix
        H, mask = cv2.findHomography(source_points, template_points, cv2.RANSAC, 5.0)
        
        # Warp source image to align with template
        h, w = self.template_img.shape[:2]
        self.warped_img = cv2.warpPerspective(self.source_img, H, (w, h))
        self.homography_matrix = H
        
        return self.warped_img
    
    def get_homography_matrix(self):
        """Trả về ma trận homography đã tính"""
        if hasattr(self, 'homography_matrix'):
            return self.homography_matrix
        return None
    
    def transform_roi_coordinates(self, roi_coordinates):
        """
        Biến đổi tọa độ ROI dựa trên ma trận homography
        
        Args:
            roi_coordinates: Danh sách các tọa độ ROI, mỗi item là tuple (x1, y1, x2, y2)
            
        Returns:
            Danh sách các tọa độ ROI đã biến đổi
        """
        try:
            H = self.get_homography_matrix()
            if H is None:
                print("No homography matrix available.")
                return roi_coordinates
        
            transformed_coordinates = []
            for coord_set in roi_coordinates:
                # Đảm bảo coord_set là list/tuple với 4 phần tử
                if len(coord_set) != 4:
                    print(f"Invalid coordinate set: {coord_set}")
                    continue
                    
                x1, y1, x2, y2 = coord_set
                
                # Chuyển đổi điểm góc trên bên trái
                tx1, ty1 = self.transform_point((x1, y1), H)
                
                # Chuyển đổi điểm góc dưới bên phải
                tx2, ty2 = self.transform_point((x2, y2), H)
                
                transformed_coordinates.append((tx1, ty1, tx2, ty2))
        
            return transformed_coordinates
        except Exception as e:
            print(f"Error transforming ROI coordinates: {str(e)}")
            return roi_coordinates
    
    def transform_point(self, point, H):
        """Áp dụng ma trận homography cho một điểm"""
        x, y = point
        # Chuyển đổi sang tọa độ thuần nhất
        p = np.array([x, y, 1])
        # Áp dụng ma trận homography
        p_transformed = np.dot(H, p)
        # Chuyển về tọa độ Cartesian
        x_transformed = p_transformed[0] / p_transformed[2]
        y_transformed = p_transformed[1] / p_transformed[2]
        
        return int(x_transformed), int(y_transformed)

# Sửa lại hàm tiền xử lý ảnh để sử dụng sau khi ảnh đã được căn chỉnh
def preprocess_hmi_image_with_alignment(image, template_path, roi_coordinates, original_filename):
    """Tiền xử lý ảnh với căn chỉnh perspective và điều chỉnh ROI"""
    # Đọc ảnh template
    template_img = cv2.imread(template_path)
    if template_img is None:
        print(f"Warning: Could not read template image at {template_path}")
        return preprocess_hmi_image(image, roi_coordinates, original_filename)
    
    print(f"Image shape: {image.shape}, Template shape: {template_img.shape}")
    
    # Tạo thư mục để lưu ảnh đã căn chỉnh
    aligned_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned')
    if not os.path.exists(aligned_folder):
        os.makedirs(aligned_folder)
    
    # Căn chỉnh ảnh
    aligner = ImageAligner(template_img, image)
    aligned_image = aligner.align_images()
    
    # Lưu ảnh đã căn chỉnh
    # aligned_filename = f"aligned_{original_filename}"
    # aligned_path = os.path.join(aligned_folder, aligned_filename)
    # cv2.imwrite(aligned_path, aligned_image)
    
    # Sử dụng ảnh đã căn chỉnh với các tọa độ ROI gốc
    # Vì ảnh nguồn đã được căn chỉnh theo template, nên các tọa độ ROI gốc sẽ hoạt động
    # trực tiếp trên ảnh đã được căn chỉnh
    results = preprocess_hmi_image(aligned_image, roi_coordinates, original_filename)
    
    # Thêm thông tin về ảnh đã căn chỉnh
    for result in results:
        result["aligned_image_path"] = f"/api/images/aligned/{aligned_filename}"
    
    return results

# Route để truy cập ảnh đã căn chỉnh
@app.route('/api/images/aligned/<filename>', methods=['GET'])
def get_aligned_image(filename):
    aligned_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned')
    try:
        return send_from_directory(aligned_folder, filename)
    except:
        abort(404)

# Hàm đọc bộ khung ROI từ file roi_info.json
def get_roi_coordinates(machine_code, screen_id=None, machine_type=None):
    try:
        # Sử dụng đường dẫn tuyệt đối để đảm bảo tìm thấy file
        roi_json_path = 'roi_data/roi_info.json'
        if not os.path.exists(roi_json_path):
            roi_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data/roi_info.json')
        
        print(f"Reading ROI info from: {roi_json_path}")
        with open(roi_json_path, 'r', encoding='utf-8') as f:
            roi_data = json.load(f)
        
        # Kiểm tra xem định dạng tọa độ có được chuẩn hóa không
        is_normalized = False
        if "metadata" in roi_data and "coordinate_format" in roi_data["metadata"]:
            is_normalized = roi_data["metadata"]["coordinate_format"].lower() == "normalized"
            print(f"Coordinate format is {'normalized' if is_normalized else 'pixel-based'}")
        
        # Nếu machine_type không được cung cấp, lấy từ machine_screens.json
        if not machine_type:
            machine_type = get_machine_type(machine_code)
            if not machine_type:
                print(f"Could not determine machine_type for machine_code: {machine_code}")
                return [], []
            print(f"Determined machine_type: {machine_type} for machine_code: {machine_code}")
        
        screen_name = None
        
        # Kiểm tra xem screen_id có phải là tên màn hình không
        if isinstance(screen_id, str) and screen_id in ["Production Data", "Faults", "Feeders and Conveyors", 
                                                  "Main Machine Parameters", "Selectors and Maintenance",
                                                  "Setting", "Temp", "Plasticizer", "Overview", "Tracking", "Production", 
                                                  "Clamp", "Ejector", "Injection"]:
            screen_name = screen_id
            print(f"Using screen_id as screen_name: {screen_name}")
        elif screen_id is not None:
            # Lấy tên màn hình từ screen_id (nếu là numeric id)
            machine_screens_path = 'roi_data/machine_screens.json'
            if not os.path.exists(machine_screens_path):
                machine_screens_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data/machine_screens.json')
            
            with open(machine_screens_path, 'r', encoding='utf-8') as f:
                machine_screens = json.load(f)
            
            # Tìm trong areas
            for area_code, area_info in machine_screens.get("areas", {}).items():
                machines = area_info.get("machines", {})
                if machine_code in machines:
                    for screen in machines[machine_code].get("screens", []):
                        if str(screen.get("id")) == str(screen_id):
                            screen_name = screen.get("screen_id")
                            print(f"Found screen_name: {screen_name} for screen_id: {screen_id} in machine_code: {machine_code}")
                            break
        
        print(f"Looking for ROIs in machine_type: {machine_type}, screen: {screen_name} (id: {screen_id})")
        
        # Tìm trong machines
        if machine_type in roi_data.get("machines", {}):
            screens_data = roi_data["machines"][machine_type].get("screens", {})
            
            if screen_name and screen_name in screens_data:
                roi_list = screens_data[screen_name]
                
                # Xử lý định dạng ROI
                roi_coordinates = []
                roi_names = []
                
                for roi_item in roi_list:
                    if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                        roi_coordinates.append(roi_item["coordinates"])
                        roi_names.append(roi_item["name"])
                    else:
                        roi_coordinates.append(roi_item)
                        roi_names.append(f"ROI_{len(roi_names)}")
                
                print(f"Found {len(roi_coordinates)} ROIs for {screen_name}")
                return roi_coordinates, roi_names
            else:
                print(f"Screen '{screen_name}' not found in roi_info.json for machine_type {machine_type}")
                print(f"Available screens: {list(screens_data.keys())}")
        else:
            print(f"Machine type '{machine_type}' not found in roi_info.json")
            print(f"Available machine types: {list(roi_data.get('machines', {}).keys())}")
        
        print(f"No ROI coordinates found for machine_code={machine_code}, screen_id={screen_id}, machine_type={machine_type}")
        return [], []
    except Exception as e:
        print(f"Error reading ROI coordinates: {str(e)}")
        traceback.print_exc()
        return [], []

# Thêm hàm mới để lấy loại máy từ mã máy
def get_machine_type(machine_code):
    """
    Lấy loại máy (machine_type) từ mã máy (machine_code)
    
    Args:
        machine_code: Mã máy (ví dụ: IE-F1-CWA01, IE-F4-WBI01)
        
    Returns:
        str: Loại máy (F1, F41, F42, ...) hoặc None nếu không tìm thấy
    """
    try:
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm kiếm trong cấu trúc areas
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            if machine_code in machines:
                return machines[machine_code].get('type')
        
        # Nếu không tìm thấy, trả về None
        return None
    except Exception as e:
        print(f"Error getting machine type: {str(e)}")
        return None

# Thêm hàm mới để lấy khu vực từ mã máy
def get_area_for_machine(machine_code):
    """
    Lấy khu vực (area) chứa mã máy (machine_code)
    
    Args:
        machine_code: Mã máy (ví dụ: IE-F1-CWA01, IE-F4-WBI01)
        
    Returns:
        str: Mã khu vực (F1, F4, ...) hoặc None nếu không tìm thấy
    """
    try:
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm kiếm trong cấu trúc areas
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            if machine_code in machines:
                return area_code
        
        # Nếu không tìm thấy, trả về None
        return None
    except Exception as e:
        print(f"Error getting area for machine: {str(e)}")
        return None

def get_machine_name_from_code(machine_code):
    """
    Lấy tên máy (machine_name) từ mã máy (machine_code) trong file machine_screens.json
    
    Args:
        machine_code: Mã máy (ví dụ: IE-F1-CWA01, IE-F4-WBI01)
        
    Returns:
        str: Tên máy (ví dụ: "Máy IE-F1-CWA01") hoặc None nếu không tìm thấy
    """
    try:
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm kiếm trong cấu trúc areas
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            if machine_code in machines:
                machine_info = machines[machine_code]
                return machine_info.get('name')
        
        # Nếu không tìm thấy, trả về None
        return None
    except Exception as e:
        print(f"Error getting machine name from code: {str(e)}")
        return None

def get_all_machine_types():
    """
    Lấy tất cả machine_type có sẵn từ file machine_screens.json
    
    Returns:
        list: Danh sách các machine_type duy nhất
    """
    try:
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return []
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        machine_types = set()
        # Tìm kiếm trong cấu trúc areas
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            for machine_code, machine_info in machines.items():
                machine_type = machine_info.get('type')
                if machine_type:
                    machine_types.add(machine_type)
        
        return list(machine_types)
    except Exception as e:
        print(f"Error getting all machine types: {str(e)}")
        return []

def find_machine_code_from_template(template_filename):
    """
    Tìm machine_code từ template filename và machine_type
    
    Args:
        template_filename: Tên file template (ví dụ: template_F1_Main Machine Parameters.jpg)
        
    Returns:
        tuple: (machine_code, area) hoặc (None, None) nếu không tìm thấy
    """
    try:
        # Trích xuất machine_type từ filename template
        # Format: template_{machine_type}_{screen_name}.ext
        parts = template_filename.split('_')
        if len(parts) < 3:
            return None, None
        
        machine_type = parts[1]  # Lấy machine_type từ template filename
        
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None, None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm machine_code đầu tiên có machine_type tương ứng
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            for machine_code, machine_info in machines.items():
                if machine_info.get('type') == machine_type:
                    return machine_code, area_code
        
        return None, None
    except Exception as e:
        print(f"Error finding machine code from template: {str(e)}")
        return None, None

# Keep original function as backup
def auto_detect_machine_and_screen_original(image):
    """Original function renamed for backup"""
    return auto_detect_machine_and_screen(image)

# Replace main function with fast version
auto_detect_machine_and_screen = auto_detect_machine_and_screen_smart

def perform_ocr_on_roi_optimized(image, roi_coordinates, original_filename, template_path=None, roi_names=None, machine_code=None, screen_id=None):
    """
    Phiên bản tối ưu của perform_ocr_on_roi với parallel processing và caching
    """
    try:
        # Kiểm tra tham số đầu vào
        if roi_coordinates is None or len(roi_coordinates) == 0:
            return []
        
        # Tạo tên ROI mặc định nếu cần
        if roi_names is None or len(roi_names) != len(roi_coordinates):
            roi_names = [f"ROI_{i}" for i in range(len(roi_coordinates))]
        
        # Kiểm tra EasyOCR
        if not HAS_EASYOCR or reader is None:
            return [{"roi_index": roi_names[i], "text": "OCR_NOT_AVAILABLE", "confidence": 0, "has_text": False, "original_value": ""} 
                   for i in range(len(roi_coordinates))]
        
        # Lấy kích thước ảnh
        img_height, img_width = image.shape[:2]
        
        # Sử dụng cached machine info
        machine_info = get_machine_info_cached()
        machine_code = machine_info.get('machine_code', 'F41') if machine_code is None else machine_code
        screen_id = machine_info.get('screen_id', 'Main') if screen_id is None else screen_id
        
        # Sử dụng cached ROI info
        roi_info = get_roi_info_cached()
        machine_type = get_machine_type(machine_code)
        
        # Chuẩn bị dữ liệu cho parallel processing
        roi_args = []
        
        for i, coords in enumerate(roi_coordinates):
            try:
                # Chuyển đổi tọa độ
                if len(coords) != 4:
                    continue
                
                # Kiểm tra normalized coordinates
                is_normalized = any(isinstance(value, float) and 0 <= value <= 1 for value in coords)
                
                if is_normalized:
                    x1, y1, x2, y2 = coords
                    x1, x2 = int(x1 * img_width), int(x2 * img_width)
                    y1, y2 = int(y1 * img_height), int(y2 * img_height)
                else:
                    x1, y1, x2, y2 = [int(float(c)) for c in coords]
                
                # Đảm bảo thứ tự tọa độ
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Kiểm tra tọa độ hợp lệ
                if x1 < 0 or y1 < 0 or x2 >= img_width or y2 >= img_height or x1 >= x2 or y1 >= y2:
                    continue
                
                # Cắt ROI
                roi_image = image[y1:y2, x1:x2]
                roi_name = roi_names[i] if i < len(roi_names) else f"ROI_{i}"
                
                # Kiểm tra allowed_values từ cache
                allowed_values = []
                is_special_on_off_case = False
                
                # Tìm allowed_values nhanh từ cache (thử với machine_code trước)
                if machine_code in roi_info.get("machines", {}):
                    screens = roi_info["machines"][machine_code].get("screens", {})
                    if screen_id in screens:
                        roi_list = screens[screen_id]
                        for roi_item in roi_list:
                            if isinstance(roi_item, dict) and roi_item.get("name") == roi_name:
                                allowed_values = roi_item.get("allowed_values", [])
                                if "ON" in allowed_values and "OFF" in allowed_values:
                                    is_special_on_off_case = True
                                break
                
                # Nếu không tìm thấy với machine_code, thử với machine_type
                if not allowed_values and machine_type in roi_info.get("machines", {}):
                    screens = roi_info["machines"][machine_type].get("screens", {})
                    if screen_id in screens:
                        roi_list = screens[screen_id]
                        for roi_item in roi_list:
                            if isinstance(roi_item, dict) and roi_item.get("name") == roi_name:
                                allowed_values = roi_item.get("allowed_values", [])
                                if "ON" in allowed_values and "OFF" in allowed_values:
                                    is_special_on_off_case = True
                                break
                
                # Thêm vào danh sách args cho parallel processing
                roi_args.append((roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id))
                
            except Exception as e:
                print(f"Error preparing ROI {i}: {e}")
                continue
        
        # Xử lý ROIs song song với Enhanced Parallel Processor
        if len(roi_args) <= 2:
            # Nếu ít ROI, xử lý tuần tự để tránh overhead
            results = [process_single_roi_optimized(args) for args in roi_args]
        else:
            # Xử dụng ROI Processor với adaptive threading
            if OPTIMIZATION_MODULES_AVAILABLE and _roi_processor:
                results = _roi_processor.process_rois(roi_args, process_single_roi_optimized)
            else:
                # Fallback to standard thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(roi_args), 8)) as executor:
                    results = list(executor.map(process_single_roi_optimized, roi_args))
        
        # Post-process: Kiểm tra kết quả có confidence thấp không và thực hiện retry nếu cần
        final_results = []
        for i, result in enumerate(results):
            if result.get('confidence', 0) < 0.3:
                # Confidence thấp, thực hiện retry với connected component analysis
                try:
                    roi_args_item = roi_args[i]
                    retry_result = process_roi_with_retry_logic_optimized(roi_args_item, original_filename)
                    if retry_result and retry_result.get('confidence', 0) > result.get('confidence', 0):
                        final_results.append(retry_result)
                    else:
                        final_results.append(result)
                except:
                    final_results.append(result)
            else:
                final_results.append(result)
        
        return final_results
        
    except Exception as e:
        print(f"Error in perform_ocr_on_roi_optimized: {e}")
        traceback.print_exc()
        return []

def process_roi_with_retry_logic(roi_args, original_filename):
    """
    Xử lý retry logic với connected component analysis cho ROI có confidence thấp
    """
    try:
        (roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id) = roi_args
        
        # Kiểm tra EasyOCR có khả dụng không
        if not HAS_EASYOCR or reader is None:
            return None
        
        print(f"Processing retry logic for ROI: {roi_name}")
        
        # 1. Chuyển ảnh ROI sang grayscale nếu chưa phải
        if len(roi_image.shape) > 2:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_image.copy()
        
        # 2. Làm mờ nhẹ để giảm nhiễu bề mặt
        roi_blur = cv2.GaussianBlur(roi_gray, (7,7), 0)
        
        # 3. Áp dụng adaptive threshold để tách nền thay đổi độ sáng
        roi_th = cv2.adaptiveThreshold(
            roi_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Đảo ngược để số là màu trắng, nền đen
            blockSize=11, C=2
        )
        
        # 4. Sử dụng connected component analysis để lọc dựa trên kích thước
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_th, connectivity=8)
        
        # Tạo một mask trống để giữ lại các số
        digit_mask = np.zeros_like(roi_th)
        
        # Lọc các thành phần dựa trên kích thước (có thể điều chỉnh các giá trị này)
        min_area = 50      # Diện tích tối thiểu của số
        max_area = 2000    # Diện tích tối đa của số
        min_width = 5      # Chiều rộng tối thiểu
        min_height = 10    # Chiều cao tối thiểu
        max_width = 100    # Chiều rộng tối đa
        max_height = 100   # Chiều cao tối đa
        aspect_ratio_min = 0.2  # Tỷ lệ chiều rộng/chiều cao tối thiểu
        aspect_ratio_max = 5.0  # Tỷ lệ chiều rộng/chiều cao tối đa
        
        # Bỏ qua label 0 vì đó là nền (background)
        for label in range(1, num_labels):
            # Lấy thông tin của component
            x, y, w, h, area = stats[label]
            
            # Tính toán tỷ lệ khung hình
            aspect_ratio = w / h if h > 0 else 0
            
            # Kiểm tra các điều kiện để xác định là số
            if (min_area < area < max_area and 
                min_width < w < max_width and 
                min_height < h < max_height and
                aspect_ratio_min < aspect_ratio < aspect_ratio_max):
                # Đây có khả năng là số, giữ lại trong mask
                digit_mask[labels == label] = 255
        
            # 5. Thực hiện OCR trên mask đã tạo với thông số tối ưu
            retry_results = reader.readtext(digit_mask, 
                                          allowlist='0123456789.-ABCDEFGHIKLNORTUabcdefghiklnortu', 
                                          detail=1, 
                                          paragraph=False, 
                                          batch_size=1, 
                                          text_threshold=0.4,
                                          link_threshold=0.2, 
                                          low_text=0.3, 
                                          mag_ratio=2, 
                                          slope_ths=0.05,
                                          decoder='beamsearch'
                                          )
        
        # Kiểm tra nếu có kết quả OCR
        if retry_results and len(retry_results) > 0:
            # Tìm kết quả có confidence cao nhất
            best_retry_result = max(retry_results, key=lambda x: x[2])
            retry_text = best_retry_result[1]  # Text
            retry_confidence = best_retry_result[2]  # Confidence
            
            # Chuyển đổi sang chữ hoa và kiểm tra pattern
            upper_text = retry_text.upper()
            upper_text_no_dot = upper_text.replace('.', '')
            
            if re.search(r'[IUO0Q]{2}', upper_text_no_dot):
                temp_text = upper_text.replace('U', '0').replace('I', '1').replace('O', '0').replace('C','0').replace('Q','0')
                if temp_text.replace('.', '').isdigit():
                    retry_text = temp_text
            # Trường hợp đặc biệt khác
            else:
                # Kiểm tra xem có ít nhất 60% ký tự là chữ cái đáng ngờ I, U, O
                digit_like_chars_count = sum(1 for char in upper_text if char in 'OUICL')
                if digit_like_chars_count / len(retry_text) >= 0.7:
                    # Chuyển đổi tất cả ký tự dễ nhầm lẫn thành số tương ứng
                    cleaned_text = upper_text
                    cleaned_text = cleaned_text.replace('O', '0').replace('U', '0').replace('Q', '0')
                    cleaned_text = cleaned_text.replace('I', '1').replace('L', '1')
                    cleaned_text = cleaned_text.replace('C', '0').replace('D', '0')
                    
                    # Loại bỏ khoảng trắng nếu kết quả là số
                    cleaned_text = cleaned_text.replace(' ', '')
                    
                    # Kiểm tra nếu kết quả chỉ chứa chữ số
                    if cleaned_text.isdigit():
                        retry_text = cleaned_text
            
            # Xử lý khoảng trắng giữa các số (tương tự như xử lý trên best_text)
            retry_text = retry_text.upper()
            retry_text = retry_text.replace('O', '0').replace('I', '1').replace('C','0').replace('S','5').replace('G','6').replace('B','8')
            
            # Kiểm tra allowed_values cho retry_text
            if allowed_values and len(allowed_values) > 0:
                retry_best_match, retry_match_score, retry_match_method = find_best_allowed_value_match(
                    retry_text, allowed_values, f"{roi_name}_retry"
                )
                
                if retry_best_match:
                    retry_text = retry_best_match
                else:
                    retry_text = allowed_values[0]
            
            return {
                "roi_index": roi_name,
                "text": retry_text,
                "confidence": retry_confidence,
                "has_text": True,
                "original_value": retry_text
            }
        
        return None
        
    except Exception as e:
        print(f"Error in retry logic for ROI {roi_name}: {e}")
        return None

# Sửa lại hàm perform_ocr_on_roi để sử dụng ảnh đã căn chỉnh
def perform_ocr_on_roi(image, roi_coordinates, original_filename, template_path=None, roi_names=None, machine_code=None, screen_id=None):
    """
    Thực hiện OCR trên các vùng ROI đã xác định
    
    Args:
        image: Ảnh đầu vào
        roi_coordinates: Danh sách các tọa độ ROI
        original_filename: Tên file gốc
        template_path: Đường dẫn đến ảnh template nếu có
        roi_names: Danh sách tên của các ROI
        machine_code: Mã máy (tùy chọn)
        screen_id: ID màn hình (tùy chọn)
        
    Returns:
        Danh sách kết quả OCR cho mỗi ROI
    """
    try:
        # Kiểm tra các tham số đầu vào
        if roi_coordinates is None or len(roi_coordinates) == 0:
            return []
        
        # Nếu không có tên ROI được truyền vào, tạo tên mặc định
        if roi_names is None or len(roi_names) != len(roi_coordinates):
            roi_names = [f"ROI_{i}" for i in range(len(roi_coordinates))]
        
        # Kiểm tra EasyOCR đã được khởi tạo chưa
        if not HAS_EASYOCR or reader is None:
            mock_results = []
            for i, coords in enumerate(roi_coordinates):
                roi_name = roi_names[i] if i < len(roi_names) else f"ROI_{i}"
                mock_results.append({
                    "roi_index": roi_name,
                    "text": "OCR_NOT_AVAILABLE",
                    "confidence": 0,
                    "has_text": False,
                    "original_value": ""
                })
            return mock_results
        
        # Lấy kích thước ảnh
        img_height, img_width = image.shape[:2]
        
        # Lấy thông tin máy hiện tại
        machine_info = get_current_machine_info()
        if not machine_info:
            return []
        
        machine_code = machine_info['machine_code'] if machine_code is None else machine_code
        screen_id = machine_info['screen_id'] if screen_id is None else screen_id
        
        # Đọc cấu hình số thập phân
        decimal_places_config = get_decimal_places_config()
        
        # Tiền xử lý ảnh và thực hiện OCR trên mỗi ROI
        results = []
        for i, coords in enumerate(roi_coordinates):
            try:
                # Đảm bảo coords có đúng 4 giá trị
                if len(coords) != 4:
                    continue
                
                # Chuyển đổi tọa độ nếu cần
                is_normalized = False
                for value in coords:
                    if isinstance(value, float) and 0 <= value <= 1:
                        is_normalized = True
                        break
                
                if is_normalized:
                    # Chuyển đổi từ tọa độ chuẩn hóa sang tọa độ pixel
                    x1, y1, x2, y2 = coords
                    x1, x2 = int(x1 * img_width), int(x2 * img_width)
                    y1, y2 = int(y1 * img_height), int(y2 * img_height)
                else:
                    # Đã là tọa độ pixel, chỉ cần chuyển sang int
                    x1, y1, x2, y2 = coords
                    x1, x2 = int(float(x1)), int(float(x2))
                    y1, y2 = int(float(y1)), int(float(y2))
                
                # Đảm bảo thứ tự tọa độ chính xác
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                roi_name = roi_names[i] if i < len(roi_names) else f"ROI_{i}"
                
                # Kiểm tra tọa độ hợp lệ
                if x1 < 0 or y1 < 0 or x2 >= image.shape[1] or y2 >= image.shape[0] or x1 >= x2 or y1 >= y2:
                    continue
                        
                # Cắt ROI
                roi = image[y1:y2, x1:x2]
                image_aligned = image
                
                # Kiểm tra xem có phải là trường hợp đặc biệt của machine_code="F41" với allowed_values chứa "ON" và "OFF" không
                is_special_on_off_case = False
                allowed_values = []
                
                # Lấy thông tin allowed_values từ roi_info.json
                try:
                    roi_json_path = 'roi_data/roi_info.json'
                    if not os.path.exists(roi_json_path):
                        roi_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data/roi_info.json')
                    
                    with open(roi_json_path, 'r', encoding='utf-8') as f:
                        roi_info = json.load(f)

                    # Lấy machine_type từ machine_code
                    machine_type = get_machine_type(machine_code)

                    # Thử tìm allowed_values cho ROI hiện tại sử dụng machine_code
                    if (machine_code in roi_info.get("machines", {}) and 
                        "screens" in roi_info["machines"][machine_code] and 
                        screen_id in roi_info["machines"][machine_code]["screens"]):
                        roi_list = roi_info["machines"][machine_code]["screens"][screen_id]
                        for roi_item in roi_list:
                            if isinstance(roi_item, dict) and roi_item.get("name") == roi_name and "allowed_values" in roi_item:
                                allowed_values = roi_item["allowed_values"]
                                # Kiểm tra nếu allowed_values chứa "ON" và "OFF"
                                if "ON" in allowed_values and "OFF" in allowed_values:
                                    is_special_on_off_case = True
                                break
                    
                    # Nếu không tìm thấy với machine_code, thử với machine_type
                    if not is_special_on_off_case and machine_type:
                        if (machine_type in roi_info.get("machines", {}) and 
                            "screens" in roi_info["machines"][machine_type] and 
                            screen_id in roi_info["machines"][machine_type]["screens"]):
                            roi_list = roi_info["machines"][machine_type]["screens"][screen_id]
                            for roi_item in roi_list:
                                if isinstance(roi_item, dict) and roi_item.get("name") == roi_name and "allowed_values" in roi_item:
                                    allowed_values = roi_item["allowed_values"]
                                    # Kiểm tra nếu allowed_values chứa "ON" và "OFF"
                                    if "ON" in allowed_values and "OFF" in allowed_values:
                                        is_special_on_off_case = True
                                    break
                except:
                    pass
                
                # Nếu là trường hợp đặc biệt ON/OFF, phân tích màu sắc thay vì OCR
                if is_special_on_off_case and machine_type != "F1":
                    # Tách các kênh màu BGR
                    b, g, r = cv2.split(roi)
                    # Tính giá trị trung bình của kênh xanh dương và đỏ
                    avg_blue = np.mean(b)
                    avg_red = np.mean(r)
                    # Xác định kết quả dựa trên màu sắc chủ đạo
                    if avg_blue > avg_red:
                        best_text = "OFF"
                    else:
                        best_text = "ON"
                    
                    # Lưu ảnh ROI với kết quả color detection
                    # save_roi_image_with_result(roi, roi_name, original_filename, best_text, 1.0, best_text, is_text_result=True)
                    
                    results.append({
                        "roi_index": roi_name,
                        "text": best_text,
                        "confidence": 1.0,  # Đặt độ tin cậy là 100% vì dựa trên phân tích màu sắc
                        "has_text": True
                    })
                    
                    continue  # Chuyển sang ROI tiếp theo
                
                # Nếu là trường hợp đặc biệt ON/OFF và không phải machine_type F1, phân tích màu sắc thay vì OCR
                if is_special_on_off_case and machine_type != "F1":
                    # Tách các kênh màu BGR
                    b, g, r = cv2.split(roi)
                    # Tính giá trị trung bình của kênh xanh dương và đỏ
                    avg_blue = np.mean(b)
                    avg_red = np.mean(r)
                    # Xác định kết quả dựa trên màu sắc chủ đạo
                    if avg_blue > avg_red:
                        best_text = "OFF"
                    else:
                        best_text = "ON"
                    
                    # Lưu ảnh ROI với kết quả color detection
                    # save_roi_image_with_result(roi, roi_name, original_filename, best_text, 1.0, best_text, is_text_result=True)
                    
                    results.append({
                        "roi_index": roi_name,
                        "text": best_text,
                        "confidence": 1.0,  # Đặt độ tin cậy là 100% vì dựa trên phân tích màu sắc
                        "has_text": True
                    })
                    
                    continue  # Chuyển sang ROI tiếp theo
                
                # Tiền xử lý ROI cho OCR (cho các trường hợp thông thường)
                roi_processed, roi_quality_info = preprocess_roi_for_ocr(roi, i, original_filename, roi_name, image_aligned, x1, y1, x2, y2)
                
                # Thêm Gaussian Blur để cải thiện OCR (dựa trên test cho thấy kết quả tốt hơn)
                if roi_processed is not None:
                    roi_processed = cv2.GaussianBlur(roi_processed, (3, 3), 0)
                
                # Kiểm tra xem ảnh đã tiền xử lý có thành công không
                if roi_processed is None:
                    print(f"ROI {i} preprocessing failed")
                    continue
                            
                # Thực hiện OCR
                has_text = False
                best_text = ""
                best_confidence = 0
                original_value = ""
                
                # Thử OCR trên toàn bộ ROI
                try:
                                        # Specify the characters to read (digits only)
                    ocr_results = reader.readtext(roi_processed, 
                                                allowlist='0123456789.-ABCDEFGHIKLNORTUabcdefghiklnortu', 
                                                detail=1, 
                                                paragraph=False, 
                                                batch_size=1, 
                                                text_threshold=0.4,
                                                link_threshold=0.2, 
                                                low_text=0.3, 
                                                mag_ratio=2, 
                                                slope_ths=0.05,
                                                decoder='beamsearch'
                                                )
                    if ocr_results and len(ocr_results) > 0:
                        # Lấy kết quả có confidence cao nhất
                        best_result = max(ocr_results, key=lambda x: x[2])
                        best_text = best_result[1]
                        best_confidence = best_result[2]
                        original_value = best_text
                        has_text = True

                        
                        # Kiểm tra nếu kết quả ban đầu có dấu trừ ở đầu
                        has_negative_sign = best_text.startswith('-')
                        
                        # Kiểm tra nếu kết quả chỉ là 1 ký tự 'o' hoặc 'O' thì chuyển thành '0' luôn
                        if len(best_text) == 1 and best_text.upper() == 'O':
                            best_text = '0'
                        
                        # Kiểm tra và chuyển đổi chuỗi kết quả nếu có dạng số
                        # Xử lý đặc biệt cho trường hợp nghi ngờ là số (chuỗi có >= 2 ký tự và chứa nhiều O, U, I, l)
                        if len(best_text) >= 2:
                            # Đếm số lượng các ký tự dễ nhầm lẫn
                            chars_to_check = '01OUouIilC'
                            suspicious_chars_count = sum(1 for char in best_text if char in chars_to_check)
                            # Nếu có ít nhất 2 ký tự đáng ngờ và chiếm >= 30% chuỗi
                            if suspicious_chars_count >= 2 and suspicious_chars_count / len(best_text) >= 0.3:
                                # Kiểm tra các mẫu đặc biệt, như chuỗi "uuuu" hoặc "iuuu" có thể là "1000"
                                upper_text = best_text.upper()
                                
                                # Trường hợp đặc biệt: chuỗi chứa nhiều U liên tiếp (có thể là số 0 lặp lại)
                                upper_text_no_dot = upper_text.replace('.', '')
                                if re.search(r'[IUO0Q]{2}', upper_text_no_dot):
                                    temp_text = upper_text.replace('U', '0').replace('I', '1').replace('O', '0').replace('C','0').replace('Q','0')
                                    if temp_text.replace('.', '').isdigit():
                                        best_text = temp_text
                                        is_text_result = False
                                # Trường hợp đặc biệt khác
                                else:
                                    # Kiểm tra xem có ít nhất 60% ký tự là chữ cái đáng ngờ I, U, O
                                    digit_like_chars_count = sum(1 for char in upper_text if char in 'OUICL')
                                    if digit_like_chars_count / len(best_text) >= 0.7:
                                        # Chuyển đổi tất cả ký tự dễ nhầm lẫn thành số tương ứng
                                        cleaned_text = upper_text
                                        cleaned_text = cleaned_text.replace('O', '0').replace('U', '0').replace('Q', '0')
                                        cleaned_text = cleaned_text.replace('I', '1').replace('L', '1')
                                        cleaned_text = cleaned_text.replace('C', '0').replace('D', '0')
                                        
                                        # Loại bỏ khoảng trắng nếu kết quả là số
                                        cleaned_text = cleaned_text.replace(' ', '')
                                        
                                        # Kiểm tra nếu kết quả chỉ chứa chữ số
                                        if cleaned_text.isdigit():
                                            best_text = cleaned_text
                                            # Đánh dấu là kết quả số để không bị xử lý như text
                                            is_text_result = False
                        
                        # Đếm số lượng chữ số và chữ cái (loại trừ số 0 và chữ O)
                        digit_count = sum(1 for char in best_text if char.isdigit() and char != '0')
                        letter_count = sum(1 for char in best_text if char.isalpha() and char.upper() != 'O')
                        
                        # Kiểm tra nếu có nhiều chữ cái hơn chữ số
                        is_text_result = letter_count > digit_count
                        
                        # Thêm lại dấu trừ ở đầu nếu kết quả ban đầu có
                        if has_negative_sign and not best_text.startswith('-'):
                            best_text = '-' + best_text
                except:
                    pass
                
                # Kiểm tra xem ROI có key "allowed_values" không rỗng hay không
                has_allowed_values = False
                
                # Kiểm tra allowed_values từ roi_names (từ ROI coordinates)
                if roi_names and i < len(roi_names) and isinstance(roi_names[i], dict) and "allowed_values" in roi_names[i]:
                    allowed_values = roi_names[i].get("allowed_values", [])
                    if allowed_values and len(allowed_values) > 0:
                        has_allowed_values = True
                        # Buộc xử lý như text nếu có allowed_values
                        is_text_result = True
                
                # Kiểm tra allowed_values từ roi_info.json (quan trọng hơn)
                try:
                    roi_json_path = 'roi_data/roi_info.json'
                    if not os.path.exists(roi_json_path):
                        roi_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data/roi_info.json')
                    
                    with open(roi_json_path, 'r', encoding='utf-8') as f:
                        roi_info = json.load(f)
                    
                    # *** FIX: Sử dụng machine_type thay vì machine_code để lookup ***
                    machine_type_for_lookup = get_machine_type(machine_code)
                    
                    # Tìm allowed_values cho ROI hiện tại từ roi_info.json
                    if (machine_type_for_lookup in roi_info.get("machines", {}) and 
                        "screens" in roi_info["machines"][machine_type_for_lookup] and 
                        screen_id in roi_info["machines"][machine_type_for_lookup]["screens"]):
                        
                        roi_list = roi_info["machines"][machine_type_for_lookup]["screens"][screen_id]
                        
                        for roi_item in roi_list:
                            if isinstance(roi_item, dict) and roi_item.get("name") == roi_name and "allowed_values" in roi_item:
                                allowed_values_from_json = roi_item["allowed_values"]
                                if allowed_values_from_json and len(allowed_values_from_json) > 0:
                                    has_allowed_values = True
                                    # Buộc xử lý như text nếu có allowed_values
                                    is_text_result = True
                                break
                except:
                    pass
                
                # Xử lý kết quả OCR dựa vào loại kết quả (số hoặc chữ)
                formatted_text = best_text
                
                # Nếu là kết quả chủ yếu là chữ hoặc ROI có allowed_values, xử lý như text
                if has_text and (is_text_result or has_allowed_values):
                    best_text = best_text.replace('0', 'O').replace('1', 'I').replace('2', 'Z').replace('3', 'E').replace('4', 'A').replace('5', 'S').replace('6', 'G').replace('7', 'T').replace('8', 'B').replace('9', 'P')
                    best_text = best_text.upper()
                    # Thêm kết quả cho ROI này (không có original_value cho kết quả text)
                    if len(best_text) == 1:
                        best_text = best_text.replace('O', '0').replace('I', '1').replace('C','0').replace('S','5').replace('G','6').replace('A','4').replace('H','8').replace('L','1').replace('T','7').replace('U','0').replace('E','3').replace('Z','2').replace('Q','0')
                    
                    # Kiểm tra nếu ROI này có allowed_values trong roi_info.json
                    try:
                        roi_json_path = 'roi_data/roi_info.json'
                        if not os.path.exists(roi_json_path):
                            roi_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data/roi_info.json')
                        
                        with open(roi_json_path, 'r', encoding='utf-8') as f:
                            roi_info = json.load(f)
                        
                        # *** FIX: Sử dụng machine_type thay vì machine_code để lookup ***
                        machine_type_for_lookup = get_machine_type(machine_code)
                        
                        # Tìm allowed_values cho ROI hiện tại
                        allowed_values = None
                        
                        if (machine_type_for_lookup in roi_info.get("machines", {}) and 
                            "screens" in roi_info["machines"][machine_type_for_lookup] and 
                            screen_id in roi_info["machines"][machine_type_for_lookup]["screens"]):
                            
                            roi_list = roi_info["machines"][machine_type_for_lookup]["screens"][screen_id]
                            
                            for roi_item in roi_list:
                                if isinstance(roi_item, dict) and roi_item.get("name") == roi_name and "allowed_values" in roi_item:
                                    allowed_values = roi_item["allowed_values"]
                                    break
                        
                        # Sử dụng hàm tối ưu mới để tìm best match
                        if allowed_values and len(allowed_values) > 0:
                            best_match, match_score, match_method = find_best_allowed_value_match(
                                best_text, allowed_values, roi_name
                            )
                            
                            if best_match:
                                best_text = best_match
                            else:
                                best_text = allowed_values[0]
                    except Exception as e:
                        print(f"Error checking allowed_values for ROI {roi_name}: {str(e)}")
                        traceback.print_exc()
                    
                    results.append({
                        "roi_index": roi_name,
                        "text": best_text,
                        "confidence": best_confidence,
                        "has_text": has_text,
                        "original_value": original_value
                    })
                    print(f"Added text result for ROI {i} ({roi_name}): '{best_text}'")
                    
                    # Lưu ảnh ROI với kết quả text detection
                    # save_roi_image_with_result(roi, roi_name, original_filename, best_text, best_confidence, original_value, is_text_result=True)
                    
                    continue  # Bỏ qua phần xử lý định dạng số tiếp theo
                
                # Nếu kết quả chủ yếu là số, xử lý theo định dạng decimal_places
                is_negative = best_text.startswith('-')
                best_text = best_text.upper()
                print(best_text)
                best_text = best_text.replace('O', '0').replace('I', '1').replace('C','0').replace('S','5').replace('G','6').replace('B','8').replace('T','7').replace('L','1').replace('H','8').replace('A','4').replace('E','3').replace('Z','2').replace('U','0')
                
                
                # Xử lý kết quả OCR có khoảng trắng giữa các số (ví dụ: "1 3")
                if ' ' in best_text and all(c.isdigit() or c == ' ' or c == '.' or c == '-' for c in best_text):
                    print(f"Found spaces in numeric result: '{best_text}'. Removing spaces...")
                    best_text = best_text.replace(' ', '')
                    print(f"After removing spaces: '{best_text}'")
                
                if '-' in best_text[1:]:
                    best_text = best_text[:-1] + best_text[-1].replace('-', '')
                
                # Kiểm tra lại sau khi đã xóa khoảng trắng
                if has_text and re.match(r'^-?\d+\.?\d*$', best_text):
                    try:
                        # Kiểm tra xem có cấu hình cho ROI này không
                        best_text = best_text[1:] if is_negative else best_text
                        
                        # Lấy machine_type từ machine_code
                        machine_type = get_machine_type(machine_code)
                        print(f"Getting decimal places config for machine_type={machine_type}, screen_id={screen_id}, roi_name={roi_name}")
                        
                        # Áp dụng decimal_places trước khi chuyển sang ROI tiếp theo
                        if (machine_type in decimal_places_config and 
                            screen_id in decimal_places_config[machine_type] and 
                            roi_name in decimal_places_config[machine_type][screen_id]):
                            
                            decimal_places = int(decimal_places_config[machine_type][screen_id][roi_name])
                            print(f"Found decimal places config for ROI {roi_name}: {decimal_places}")
                            
                            # Xử lý các trường hợp khác nhau dựa trên decimal_places
                            if decimal_places == 0:
                                # Nếu decimal_places là 0, giữ lại tất cả các chữ số nhưng bỏ dấu chấm
                                formatted_text = str(best_text).replace('.', '')
                                formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                print(f"Removed decimal point for ROI {roi_name}: {formatted_text}")
                            else:
                                # Đếm số chữ số thập phân hiện tại
                                current_decimal_places = 0
                                if '.' in best_text:
                                    dec_part = best_text.split('.')[1]
                                    current_decimal_places = len(dec_part)
                                
                                # Nếu số chữ số thập phân hiện tại bằng đúng số chữ số thập phân cần có
                                if current_decimal_places == decimal_places:
                                    # Giữ nguyên số
                                    formatted_text = best_text
                                    formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                    print(f"Already correct format for ROI {roi_name}: {best_text}")
                                else:
                                    # Xử lý khi có dấu thập phân
                                    if '.' in best_text:
                                        int_part, dec_part = best_text.split('.')
                                        
                                        # Kết hợp phần nguyên và phần thập phân thành một chuỗi không có dấu chấm
                                        all_digits = int_part + dec_part
                                        
                                        # Đặt dấu chấm vào vị trí thích hợp theo decimal_places
                                        if decimal_places > 0:
                                            if len(all_digits) <= decimal_places:
                                                # Thêm số 0 phía trước và đặt dấu chấm sau số 0 đầu tiên
                                                padded_str = all_digits.zfill(decimal_places)
                                                formatted_text = f"0.{padded_str}"
                                                formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                            else:
                                                # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                                insert_pos = len(all_digits) - decimal_places
                                                formatted_text = f"{all_digits[:insert_pos]}.{all_digits[insert_pos:]}"
                                                formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                        else:
                                            # Nếu decimal_places = 0, bỏ dấu chấm
                                            formatted_text = all_digits
                                            formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                    else:
                                        # Không có dấu chấm (số nguyên)
                                        num_str = str(best_text)
                                    
                                    # Thêm phần thập phân nếu cần
                                    if decimal_places > 0:
                                        # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                        if len(num_str) <= decimal_places:
                                            # Nếu số chữ số ít hơn hoặc bằng decimal_places, thêm số 0 ở đầu
                                            padded_str = num_str.zfill(decimal_places)
                                            formatted_text = f"0.{padded_str}"
                                        else:
                                            # Đặt dấu chấm vào vị trí thích hợp
                                            insert_pos = len(num_str) - decimal_places
                                            formatted_text = f"{num_str[:insert_pos]}.{num_str[insert_pos:]}"
                                        
                                        formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                        print(f"Formatted integer value {num_str} with decimal_places={decimal_places}: {formatted_text}")
                                    else:
                                        # Giữ nguyên số nguyên nếu không cần thập phân
                                        formatted_text = num_str
                                        formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                    print(f"Formatted value for ROI {roi_name}: Original: {best_text}, Formatted: {formatted_text}")
                                
                                # Cập nhật best_text cho các bước xử lý tiếp theo
                                best_text = formatted_text
                        else:
                            # Thêm xử lý đặc biệt cho "Machine OEE" nếu không tìm thấy trong cấu hình
                            if roi_name == "Machine OEE":
                                decimal_places = 2  # Áp dụng 2 chữ số thập phân cho Machine OEE theo yêu cầu
                                print(f"Special case: Applying {decimal_places} decimal places for Machine OEE")
                                
                                # Xử lý định dạng số như các trường hợp khác
                                num_str = str(best_text)
                                if len(num_str) <= decimal_places:
                                    padded_str = num_str.zfill(decimal_places)
                                    formatted_text = f"0.{padded_str}"
                                else:
                                    insert_pos = len(num_str) - decimal_places
                                    formatted_text = f"{num_str[:insert_pos]}.{num_str[insert_pos:]}"
                                
                                formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                print(f"Special handling for Machine OEE: Formatted value: {formatted_text}")
                                best_text = formatted_text
                            else:
                                # Nếu không có cấu hình decimal_places, giữ nguyên giá trị
                                formatted_text = best_text
                                formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                                print(f"No decimal places config found for {machine_type}/{screen_id}/{roi_name}. Keeping original value.")
                    except Exception as e:
                        print(f"Error applying decimal places format for ROI {roi_name}: {str(e)}")
                        formatted_text = best_text
                        formatted_text = '-' + str(formatted_text) if is_negative else str(formatted_text)
                else:
                    # Nếu không phải là số, giữ nguyên text
                    formatted_text = best_text
                
                # Kiểm tra nếu ROI có chứa "working hours" trong tên 
                if "working hours" in roi_name.lower():
                    # Loại bỏ tất cả các ký tự không phải số
                    digits_only = re.sub(r'[^0-9]', '', formatted_text)
                    
                    # Kiểm tra xem chuỗi có đủ số để định dạng không
                    if len(digits_only) >= 2:
                        # Thêm dấu ":" sau mỗi 2 ký tự từ phải sang trái
                        result = ""
                        for i in range(len(digits_only) - 1, -1, -1):
                            result = digits_only[i] + result
                            if i > 0 and (len(digits_only) - i) % 2 == 0:
                                result = ":" + result
                        
                        formatted_text = result
                        print(f"Reformatted working hours from original '{original_value}' to '{formatted_text}'")
                
                # Xử lý dấu "-" không ở vị trí đầu tiên
                if "-" in formatted_text[1:]:
                    formatted_text = formatted_text[0] + formatted_text[1:].replace('-', '.')
                    print(f"Replaced dash in middle with dot: {formatted_text}")
                
                # Thêm kết quả cho ROI này
                results.append({
                    "roi_index": roi_name,
                    "text": formatted_text,  # Trả về text đã định dạng theo quy định số chữ số thập phân
                    "confidence": best_confidence,
                    "has_text": has_text,
                    "original_value": original_value
                })
                print(f"Added result for ROI {i} ({roi_name}): Original: '{best_text}', Formatted: '{formatted_text}'")
                
                # Lưu ảnh ROI với kết quả numeric detection sử dụng hàm mới
                # save_roi_image_with_result(roi, roi_name, original_filename, formatted_text, best_confidence, original_value, is_text_result=False)
                
                if best_confidence < 0.3 or (roi_quality_info is not None and ('low_contrast' in roi_quality_info['issues'] or roi_quality_info.get('has_moire', False))):
                    print(f"Confidence is below threshold or image has low contrast. Trying alternative approach with connected component analysis...")
                    
                    # Thực hiện phân tích thành phần liên kết như trong color_detector.py
                    # 1. Chuyển ảnh ROI sang grayscale nếu chưa phải
                    if len(roi.shape) > 2:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    else:
                        roi_gray = roi.copy()
                    
                    # 2. Làm mờ nhẹ để giảm nhiễu bề mặt
                    roi_blur = cv2.GaussianBlur(roi_gray, (7,7), 0)
                    
                    # 3. Áp dụng adaptive threshold để tách nền thay đổi độ sáng
                    roi_th = cv2.adaptiveThreshold(
                        roi_blur, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,  # Đảo ngược để số là màu trắng, nền đen
                        blockSize=11, C=2
                    )
                    
                    # 4. Sử dụng connected component analysis để lọc dựa trên kích thước
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_th, connectivity=8)
                    
                    # Tạo một mask trống để giữ lại các số
                    digit_mask = np.zeros_like(roi_th)
                    
                    # Lọc các thành phần dựa trên kích thước (có thể điều chỉnh các giá trị này)
                    min_area = 50      # Diện tích tối thiểu của số
                    max_area = 2000    # Diện tích tối đa của số
                    min_width = 5      # Chiều rộng tối thiểu
                    min_height = 10    # Chiều cao tối thiểu
                    max_width = 100    # Chiều rộng tối đa
                    max_height = 100   # Chiều cao tối đa
                    aspect_ratio_min = 0.2  # Tỷ lệ chiều rộng/chiều cao tối thiểu
                    aspect_ratio_max = 5.0  # Tỷ lệ chiều rộng/chiều cao tối đa
                    
                    # Bỏ qua label 0 vì đó là nền (background)
                    for label in range(1, num_labels):
                        # Lấy thông tin của component
                        x, y, w, h, area = stats[label]
                        
                        # Tính toán tỷ lệ khung hình
                        aspect_ratio = w / h if h > 0 else 0
                        
                        # Kiểm tra các điều kiện để xác định là số
                        if (min_area < area < max_area and 
                            min_width < w < max_width and 
                            min_height < h < max_height and
                            aspect_ratio_min < aspect_ratio < aspect_ratio_max):
                            # Đây có khả năng là số, giữ lại trong mask
                            digit_mask[labels == label] = 255
                    
                    # Lưu ảnh digit_mask để debug
                    processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_roi')
                    base_filename = os.path.splitext(original_filename)[0]
                    digit_mask_filename = f"{base_filename}_{roi_name}_digit_mask.png"
                    digit_mask_path = os.path.join(processed_folder, digit_mask_filename)
                    # cv2.imwrite(digit_mask_path, digit_mask)
                    # print(f"Saved digit mask to: {digit_mask_path}")
                    
                                        # 5. Thực hiện OCR trên mask đã tạo với thông số tối ưu
                    retry_results = reader.readtext(digit_mask, 
                                                  allowlist='0123456789.-ABCDEFGHIKLNORTUabcdefghiklnortu', 
                                                  detail=1, 
                                                  paragraph=False, 
                                                  batch_size=1, 
                                                  text_threshold=0.4,
                                                  link_threshold=0.2, 
                                                  low_text=0.3, 
                                                  mag_ratio=2, 
                                                  slope_ths=0.05,
                                                  decoder='beamsearch'
                                                  )
                    
                    # Kiểm tra nếu có kết quả OCR
                    if retry_results and len(retry_results) > 0:
                        # Tìm kết quả có confidence cao nhất
                        best_retry_result = max(retry_results, key=lambda x: x[2])
                        retry_text = best_retry_result[1]  # Text
                        retry_confidence = best_retry_result[2]  # Confidence
                        
                        # Chuyển đổi sang chữ hoa và kiểm tra pattern
                        upper_text = retry_text.upper()
                        upper_text_no_dot = upper_text.replace('.', '')
                        print(f"upper_text: '{upper_text}', upper_text_no_dot: '{upper_text_no_dot}'")
                        print(f"Pattern matched: {re.search(r'[IUO0Q]{2}', upper_text_no_dot)}")
                        
                        if re.search(r'[IUO0Q]{2}', upper_text_no_dot):
                            temp_text = upper_text.replace('U', '0').replace('I', '1').replace('O', '0').replace('C','0').replace('Q','0')
                            if temp_text.replace('.', '').isdigit():
                                print(f"Pattern with repeated U/I detected. Converting '{retry_text}' to '{temp_text}'")
                                retry_text = temp_text
                            # Trường hợp đặc biệt khác
                            else:
                                # Kiểm tra xem có ít nhất 60% ký tự là chữ cái đáng ngờ I, U, O
                                digit_like_chars_count = sum(1 for char in upper_text if char in 'OUICL')
                                if digit_like_chars_count / len(retry_text) >= 0.7:
                                    # Chuyển đổi tất cả ký tự dễ nhầm lẫn thành số tương ứng
                                    cleaned_text = upper_text
                                    cleaned_text = cleaned_text.replace('O', '0').replace('U', '0').replace('Q', '0')
                                    cleaned_text = cleaned_text.replace('I', '1').replace('L', '1')
                                    cleaned_text = cleaned_text.replace('C', '0').replace('D', '0')
                                    
                                    # Loại bỏ khoảng trắng nếu kết quả là số
                                    cleaned_text = cleaned_text.replace(' ', '')
                                    
                                    # Kiểm tra nếu kết quả chỉ chứa chữ số
                                    if cleaned_text.isdigit():
                                        print(f"Likely numeric value detected. Converting '{retry_text}' to '{cleaned_text}'")
                                        retry_text = cleaned_text
                        
                        print(f"Best retry result: '{retry_text}' with confidence {retry_confidence}")
                        
                        # Lưu ảnh retry OCR result với text tốt nhất
                        processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_roi')
                        base_filename = os.path.splitext(original_filename)[0]
                        retry_result_filename = f"{base_filename}_{roi_name}_step6_retry_ocr_result.png"
                        retry_result_path = os.path.join(processed_folder, retry_result_filename)
                        
                        # Tạo ảnh với retry text overlay trên digit mask
                        retry_result_img = digit_mask.copy()
                        if len(retry_result_img.shape) == 2:  # Grayscale
                            retry_result_img = cv2.cvtColor(retry_result_img, cv2.COLOR_GRAY2BGR)
                        
                        # Vẽ retry text lên ảnh
                        # font_scale = max(0.5, min(retry_result_img.shape[0], retry_result_img.shape[1]) / 100)
                        # cv2.putText(retry_result_img, f"RETRY: '{retry_text}' ({retry_confidence:.2f})", 
                        #           (5, int(retry_result_img.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
                        #           font_scale, (0, 255, 255), 2)  # Yellow color for retry
                        # # cv2.imwrite(retry_result_path, retry_result_img)
                        # cv2.imwrite(retry_result_path, retry_result_img)
                        # print(f"Saved retry OCR result image to: {retry_result_path}")
                        
                        # Xử lý khoảng trắng giữa các số (tương tự như xử lý trên best_text)
                        retry_text = retry_text.upper()
                        retry_text = retry_text.replace('O', '0').replace('I', '1').replace('C','0').replace('S','5').replace('G','6').replace('B','8')
                        
                        # Thêm lại dấu trừ ở đầu nếu kết quả ban đầu có
                        if has_negative_sign and not retry_text.startswith('-'):
                            retry_text = '-' + retry_text
                            print(f"Added negative sign back to retry result: '{retry_text}'")
                        
                        # Xử lý định dạng working hours trước khi loại bỏ khoảng trắng
                        if "working hours" in roi_name.lower():
                            # Loại bỏ tất cả các ký tự không phải số
                            digits_only = re.sub(r'[^0-9]', '', retry_text)
                            
                            # Kiểm tra xem chuỗi có đủ số để định dạng không
                            if len(digits_only) >= 2:
                                # Thêm dấu ":" sau mỗi 2 ký tự từ phải sang trái
                                result = ""
                                for i in range(len(digits_only) - 1, -1, -1):
                                    result = digits_only[i] + result
                                    if i > 0 and (len(digits_only) - i) % 2 == 0:
                                        result = ":" + result
                                
                                retry_text = result
                                print(f"Reformatted retry working hours from original to '{retry_text}'")
                        
                        # Xử lý dấu "-" không ở vị trí đầu tiên
                        if "-" in retry_text[1:]:
                            retry_text = retry_text[0] + retry_text[1:].replace('-', '.')
                            print(f"Replaced dash in middle with dot in retry_text: {retry_text}")
                        
                        # Kiểm tra allowed_values cho retry_text, tương tự như đã làm với best_text
                        retry_matched_with_allowed_values = False
                        try:
                            # Kiểm tra xem file ROIs JSON có tồn tại
                            roi_json_path = 'roi_data/roi_info.json'
                            if not os.path.exists(roi_json_path):
                                roi_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data/roi_info.json')
                            
                            if os.path.exists(roi_json_path):
                                with open(roi_json_path, 'r', encoding='utf-8') as f:
                                    roi_info = json.load(f)
                                
                                # Lấy machine_type từ machine_code
                                machine_type_for_roi = get_machine_type(machine_code)
                                print(f"Using machine_code={machine_code}, machine_type={machine_type_for_roi} to find allowed_values")
                                
                                # Tìm allowed_values cho ROI hiện tại
                                if (machine_code in roi_info.get("machines", {}) and 
                                    "screens" in roi_info["machines"][machine_code] and 
                                    screen_id in roi_info["machines"][machine_code]["screens"]):
                                    
                                    roi_list = roi_info["machines"][machine_code]["screens"][screen_id]
                                    allowed_values = None
                                    
                                    for roi_item in roi_list:
                                        if isinstance(roi_item, dict) and roi_item.get("name") == roi_name and "allowed_values" in roi_item:
                                            allowed_values = roi_item["allowed_values"]
                                            break
                                    
                                    # Sử dụng hàm tối ưu mới để tìm best match cho retry_text
                                    if allowed_values and len(allowed_values) > 0:
                                        print(f"Found allowed_values for retry ROI {roi_name}: {allowed_values}")
                                        
                                        retry_best_match, retry_match_score, retry_match_method = find_best_allowed_value_match(
                                            retry_text, allowed_values, f"{roi_name}_retry"
                                        )
                                        
                                        if retry_best_match:
                                            print(f"[OK] RETRY MATCHED: '{retry_text}' -> '{retry_best_match}' (score: {retry_match_score:.3f}, method: {retry_match_method})")
                                            retry_text = retry_best_match
                                            # Cập nhật kết quả ngay lập tức và bỏ qua các xử lý tiếp theo
                                            results[-1]["text"] = retry_text
                                            results[-1]["confidence"] = retry_confidence
                                            results[-1]["has_text"] = True
                                            results[-1]["original_value"] = retry_text
                                            best_confidence = retry_confidence
                                            retry_matched_with_allowed_values = True
                                            print(f"[OK] Updated result with allowed value match: '{retry_text}'")
                                        else:
                                            print(f"[ERROR] NO SUITABLE RETRY MATCH FOUND. Using first allowed value: '{allowed_values[0]}'")
                                            retry_text = allowed_values[0]
                                            results[-1]["text"] = retry_text
                                            results[-1]["confidence"] = retry_confidence
                                            results[-1]["has_text"] = True
                                            results[-1]["original_value"] = retry_text
                                            best_confidence = retry_confidence
                                            retry_matched_with_allowed_values = True
                                            print(f"[OK] Updated result with first allowed value: '{retry_text}'")
                        except Exception as e:
                            print(f"Error checking allowed_values for retry ROI {roi_name}: {str(e)}")
                        
                        # Nếu chưa match với allowed_values hoặc không có allowed_values,
                        # kiểm tra xem retry result có tốt hơn original result không
                        if not retry_matched_with_allowed_values:
                            # Chấp nhận retry result nếu confidence > 0.3 (theo yêu cầu user)
                            # hoặc nếu retry confidence tốt hơn original
                            if retry_confidence > 0.3 or retry_confidence > best_confidence:
                                print(f"[OK] Accepting re0812try result: confidence {retry_confidence:.4f} (threshold: 0.3, original: {best_confidence:.4f})")
                                
                                # Áp dụng định dạng theo decimal_places nếu kết quả là số và có cấu hình
                                formatted_retry_text = retry_text
                                
                                # Xử lý trường hợp retry_text chỉ có các ký tự số nhưng không có dấu chấm thập phân
                                is_numeric = re.match(r'^-?\d+$', retry_text) is not None
                                has_decimal_point = '.' in retry_text
                                
                                if (is_numeric or re.match(r'^-?\d+\.?\d*$', retry_text)) and (
                                    machine_type in decimal_places_config and 
                                    screen_id in decimal_places_config[machine_type] and 
                                    roi_name in decimal_places_config[machine_type][screen_id]):
                                    
                                    try:
                                        is_negative = retry_text.startswith('-')
                                        clean_text = retry_text[1:] if is_negative else retry_text
                                        decimal_places = int(decimal_places_config[machine_type][screen_id][roi_name])
                                        print(f"Getting decimal places config for machine_type={machine_type}, screen_id={screen_id}, roi_name={roi_name}")
                                        print(f"Found decimal places config for ROI {roi_name}: {decimal_places}")
                                        
                                        # Xử lý tương tự như phần xử lý decimal_places ở trên
                                        if decimal_places == 0:
                                            # Nếu decimal_places là 0, bỏ dấu chấm
                                            formatted_retry_text = clean_text.replace('.', '')
                                            print(f"Removed decimal point for ROI {roi_name}: {formatted_retry_text}")
                                        else:
                                            # Xử lý vị trí dấu thập phân
                                            if '.' in clean_text:
                                                int_part, dec_part = clean_text.split('.')
                                                
                                                # Kết hợp phần nguyên và phần thập phân thành một chuỗi không có dấu chấm
                                                all_digits = int_part + dec_part
                                                
                                                # Đặt dấu chấm vào vị trí thích hợp theo decimal_places
                                                if decimal_places > 0:
                                                    if len(all_digits) <= decimal_places:
                                                        # Thêm số 0 phía trước và đặt dấu chấm sau số 0 đầu tiên
                                                        padded_str = all_digits.zfill(decimal_places)
                                                        formatted_retry_text = f"0.{padded_str}"
                                                    else:
                                                        # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                                        insert_pos = len(all_digits) - decimal_places
                                                        formatted_retry_text = f"{all_digits[:insert_pos]}.{all_digits[insert_pos:]}"
                                                else:
                                                    # Nếu decimal_places = 0, bỏ dấu chấm
                                                    formatted_retry_text = all_digits
                                            else:
                                                # Không có dấu chấm (số nguyên)
                                                num_str = clean_text
                                                
                                                # Thêm phần thập phân nếu cần
                                                if decimal_places > 0:
                                                    # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                                    if len(num_str) <= decimal_places:
                                                        # Nếu số chữ số ít hơn hoặc bằng decimal_places, thêm số 0 ở đầu
                                                        padded_str = num_str.zfill(decimal_places)
                                                        formatted_retry_text = f"0.{padded_str}"
                                                    else:
                                                        # Đặt dấu chấm vào vị trí thích hợp
                                                        insert_pos = len(num_str) - decimal_places
                                                        formatted_retry_text = f"{num_str[:insert_pos]}.{num_str[insert_pos:]}"
                                                    
                                                    print(f"Formatted integer retry value {num_str} with decimal_places={decimal_places}: {formatted_retry_text}")
                                                else:
                                                    # Giữ nguyên số nguyên nếu không cần thập phân
                                                    formatted_retry_text = num_str
                                        
                                        # Thêm dấu âm nếu cần
                                        if is_negative:
                                            formatted_retry_text = f"-{formatted_retry_text}"
                                            
                                        print(f"Formatted retry text: Original: '{retry_text}', Formatted: '{formatted_retry_text}'")
                                    except Exception as e:
                                        print(f"Error formatting retry text: {str(e)}")
                                
                                # Kiểm tra nếu ROI có chứa "working hours" trong tên 
                                # và kết quả đọc được là định dạng kiểu số.số.số
                                if "working hours" in roi_name.lower() and re.match(r'^\d+\.\d+\.\d+$', formatted_retry_text):
                                    # Chuyển đổi từ định dạng số.số.số sang số:số:số
                                    formatted_retry_text = formatted_retry_text.replace('.', ':').replace(' ', ':').replace('-', ':')
                                
                                # Xử lý dấu "-" không ở vị trí đầu tiên
                                if "-" in formatted_retry_text[1:]:
                                    formatted_retry_text = formatted_retry_text[0] + formatted_retry_text[1:].replace('-', '.')
                                
                                # Cập nhật kết quả với formatted text
                                results[-1]["text"] = formatted_retry_text.replace('C','0')
                                results[-1]["confidence"] = retry_confidence
                                results[-1]["has_text"] = True
                                results[-1]["original_value"] = retry_text
                                best_confidence = retry_confidence
                                print(f"[OK] Updated result with retry OCR: '{formatted_retry_text}' (confidence: {retry_confidence:.4f})")
                            else:
                                print(f"Keeping original result: retry confidence {retry_confidence:.4f} not meeting threshold 0.3")
                    else:
                        print(f"[ERROR] No retry OCR results found on digit mask")
                    
                    # Nếu confidence vẫn thấp dưới 0.1, trả về mảng rỗng
                    if best_confidence < 0.1:
                        return []
                # Kiểm tra độ tin cậy của OCR
                # if best_confidence < 0.3:  # Nếu độ tin cậy < 30%
                #     print(f"Warning: Low confidence ({best_confidence:.2f}) for ROI {roi_name}, text: '{best_text}'")
                
            except:
                continue
        
        return results
    
    except:
        # Trả về kết quả giả khi không thể thực hiện OCR (để testing)
        mock_results = []
        for i in range(len(roi_coordinates)):
            roi_name = roi_names[i] if roi_names and i < len(roi_names) else f"ROI_{i}"
            mock_results.append({
                "roi_index": roi_name,
                "text": "OCR_ERROR",
                "confidence": 0,
                "has_text": False,
                "original_value": ""
            })
        return mock_results

# Thêm helper function để xóa tất cả files trong processed_roi folder
def clear_processed_roi_folder():
    """Xóa tất cả các file trong thư mục processed_roi để đảm bảo không có file cũ"""
    try:
        processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_roi')
        if os.path.exists(processed_folder):
            # Lấy danh sách tất cả files trong folder
            files = os.listdir(processed_folder)
            deleted_count = 0
            
            for filename in files:
                file_path = os.path.join(processed_folder, filename)
                if os.path.isfile(file_path):  # Chỉ xóa files, không xóa folders
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Warning: Could not delete {file_path}: {str(e)}")
            
            if deleted_count > 0:
                print(f"🗑️ Cleared {deleted_count} old files from processed_roi folder")
            else:
                print("[FILE] Processed_roi folder is already empty")
        else:
            print("[FILE] Processed_roi folder does not exist yet")
    except Exception as e:
        print(f"Warning: Error clearing processed_roi folder: {str(e)}")

# Sửa lại route upload_image để sử dụng template reference từ thư mục mới
@app.route('/api/images', methods=['POST'])
def upload_image():
    """API để tải lên ảnh và thực hiện OCR trên các vùng ROI được định nghĩa - OPTIMIZED VERSION"""
    
    # Kiểm tra request nhanh
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Đọc ảnh trực tiếp từ memory thay vì lưu file
    filename = secure_filename(file.filename)
    
    # Đọc ảnh trực tiếp từ stream để tránh I/O
    file_bytes = np.frombuffer(file.read(), np.uint8)
    uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if uploaded_image is None:
        return jsonify({"error": "Could not read uploaded image"}), 400
    
    try:
        # Lấy area và machine_code từ form data nếu có
        area = request.form.get('area', None)
        machine_code = request.form.get('machine_code', None)
        
        # Gọi hàm smart detection với tham số area và machine_code
        detection_result = auto_detect_machine_and_screen(uploaded_image, area=area, machine_code=machine_code)
        
        if detection_result is None:
            return jsonify({
                "error": "Could not automatically detect machine and screen from image. Please ensure the image contains a clear HMI screen."
            }), 400
        
        # Lấy thông tin từ kết quả detection
        machine_code = detection_result['machine_code']
        machine_type = detection_result['machine_type']
        area = detection_result['area']
        machine_name = detection_result['machine_name']
        screen_id = detection_result['screen_id']
        screen_numeric_id = detection_result['screen_numeric_id']
        template_path = detection_result['template_path']
        similarity_score = detection_result['similarity_score']
        
        # THIẾT LẬP TEMPLATE PATH - Đây là điều thiếu!
        if template_path is None:
            template_path = get_reference_template_path(machine_type, screen_id)
            print(f"🔧 DEBUG: Auto-detected template_path: {template_path}")
        else:
            print(f"🔧 DEBUG: Using provided template_path: {template_path}")
            
        # Thêm mới: Phát hiện màn hình HMI
        hmi_detected = False
        visualization_path = None
        hmi_refined_filename = None
        hmi_screen, visualization, roi_coords = detect_hmi_screen(uploaded_image)
        
        # Lưu trữ thông tin phát hiện HMI
        hmi_detection_info = {
            "hmi_detected": False,
            "hmi_image": None,
            "hmi_refined_filename": None,
            "visualization": None
        }
        
        if hmi_screen is not None:
            hmi_detected = True
            uploaded_image = hmi_screen
            
            # Lưu ảnh HMI refined
            # hmi_refined_filename = f"hmi_refined_{filename}"
            # hmi_refined_path = os.path.join(app.config['HMI_REFINED_FOLDER'], hmi_refined_filename)
            # cv2.imwrite(hmi_refined_path, hmi_screen)
            
            # Cập nhật thông tin phát hiện HMI
            hmi_detection_info = {
                "hmi_detected": True,
                "hmi_image": None,
                "hmi_refined_filename": hmi_refined_filename,
                "visualization": None
            }
        else:
            hmi_detection_info = {
                "hmi_detected": False,
                "hmi_image": None,
                "hmi_refined_filename": None,
                "visualization": None
            }
        
        # Lấy ROI coordinates và tên ROI dựa trên machine_type và screen_id đã phát hiện
        roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_id, machine_type)
        
        if not roi_coordinates or len(roi_coordinates) == 0:
            return jsonify({
                "error": f"No ROI coordinates found for machine_code={machine_code}, screen_id={screen_id}, machine_type={machine_type}"
            }), 404
        
        # Tiền xử lý ảnh với căn chỉnh nếu có template
        image = uploaded_image  # Sử dụng ảnh gốc hoặc ảnh HMI đã phát hiện
        if template_path:
            # Sử dụng cached template image
            template_img = get_template_image_cached(template_path)
            if template_img is not None:
                # Căn chỉnh ảnh nhanh
                aligner = ImageAligner(template_img, image)
                aligned_image = aligner.align_images()
                image = aligned_image
        
        # Thực hiện OCR trên các vùng ROI với phiên bản tối ưu
        ocr_results = perform_ocr_on_roi_optimized(
            image, 
            roi_coordinates, 
            filename, 
            template_path,
            roi_names,
            machine_code,
            screen_id
        )
        
        # Tạo cấu trúc dữ liệu giống với file OCR result
        result_data = {
            "filename": filename,
            "machine_code": machine_code,
            "area": area,
            "machine_name": machine_name,
            "screen_id": screen_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "template_path": template_path if template_path else None,
            "results": ocr_results,
            "hmi_detection": hmi_detection_info,
            "detected_screen": {
                "screen_id": screen_id,
                "screen_numeric_id": screen_numeric_id,
                "similarity_score": similarity_score,
                "machine_type": machine_type
            }
        }
        
        # Lưu kết quả vào file JSON bất đồng bộ để không chặn response
        def save_json_async():
            try:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                base_filename = os.path.splitext(filename)[0]
                json_filename = f"ocr_result_{timestamp_str}_{machine_type}_{screen_id}_{base_filename}_{machine_code}_{screen_id}.json"
                json_file_path = os.path.join(app.config['OCR_RESULTS_FOLDER'], json_filename)
                os.makedirs(app.config['OCR_RESULTS_FOLDER'], exist_ok=True)
                
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(result_data, json_file, ensure_ascii=False, indent=2)
                print(f"[FILE] OCR result saved to: {json_file_path}")
            except Exception as e:
                print(f"[ERROR] Error saving OCR result: {str(e)}")
        
        # Submit to thread pool để save async
        _ocr_thread_pool.submit(save_json_async)
        
        # Trả về kết quả ngay lập tức
        return jsonify(result_data), 201
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

# API 2: Lấy danh sách hình ảnh
@app.route('/api/images', methods=['GET'])
def get_images():
    """Trả về kết quả OCR gần nhất giống cấu trúc của file OCR result"""
    try:
        # Tìm file OCR result mới nhất
        ocr_results = []
        for filename in os.listdir(app.config['OCR_RESULTS_FOLDER']):
            if filename.startswith("ocr_result_") and filename.endswith(".json"):
                file_path = os.path.join(app.config['OCR_RESULTS_FOLDER'], filename)
                ocr_results.append({
                    'path': file_path,
                    'filename': filename,
                    'modified_time': os.path.getmtime(file_path)
                })
        
        if not ocr_results:
            return jsonify({
                "error": "No OCR results found"
            }), 404
        
        # Sắp xếp theo thời gian sửa đổi giảm dần (mới nhất trước)
        ocr_results.sort(key=lambda x: x['modified_time'], reverse=True)
        
        # Đọc nội dung của file mới nhất
        latest_result_path = ocr_results[0]['path']
        print(f"Returning content of the latest OCR result: {latest_result_path}")
        
        with open(latest_result_path, 'r', encoding='utf-8') as f:
            latest_result = json.load(f)
        
        # Trả về nội dung của file OCR result
        return jsonify(latest_result), 200
    except Exception as e:
        print(f"Error getting OCR results: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get OCR results: {str(e)}"}), 500

# API 3: Xem hình ảnh
@app.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except:
        abort(404)

# API 4: Xóa hình ảnh
@app.route('/api/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": f"Image {filename} has been deleted successfully"}), 200
    else:
        return jsonify({"error": "Image not found"}), 404

# Thêm route để truy cập ảnh ROI đã xử lý
@app.route('/api/images/processed_roi/<filename>', methods=['GET'])
def get_processed_roi(filename):
    processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_roi')
    try:
        return send_from_directory(processed_folder, filename)
    except:
        abort(404)

# Thêm hàm để đọc cấu hình số thập phân
def get_decimal_places_config():
    """Đọc cấu hình số chữ số thập phân từ file"""
    decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
    if os.path.exists(decimal_config_path):
        try:
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading decimal places config: {str(e)}")
    return {}  # Trả về dictionary rỗng nếu không tìm thấy file

# API mới: Cập nhật cấu hình số chữ số thập phân cho các ROI
@app.route('/api/decimal_places', methods=['POST'])
def update_decimal_places():
    """
    Cập nhật cấu hình số chữ số thập phân cho các ROI
    
    Cấu trúc dữ liệu đầu vào:
    {
        "machine_code": "F1",  // Mã máy
        "screen_id": 3,        // ID màn hình
        "roi_config": {
            "Tgian_chu_ki": 1,  // Tên ROI "Tgian_chu_ki" có 1 chữ số thập phân
            "Vtri_khuon": 2,    // Tên ROI "Vtri_khuon" có 2 chữ số thập phân
            "ROI_2": 0          // ROI không có tên cụ thể sẽ sử dụng "ROI_<index>"
        }
    }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        if 'machine_code' not in data or 'screen_id' not in data or 'roi_config' not in data:
            return jsonify({"error": "Missing required fields: machine_code, screen_id, roi_config"}), 400
        
        machine_code = data['machine_code']
        screen_id = data['screen_id']
        roi_config = data['roi_config']
        
        # Đọc cấu hình hiện tại
        decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Cập nhật cấu hình
        if machine_code not in config:
            config[machine_code] = {}
        if screen_id not in config[machine_code]:
            config[machine_code][screen_id] = {}
        
        # Cập nhật hoặc thêm mới cấu hình cho từng ROI, sử dụng tên ROI làm key
        for roi_name, decimal_places in roi_config.items():
            config[machine_code][screen_id][roi_name] = int(decimal_places)
        
        # Lưu cấu hình
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            "message": "Decimal places configuration updated successfully",
            "config": config
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to update decimal places configuration: {str(e)}"}), 500

# API mới: Lấy cấu hình số chữ số thập phân cho các ROI
@app.route('/api/decimal_places', methods=['GET'])
def get_decimal_places():
    """Lấy cấu hình số chữ số thập phân cho các ROI"""
    try:
        decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return jsonify(config), 200
        else:
            return jsonify({}), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to get decimal places configuration: {str(e)}"}), 500

# API mới: Lấy cấu hình số chữ số thập phân cho một máy cụ thể
@app.route('/api/decimal_places/<machine_code>', methods=['GET'])
def get_decimal_places_for_machine(machine_code):
    """Lấy cấu hình số chữ số thập phân cho một máy cụ thể"""
    try:
        decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if machine_code in config:
                return jsonify(config[machine_code]), 200
            else:
                return jsonify({}), 200
        else:
            return jsonify({}), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to get decimal places configuration: {str(e)}"}), 500

# API mới: Lấy cấu hình số chữ số thập phân cho một màn hình cụ thể của một máy
@app.route('/api/decimal_places/<machine_code>/<screen_name>', methods=['GET'])
def get_decimal_places_for_screen(machine_code, screen_name):
    """Lấy cấu hình số chữ số thập phân cho một màn hình cụ thể của một máy"""
    try:
        decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if machine_code in config and screen_name in config[machine_code]:
                return jsonify(config[machine_code][screen_name]), 200
            else:
                return jsonify({}), 200
        else:
            return jsonify({}), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to get decimal places configuration: {str(e)}"}), 500

# API mới: Thêm API cập nhật giá trị decimal_places dựa trên máy và màn hình hiện tại
@app.route('/api/set_decimal_value', methods=['POST'])
def set_decimal_value():
    """
    Cập nhật giá trị số chữ số thập phân cho ROI cụ thể
    
    Form data với các keys:
    - machine_code: Mã máy (text, ví dụ: "F1")
    - screen_id: Tên màn hình (text, ví dụ: "Faults")
    - key: Tên ROI (text, ví dụ: "Tgian_chu_ki")
    - value: Giá trị số thập phân (text, ví dụ: "5")
    """
    try:
        # Kiểm tra đầu vào
        if 'machine_code' not in request.form or 'screen_id' not in request.form or 'key' not in request.form or 'value' not in request.form:
            return jsonify({"error": "Missing required fields: machine_code, screen_id, key, value"}), 400
        
        machine_code = request.form['machine_code']
        screen_id = request.form['screen_id']  # Tên màn hình (ví dụ: "Faults")
        roi_key = request.form['key']          # Tên ROI (ví dụ: "Tgian_chu_ki")
        
        # Chuyển đổi value từ text sang integer
        try:
            decimal_value = int(request.form['value'])
        except ValueError:
            return jsonify({"error": "value must be an integer"}), 400
        
        # Đọc cấu hình hiện tại
        decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Nếu machine_code chưa tồn tại trong config, thêm mới
        if machine_code not in config:
            config[machine_code] = {}
        
        # Nếu screen_id chưa tồn tại trong config của máy này, thêm mới
        if screen_id not in config[machine_code]:
            config[machine_code][screen_id] = {}
        
        # Cập nhật giá trị
        config[machine_code][screen_id][roi_key] = decimal_value
        
        # Lưu cấu hình
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            "message": "Decimal places value updated successfully",
            "machine_code": machine_code,
            "screen_id": screen_id,
            "key": roi_key,
            "value": decimal_value,
            "config": config[machine_code][screen_id]
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to update decimal places value: {str(e)}"}), 500

# API mới: Lấy thông tin máy và màn hình hiện tại
@app.route('/api/current_machine_screen', methods=['GET'])
def get_current_machine_screen():
    """Lấy thông tin về máy và màn hình hiện tại đã chọn"""
    try:
        machine_info = get_current_machine_info()
        if not machine_info:
            return jsonify({"error": "Current machine and screen information not found"}), 404
        
        area = machine_info['area']
        machine_code = machine_info['machine_code']
        machine_type = machine_info['machine_type']
        screen_id = machine_info['screen_id']
        
        # Đọc thông tin máy và màn hình
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Kiểm tra area, machine_code và machine_type
        if area not in data.get('areas', {}):
            return jsonify({"error": f"Area {area} not found"}), 404
        
        if machine_code not in data['areas'][area].get('machines', {}):
            return jsonify({"error": f"Machine {machine_code} not found in area {area}"}), 404
        
        if machine_type not in data.get('machine_types', {}):
            return jsonify({"error": f"Machine type {machine_type} not found"}), 404
        
        # Lấy thông tin màn hình đã chọn
        selected_screen = None
        for screen in data['machine_types'][machine_type]['screens']:
            if screen.get('screen_id') == screen_id:
                selected_screen = screen
                break
        
        if not selected_screen:
            return jsonify({"error": f"Screen {screen_id} not found for machine type {machine_type}"}), 404
        
        return jsonify({
            "area": {
                "area_code": area,
                "name": data['areas'][area]['name']
            },
            "machine": {
                "machine_code": machine_code,
                "name": data['areas'][area]['machines'][machine_code]['name'],
                "type": machine_type
            },
            "screen": selected_screen
        }), 200
    
    except ValueError as ve:
        return jsonify({"error": f"Invalid screen ID format: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to get current machine and screen: {str(e)}"}), 500

# Hàm helper để lấy thông tin máy hiện tại - OPTIMIZED VERSION
def get_current_machine_info():
    """Lấy thông tin máy hiện tại từ cache - tối ưu I/O"""
    return get_machine_info_cached()

# API mới: Lấy thông tin máy theo ID
@app.route('/api/machines', methods=['GET'])
def get_machine_info():
    """Lấy thông tin chi tiết về các máy và khu vực"""
    try:
        machine_code = request.args.get('machine_code', '').strip().upper()
        area = request.args.get('area', '').strip().upper()
        
        # Đọc file cấu hình
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Nếu có cả area và machine_code, trả về thông tin chi tiết của một máy cụ thể
        if area and machine_code:
            if area not in data.get('areas', {}):
                return jsonify({"error": f"Area {area} not found"}), 404
                
            if machine_code not in data['areas'][area].get('machines', {}):
                return jsonify({"error": f"Machine {machine_code} not found in area {area}"}), 404
                
            machine_info = data['areas'][area]['machines'][machine_code]
            machine_type = machine_info.get('type')
            
            if not machine_type or machine_type not in data.get('machine_types', {}):
                return jsonify({"error": f"Machine type {machine_type} not found for machine {machine_code}"}), 404
                
            # Lấy thông tin màn hình từ machine_types
            screens = []
            for screen in data['machine_types'][machine_type].get('screens', []):
                screen_id = screen['id']
                screen_name = screen.get('screen_id', '')
                
                # Lấy thông tin ROI
                roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_id, machine_type)
                
                screen_info = {
                    "id": screen_id,
                    "screen_id": screen_name,
                    "description": screen.get('description', ''),
                    "roi_count": len(roi_coordinates) if roi_coordinates else 0
                }
                
                # Kiểm tra cấu hình decimal places
                decimal_config = get_decimal_places_config()
                has_decimal_config = (machine_code in decimal_config and 
                                     screen_name in decimal_config[machine_code])
                
                screen_info['has_decimal_config'] = has_decimal_config
                
                screens.append(screen_info)
                
            # Trả về thông tin chi tiết máy
            return jsonify({
                "area": area,
                "area_name": data['areas'][area]['name'],
                "machine_code": machine_code,
                "machine_name": machine_info['name'],
                "machine_type": machine_type,
                "screens": screens
            }), 200
            
        # Nếu chỉ có area, trả về danh sách máy trong khu vực đó
        elif area:
            if area not in data.get('areas', {}):
                return jsonify({"error": f"Area {area} not found"}), 404
                
            machines = []
            for m_code, m_info in data['areas'][area]['machines'].items():
                machines.append({
                    "machine_code": m_code,
                    "name": m_info['name'],
                    "type": m_info['type']
                })
                
            return jsonify({
                "area": area,
                "area_name": data['areas'][area]['name'],
                "machines": machines,
                "machines_count": len(machines)
            }), 200
        
        # Nếu không có tham số, trả về danh sách tất cả các khu vực
        areas = []
        for area_code, area_info in data.get('areas', {}).items():
            machine_count = len(area_info.get('machines', {}))
            areas.append({
                "area": area_code,
                "name": area_info['name'],
                "machine_count": machine_count
            })
            
        return jsonify({
            "areas": areas,
            "areas_count": len(areas)
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to get machine information: {str(e)}"}), 500

# API mới: Cập nhật máy và màn hình
@app.route('/api/set_machine_screen', methods=['POST'])
def set_machine_screen():
    """
    Cập nhật machine_order và parameter_order dựa trên mã máy và tên màn hình
    
    Cấu trúc dữ liệu đầu vào:
    Form data với các key:
    - machine_code: Mã máy (ví dụ: "F1")
    - screen_id: Tên của màn hình (chuỗi, ví dụ: "Production")
    """
    try:
        # Kiểm tra đầu vào
        if 'machine_code' not in request.form or 'screen_id' not in request.form:
            return jsonify({
                "error": "Missing required fields. Please provide machine_code and screen_id in form-data"
            }), 400
        
        machine_code = request.form['machine_code'].strip().upper()
        screen_name = request.form['screen_id'].strip()  # screen_id giờ là tên màn hình
        
        # Kiểm tra tính hợp lệ của mã máy và tên màn hình
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if machine_code not in data['machines']:
            return jsonify({"error": f"Machine {machine_code} not found"}), 404
        
        # Tìm màn hình có tên trùng khớp và lấy ID số
        screen_numeric_id = None
        selected_screen = None
        for screen in data['machines'][machine_code]['screens']:
            if screen['screen_id'] == screen_name:
                screen_numeric_id = screen['id']
                selected_screen = screen
                break
        
        if not screen_numeric_id:
            return jsonify({"error": f"Screen '{screen_name}' not found for machine {machine_code}"}), 404
        
        # Cập nhật parameter_order_value.txt với ID số của màn hình
        parameter_order_file_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'parameter_order_value.txt')
        with open(parameter_order_file_path, 'w', encoding='utf-8') as f:
            f.write(str(screen_numeric_id))
        
        return jsonify({
            "message": "Machine and screen selection updated successfully",
            "machine": {
                "machine_code": machine_code
            },
            "screen": {
                "id": screen_numeric_id,
                "screen_id": selected_screen['screen_id'],
                "description": selected_screen.get('description', '')
            }
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to update machine and screen selection: {str(e)}"}), 500

# API mới: Kiểm tra tình trạng cấu hình của một máy và màn hình cụ thể.
# Kiểm tra xem đã có ROI và cấu hình decimal places cho mỗi ROI hay chưa.
# 
# Query parameters:
# - machine_code: Mã máy (ví dụ: "IE-F1-CWA01")
# - screen_id: ID màn hình (số nguyên)
# 
# Nếu không cung cấp tham số, lấy thông tin từ máy và màn hình hiện tại.
@app.route('/api/machine_screen_status', methods=['GET'])
def check_machine_screen_status():
    try:
        # Lấy machine_code và screen_id từ query parameters
        machine_code = request.args.get('machine_code')
        screen_id = request.args.get('screen_id')
        
        # Nếu không có, thử lấy từ thông tin máy hiện tại
        if not machine_code or not screen_id:
            machine_info = get_current_machine_info()
            if machine_info:
                machine_code = machine_info['machine_code']
                screen_id = machine_info['screen_id']
            else:
                return jsonify({
                    "error": "Missing machine_code and screen_id. Please provide them as query parameters or set them using /api/set_machine_screen first."
                }), 400
        else:
            # Chuyển đổi screen_id thành integer nếu được cung cấp
            try:
                screen_id = int(screen_id)
            except ValueError:
                return jsonify({"error": "screen_id must be an integer"}), 400
        
        # Lấy thông tin máy
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm area và machine_type từ machine_code
        area = get_area_for_machine(machine_code)
        if not area:
            return jsonify({"error": f"Area not found for machine {machine_code}"}), 404
            
        if machine_code not in data['areas'][area]['machines']:
            return jsonify({"error": f"Machine {machine_code} not found in area {area}"}), 404
        
        machine_type = data['areas'][area]['machines'][machine_code]['type']
        
        # Kiểm tra screen_id có tồn tại không
        screen_exists = False
        screen_name = None
        for screen in data['machine_types'][machine_type]['screens']:
            if screen['id'] == screen_id:
                screen_exists = True
                screen_name = screen['screen_id']
                break
                
        if not screen_exists:
            return jsonify({"error": f"Screen ID {screen_id} not found for machine type {machine_type}"}), 404
        
        # Kiểm tra ROI
        roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_id, machine_type)
        has_roi = roi_coordinates is not None and len(roi_coordinates) > 0
        roi_count = len(roi_coordinates) if has_roi else 0
        
        # Kiểm tra cấu hình decimal places
        decimal_config = get_decimal_places_config()
        has_decimal_config = (machine_code in decimal_config and 
                             screen_id in decimal_config[machine_code] and 
                             len(decimal_config[machine_code][screen_id]) > 0)
        
        # Kiểm tra từng ROI có cấu hình decimal places không
        roi_status = []
        if has_roi:
            for i in range(roi_count):
                roi_name = roi_names[i] if i < len(roi_names) else f"ROI_{i}"
                has_decimal = (machine_code in decimal_config and 
                              screen_id in decimal_config[machine_code] and 
                              (roi_name in decimal_config[machine_code][screen_id] or 
                               str(i) in decimal_config[machine_code][screen_id]))
                
                decimal_value = None
                if has_decimal:
                    if roi_name in decimal_config[machine_code][screen_id]:
                        decimal_value = decimal_config[machine_code][screen_id][roi_name]
                    elif str(i) in decimal_config[machine_code][screen_id]:
                        decimal_value = decimal_config[machine_code][screen_id][str(i)]
                
                roi_status.append({
                    "roi_index": i,
                    "roi_name": roi_name,
                    "has_decimal_config": has_decimal,
                    "decimal_places": decimal_value
                })
        
        # Trả về kết quả tình trạng
        status = {
            "area": area,
            "area_name": data['areas'][area]['name'],
            "machine_code": machine_code,
            "machine_name": data['areas'][area]['machines'][machine_code]['name'],
            "machine_type": machine_type,
            "screen_id": screen_id,
            "screen_name": screen_name,
            "has_roi": has_roi,
            "roi_count": roi_count,
            "has_decimal_config": has_decimal_config,
            "is_fully_configured": has_roi and all(roi["has_decimal_config"] for roi in roi_status),
            "roi_status": roi_status
        }
        
        return jsonify(status), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to check machine screen status: {str(e)}"}), 500

# API mới: Thiết lập tất cả decimal places cho một màn hình
@app.route('/api/set_all_decimal_values', methods=['POST'])
def set_all_decimal_values():
    """
    Thiết lập giá trị decimal places cho tất cả ROI của một màn hình trong một lần gọi API
    
    Cấu trúc dữ liệu đầu vào:
    {
        "machine_code": "F1",  // Mã máy (bắt buộc)
        "screen_id": 1,       // ID màn hình (bắt buộc)
        "decimal_values": {    // Giá trị decimal places cho từng ROI
            "0": 1,            // ROI 0 có 1 chữ số thập phân
            "1": 2,            // ROI 1 có 2 chữ số thập phân
            "2": 0             // ROI 2 không có chữ số thập phân
        }
    }
    
    Nếu không cung cấp decimal_values cho một ROI nào đó, giá trị hiện tại sẽ được giữ nguyên.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        if 'machine_code' not in data or 'screen_id' not in data or 'decimal_values' not in data:
            return jsonify({"error": "Missing required fields: machine_code, screen_id, decimal_values"}), 400
        
        machine_code = data['machine_code']
        screen_id = data['screen_id']
        decimal_values = data['decimal_values']
        
        # Kiểm tra ROI tồn tại cho máy và màn hình này
        roi_coordinates = get_roi_coordinates(machine_code, screen_id)
        if not roi_coordinates:
            return jsonify({"error": f"No ROI defined for machine {machine_code}, screen {screen_id}. Please define ROI first."}), 404
        
        # Kiểm tra dữ liệu decimal_values
        if not isinstance(decimal_values, dict):
            return jsonify({"error": "decimal_values must be a dictionary with ROI indices as keys"}), 400
        
        # Kiểm tra các roi_index hợp lệ
        for roi_index in decimal_values:
            try:
                roi_idx = int(roi_index)
                if roi_idx < 0 or roi_idx >= len(roi_coordinates):
                    return jsonify({"error": f"Invalid roi_index {roi_index}. Must be between 0 and {len(roi_coordinates)-1}"}), 400
                
                # Kiểm tra decimal value là integer
                if not isinstance(decimal_values[roi_index], int):
                    return jsonify({"error": f"Decimal value for ROI {roi_index} must be an integer"}), 400
            except ValueError:
                return jsonify({"error": f"roi_index {roi_index} must be an integer"}), 400
        
        # Đọc cấu hình hiện tại
        decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Nếu machine_code chưa tồn tại trong config, thêm mới
        if machine_code not in config:
            config[machine_code] = {}
        
        # Nếu screen_id chưa tồn tại trong config của máy này, thêm mới
        if screen_id not in config[machine_code]:
            config[machine_code][screen_id] = {}
        
        # Cập nhật giá trị cho tất cả ROI được cung cấp
        for roi_index, decimal_value in decimal_values.items():
            config[machine_code][screen_id][roi_index] = decimal_value
        
        # Lưu cấu hình
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Kiểm tra xem còn ROI nào chưa được cấu hình không
        unconfigured_rois = []
        for i in range(len(roi_coordinates)):
            if str(i) not in config[machine_code][screen_id]:
                unconfigured_rois.append(i)
        
        return jsonify({
            "message": "Decimal places values updated successfully",
            "machine_code": machine_code,
            "screen_id": screen_id,
            "updated_rois": list(decimal_values.keys()),
            "unconfigured_rois": unconfigured_rois,
            "config": config[machine_code][screen_id]
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to update decimal places values: {str(e)}"}), 500

# Thêm hàm mới để lấy ID số của màn hình từ tên màn hình
def get_screen_numeric_id(machine_type, screen_name):
    """
    Lấy ID số của một màn hình dựa trên tên màn hình
    
    Args:
        machine_type: Loại máy (ví dụ: F1, F41, F42)
        screen_name: Tên màn hình (ví dụ: Plasticizer)
        
    Returns:
        int: ID số của màn hình, hoặc None nếu không tìm thấy
    """
    try:
        # Sử dụng đường dẫn tuyệt đối để đảm bảo tìm thấy file
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            print(f"Machine screens file not found at {machine_screens_path}")
            return None
        
        print(f"Reading machine screens from: {machine_screens_path}")
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Looking for screen '{screen_name}' in machine_type '{machine_type}'")
        
        # Kiểm tra trong machine_types
        if machine_type in data.get('machine_types', {}):
            # Tìm màn hình có screen_id (tên màn hình) trùng khớp
            for screen in data['machine_types'][machine_type].get('screens', []):
                print(f"Checking screen: ID={screen['id']}, screen_id={screen['screen_id']}")
                if screen['screen_id'] == screen_name:
                    print(f"Found matching screen! ID={screen['id']}, screen_id={screen['screen_id']}")
                    return screen['id']
            
            print(f"No matching screen found for '{screen_name}' in machine_type '{machine_type}'")
            return None
        else:
            print(f"Machine type '{machine_type}' not found in machine_screens.json")
            return None
    except Exception as e:
        print(f"Error getting screen numeric ID: {str(e)}")
        traceback.print_exc()
        return None

# Hàm tiền xử lý ảnh để tối ưu cho OCR với màn hình HMI
def preprocess_hmi_image(image, roi_coordinates, original_filename):
    results = []
    
    # Tạo thư mục để lưu ảnh ROI đã xử lý (nếu chưa tồn tại)
    processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_roi')
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    # Lấy tên file gốc không có phần mở rộng
    base_filename = os.path.splitext(original_filename)[0]
    
    # Xóa tất cả các ảnh ROI processed và original cũ liên quan đến file này
    for old_file in os.listdir(processed_folder):
        if old_file.startswith(f"{base_filename}_roi_"):
            try:
                os.remove(os.path.join(processed_folder, old_file))
                print(f"Removed old ROI file: {old_file}")
            except Exception as e:
                print(f"Could not remove old ROI file {old_file}: {str(e)}")
    
    for i, (x1, y1, x2, y2) in enumerate(roi_coordinates):
        # Đảm bảo tọa độ là số nguyên khi cắt ROI
        x1, x2 = int(min(x1, x2)), int(max(x1, x2))
        y1, y2 = int(min(y1, y2)), int(max(y1, y2))
        
        # Bây giờ sử dụng tọa độ đã chuyển đổi
        roi = image[y1:y2, x1:x2]
        
        # Kiểm tra nếu ROI rỗng hoặc kích thước quá nhỏ
        if roi.size == 0 or roi.shape[0] <= 5 or roi.shape[1] <= 5:
            continue
        
        # -------------------------------
        # 1. Đọc ảnh và chuyển sang mức xám
        # -------------------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # -------------------------------
        # 2. Tiền xử lý (làm mượt, giảm nhiễu)
        # -------------------------------
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        blur = cv2.bilateralFilter(blur, d=9, sigmaColor =75, sigmaSpace=75)
        # -------------------------------
        # 3. Ngưỡng (threshold) để tách chữ số
        # -------------------------------
        # Sử dụng Otsu's threshold
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # -------------------------------
        # 4. (Tuỳ chọn) Invert nếu chữ số tối trên nền sáng
        # -------------------------------
        # Nếu cần, có thể đảo ngược ngưỡng
        # thresh = 255 - thresh

        # -------------------------------
        # 5. Morphological Operations (nếu cần)
        # -------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Lưu kết quả đã xử lý
        results.append({
            "roi_index": i,
            "coordinates": [x1, y1, x2, y2],
            "processed_image": closing,
            "processed_image_path": f"/api/images/processed_roi/{base_filename}_roi_{i}_processed.png",
            "original_roi_path": f"/api/images/processed_roi/{base_filename}_roi_{i}_original.png"
        })
        
        # Lưu ảnh đã xử lý vào thư mục processed_roi
        # processed_filename = f"{base_filename}_roi_{i}_processed.png"
        # processed_path = os.path.join(processed_folder, processed_filename)
        # cv2.imwrite(processed_path, closing)  # Save the processed image directly
        
    return results

# API mới: Upload/Update ảnh template mẫu
@app.route('/api/reference_images', methods=['POST'])
def upload_reference_image():
    """
    API để tải lên ảnh template mẫu dùng cho việc căn chỉnh
    
    Form data parameters:
    - file: File ảnh template mẫu
    - machine_type: Loại máy (ví dụ: F1, F41, F42)
    - screen_id: Mã màn hình (ví dụ: Faults)
    """
    # Kiểm tra xem có file trong request không
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Kiểm tra xem có chọn file chưa
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Kiểm tra machine_type và screen_id từ form data
    machine_type = request.form.get('machine_type')
    screen_id = request.form.get('screen_id')
    
    if not machine_type or not screen_id:
        return jsonify({
            "error": "Missing machine_type or screen_id. Both are required."
        }), 400
    
    # Kiểm tra file có phải là hình ảnh không
    if file and allowed_file(file.filename):
        # Tạo tên file cho ảnh template (format: template_{machine_type}_{screen_id}.jpg)
        extension = os.path.splitext(file.filename)[1].lower()
        reference_filename = f"template_{machine_type}_{screen_id}{extension}"
        reference_path = os.path.join(app.config['REFERENCE_IMAGES_FOLDER'], reference_filename)
        
        # Kiểm tra xem file đã tồn tại chưa và xóa nếu có
        if os.path.exists(reference_path):
            os.remove(reference_path)
        
        # Lưu file
        file.save(reference_path)
        
        # Kiểm tra file có thể đọc được bằng OpenCV không
        try:
            image = cv2.imread(reference_path)
            if image is None:
                os.remove(reference_path)  # Xóa file nếu không đọc được
                return jsonify({"error": "File could not be read as an image with OpenCV"}), 400
            
            image_height, image_width = image.shape[:2]
        except Exception as e:
            if os.path.exists(reference_path):
                os.remove(reference_path)
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
        
        return jsonify({
            "message": "Reference template uploaded successfully",
            "template": {
                "filename": reference_filename,
                "path": f"/api/reference_images/{reference_filename}",
                "machine_type": machine_type,
                "screen_id": screen_id,
                "size": os.path.getsize(reference_path),
                "dimensions": f"{image_width}x{image_height}"
            }
        }), 201
    
    return jsonify({"error": "File type not allowed"}), 400

# API: Lấy danh sách ảnh template mẫu
@app.route('/api/reference_images', methods=['GET'])
def get_reference_images():
    """Lấy danh sách các ảnh template mẫu đã tải lên"""
    reference_images = []
    
    # Filter theo machine_type và screen_id nếu được cung cấp
    machine_type = request.args.get('machine_type')
    screen_id = request.args.get('screen_id')
    
    for filename in os.listdir(app.config['REFERENCE_IMAGES_FOLDER']):
        if allowed_file(filename):
            file_path = os.path.join(app.config['REFERENCE_IMAGES_FOLDER'], filename)
            
            # Trích xuất thông tin từ tên file
            file_info = filename.split('_')
            if len(file_info) >= 3 and file_info[0] == 'template':
                file_machine_type = file_info[1]
                # Trích xuất screen_id (có thể chứa dấu '_')
                file_screen_id = '_'.join(file_info[2:]).split('.')[0]
                
                # Lọc theo machine_type nếu được cung cấp
                if machine_type and file_machine_type != machine_type:
                    continue
                
                # Lọc theo screen_id nếu được cung cấp
                if screen_id and file_screen_id != screen_id:
                    continue
                
                try:
                    # Đọc kích thước ảnh
                    image = cv2.imread(file_path)
                    if image is not None:
                        image_height, image_width = image.shape[:2]
                        dimensions = f"{image_width}x{image_height}"
                    else:
                        dimensions = "Unknown"
                except:
                    dimensions = "Unknown"
                
                reference_images.append({
                    "filename": filename,
                    "path": f"/api/reference_images/{filename}",
                    "machine_type": file_machine_type,
                    "screen_id": file_screen_id,
                    "size": os.path.getsize(file_path),
                    "dimensions": dimensions,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getctime(file_path)))
                })
    
    return jsonify({
        "reference_images": reference_images,
        "count": len(reference_images)
    })

# API: Xem ảnh template mẫu
@app.route('/api/reference_images/<filename>', methods=['GET'])
def get_reference_image(filename):
    """Trả về file ảnh template mẫu"""
    try:
        return send_from_directory(app.config['REFERENCE_IMAGES_FOLDER'], filename)
    except:
        abort(404)

# API: Xóa ảnh template mẫu
@app.route('/api/reference_images/<filename>', methods=['DELETE'])
def delete_reference_image(filename):
    """Xóa file ảnh template mẫu"""
    file_path = os.path.join(app.config['REFERENCE_IMAGES_FOLDER'], filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": f"Reference image {filename} has been deleted successfully"}), 200
    else:
        return jsonify({"error": "Reference image not found"}), 404

# API: Truy cập ảnh kết quả phát hiện HMI
@app.route('/api/images/hmi_detection/<filename>', methods=['GET'])
def get_hmi_detection_image(filename):
    """Trả về file ảnh kết quả phát hiện HMI"""
    try:
        print(f"Accessing HMI detection image: {filename}")
        print(f"Looking in directory: {app.config['UPLOAD_FOLDER']}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving HMI detection image: {str(e)}")
        abort(404)

@app.route('/api/images/hmi_refined/<filename>', methods=['GET'])
def get_hmi_refined_image(filename):
    """Trả về file ảnh HMI refined đã được lưu"""
    try:
        print(f"Accessing HMI refined image: {filename}")
        print(f"Looking in directory: {app.config['HMI_REFINED_FOLDER']}")
        return send_from_directory(app.config['HMI_REFINED_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving HMI refined image: {str(e)}")
        abort(404)

# Hàm mới: Lấy đường dẫn đến ảnh template mẫu dựa trên machine_code và screen_id
def get_reference_template_path(machine_type, screen_id):
    """
    Tìm kiếm ảnh template mẫu dựa trên machine_type và screen_id
    
    Returns:
        str: Đường dẫn đến file template nếu tìm thấy, None nếu không tìm thấy
    """
    reference_folder = app.config['REFERENCE_IMAGES_FOLDER']
    
    # Tạo pattern tên file
    file_pattern = f"template_{machine_type}_{screen_id}.*"
    
    # Tìm kiếm file theo pattern
    for filename in os.listdir(reference_folder):
        if fnmatch.fnmatch(filename, file_pattern):
            return os.path.join(reference_folder, filename)
    
    return None

def preprocess_roi_for_ocr(roi, roi_index, original_filename, roi_name=None, image_aligned=None, x1=None, y1=None, x2=None, y2=None):
    """
    Tiền xử lý ảnh ROI để tối ưu cho OCR
    
    Args:
        roi: Ảnh ROI cần xử lý
        roi_index: Chỉ số ROI
        original_filename: Tên file gốc
        roi_name: Tên ROI (tùy chọn)
        image_aligned: Ảnh đã được căn chỉnh
        x1, y1, x2, y2: Tọa độ của ROI trong ảnh gốc
        
    Returns:
        Tuple: (Ảnh ROI đã được tiền xử lý, thông tin chất lượng ảnh)
    """
    # Sử dụng tên ROI nếu có, nếu không sử dụng chỉ số
    x1, y1, x2, y2 = x1, y1, x2, y2
    identifier = roi_name if roi_name else f"ROI_{roi_index}"
    
    # Kiểm tra nếu ảnh rỗng
    if roi is None or roi.size == 0 or roi.shape[0] <= 5 or roi.shape[1] <= 5:
        return None, None
    
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    quality_info = check_image_quality(gray)
    # Kiểm tra quality_info có phải là None không
    if quality_info is not None and not quality_info['is_good_quality']:
        # Cải thiện chất lượng ảnh
        enhanced_gray = enhance_image_quality(gray, quality_info)
        # Sử dụng ảnh đã cải thiện cho các bước tiếp theo
        gray = enhanced_gray
    
    # 2. Tiền xử lý (làm mượt, giảm nhiễu)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
    contrast_enhanced = clahe.apply(blur)
    # 3. Threshold (Otsu)
    _, thresh_otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = thresh_otsu

    # 4. Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Tăng lên (5,5) thay vì (2,2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 5. Đảo ngược ảnh để số trở thành foreground (trắng trên nền đen)
    inverted = cv2.bitwise_not(closing)
    
    # 6. Tìm contour với RETR_LIST để tìm tất cả contours
    contours, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 7. Tính giới hạn trên và dưới cho mỗi contour
    contour_limits = []
    for i, cnt in enumerate(contours):
        # Tính bounding rectangle cho contour
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Điều kiện lọc mới dựa trên phân tích thực tế:
        # - Loại bỏ contour quá nhỏ (nhiễu)
        # - Loại bỏ contour quá lớn (toàn bộ ảnh)
        # - Chấp nhận contour có kích thước phù hợp với số
        if w <= 3 or h <= 8 or area < 20:  # Quá nhỏ - nhiễu
            continue
        if w > 50 or h > 50 or area > 1000:  # Quá lớn - có thể là background
            continue
        
        y_coords = [point[0][1] for point in cnt]  # Lấy tọa độ y của các điểm trong contour
        upper_limit = min(y_coords)
        lower_limit = max(y_coords)
        contour_limits.append((upper_limit, lower_limit, cnt))

    # 8. Đếm số lượng contour nằm trong giới hạn y của từng contour
    max_overlap_count = 0
    best_contour = None
    best_limits = None

    for i, (upper, lower, cnt) in enumerate(contour_limits):
        overlap_count = sum(1 for j, (other_upper, other_lower, _) in enumerate(contour_limits) if i != j and not (lower < other_upper or upper > other_lower))
        
        if overlap_count > max_overlap_count:
            max_overlap_count = overlap_count
            best_contour = cnt
            best_limits = (upper, lower)
        elif overlap_count == max_overlap_count and best_contour is not None:
            # Chọn ngẫu nhiên một contour nếu có nhiều contour có số lượng overlap giống nhau
            if random.choice([True, False]):
                best_contour = cnt
                best_limits = (upper, lower)

    # 9. Gộp tất cả các contour trong vùng giới hạn y của contour tốt nhất
    if best_contour is not None and len(contour_limits) > 0:
        # Nếu chỉ có 1 contour thì sử dụng contour đó luôn mà không cần gộp
        if len(contour_limits) == 1:
            merged_contour = contour_limits[0][2]
        else:
            merged_contour = np.vstack([cnt for upper, lower, cnt in contour_limits if not (best_limits[1] < upper or best_limits[0] > lower)])
        
        # 10. Cắt (crop) vùng boundingRect của contour lớn nhất với padding
        x, y, w, h = cv2.boundingRect(merged_contour)
        
        # Mở rộng thêm padding xung quanh
        pad = 5
        x3, y3 = x1+x-pad, y1+y-pad
        x4, y4 = x1+x+w+pad, y1+y+h+pad

        # Đảm bảo không vượt quá biên ảnh
        x3 = max(0, x3)
        y3 = max(0, y3)
        x4 = min(image_aligned.shape[1], x4)
        y4 = min(image_aligned.shape[0], y4)

        # Lưu ý: Cắt từ ảnh đã xử lý (closing) để có dạng grayscale
        cropped_closing = image_aligned[y3:y4, x3:x4]

        gray = cv2.cvtColor(cropped_closing, cv2.COLOR_BGR2GRAY)

        # 11. Tiền xử lý (làm mượt, giảm nhiễu)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 12. Threshold (Otsu)
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = thresh_otsu

        # 13. Morphological Closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Tăng lên (5,5) thay vì (2,2)
        closing_final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closing_final = cv2.blur(closing_final, (3, 3))
        
        return closing_final, quality_info  # Trả về ảnh grayscale và thông tin chất lượng
    else:
        closing = cv2.blur(closing, (4, 4))
        # Trả về ảnh grayscale nếu không tìm thấy contour
        return closing, quality_info  # Trả về ảnh grayscale và thông tin chất lượng

def is_named_roi_format(roi_list):
    """Kiểm tra xem danh sách ROI có phải là định dạng mới (có name và coordinates) hay không"""
    if not roi_list:
        return False
    
    first_item = roi_list[0]
    return isinstance(first_item, dict) and "name" in first_item and "coordinates" in first_item

# Thêm route mới cho /api/machines/<machine_code>
@app.route('/api/machine_screens/<machine_code>', methods=['GET'])
def get_machine_screens(machine_code):
    """
    Lấy danh sách các màn hình (screens) cho một máy cụ thể
    
    Path Parameters:
    - machine_code: Mã máy (bắt buộc, ví dụ: IE-F1-CWA01)
    """
    try:
        # Đọc file JSON chứa thông tin về máy và màn hình
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm khu vực chứa máy này
        area = get_area_for_machine(machine_code)
        if not area:
            return jsonify({"error": f"Area not found for machine {machine_code}"}), 404
        
        # Kiểm tra xem machine_code có tồn tại trong khu vực đó không
        if machine_code not in data['areas'][area]['machines']:
            return jsonify({"error": f"Machine {machine_code} not found in area {area}"}), 404
        
        # Lấy thông tin cơ bản về máy
        machine_info = data['areas'][area]['machines'][machine_code]
        machine_type = machine_info.get('type')
        
        if not machine_type or machine_type not in data.get('machine_types', {}):
            return jsonify({"error": f"Machine type {machine_type} not found for machine {machine_code}"}), 404
        
        # Lấy thông tin các màn hình của loại máy này
        screens = []
        for screen in data['machine_types'][machine_type].get('screens', []):
            screen_id = screen['id']
            screen_name = screen.get('screen_id', '')
            
            # Lấy thông tin ROI
            roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_id, machine_type)
            
            screen_info = {
                "id": screen_id,
                "screen_id": screen_name,
                "description": screen.get('description', ''),
                "roi_count": len(roi_coordinates) if roi_coordinates else 0
            }
            
            # Kiểm tra cấu hình decimal places
            decimal_config = get_decimal_places_config()
            has_decimal_config = (machine_code in decimal_config and 
                                screen_name in decimal_config[machine_code])
            
            screen_info['has_decimal_config'] = has_decimal_config
            
            screens.append(screen_info)
        
        return jsonify({
            "area": area,
            "area_name": data['areas'][area]['name'],
            "machine_code": machine_code,
            "machine_name": machine_info['name'],
            "machine_type": machine_type,
            "screens_count": len(screens),
            "screens": screens
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to get screens: {str(e)}"}), 500

# Thêm API mới để lấy danh sách máy trong một khu vực
@app.route('/api/machines/<area_code>', methods=['GET'])
def get_machines_by_area(area_code):
    """
    Lấy danh sách các máy trong một khu vực cụ thể
    
    Path Parameters:
    - area_code: Mã khu vực (bắt buộc, ví dụ: F1)
    """
    try:
        area_code = area_code.strip().upper()
        
        # Đọc file cấu hình
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Kiểm tra xem area_code có tồn tại không
        if area_code not in data.get('areas', {}):
            return jsonify({"error": f"Area {area_code} not found"}), 404
        
        # Lấy danh sách máy trong khu vực
        machines = []
        for machine_code, machine_info in data['areas'][area_code]['machines'].items():
            machines.append({
                "machine_code": machine_code,
                "name": machine_info['name'],
                "type": machine_info['type'],
                "description": machine_info.get('description', 'monitor 1.png')
            })
        
        return jsonify({
            "area": area_code,
            "area_name": data['areas'][area_code]['name'],
            "machines": machines,
            "machines_count": len(machines)
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to get machines for area {area_code}: {str(e)}"}), 500

# API mới: Cập nhật cấu hình số chữ số thập phân cho một màn hình cụ thể
@app.route('/api/decimal_places/<machine_code>/<screen_name>', methods=['POST'])
def update_decimal_places_for_screen(machine_code, screen_name):
    """
    Cập nhật cấu hình số chữ số thập phân cho một màn hình cụ thể
    
    Path Parameters:
    - machine_code: Mã máy (ví dụ: F1)
    - screen_name: Tên màn hình (ví dụ: Faults)
    
    Request Body (JSON):
    {
        "key1": value1,  // Giá trị mới cho key1
        "key2": value2   // Giá trị mới cho key2
    }
    
    Chỉ cập nhật các key được gửi trong request, các key khác giữ nguyên giá trị
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        # Đọc dữ liệu từ request body
        new_values = request.json
        
        # Đọc cấu hình hiện tại
        decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Tạo cấu trúc nếu chưa tồn tại
        if machine_code not in config:
            config[machine_code] = {}
        
        if screen_name not in config[machine_code]:
            config[machine_code][screen_name] = {}
        
        # Lưu lại cấu hình hiện tại để so sánh
        original_config = config[machine_code][screen_name].copy() if screen_name in config[machine_code] else {}
        
        # Cập nhật các giá trị mới (chỉ ghi đè các key được gửi trong request)
        for key, value in new_values.items():
            config[machine_code][screen_name][key] = value
        
        # Lưu cấu hình mới
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Tạo thông tin về các thay đổi
        changes = {
            "added": {},
            "updated": {}
        }
        
        for key, value in new_values.items():
            if key in original_config:
                if original_config[key] != value:
                    changes["updated"][key] = {
                        "old_value": original_config[key],
                        "new_value": value
                    }
            else:
                changes["added"][key] = value
        
        return jsonify({
            "message": "Decimal places configuration updated successfully",
            "machine_code": machine_code,
            "screen_name": screen_name,
            "changes": changes,
            "config": config[machine_code][screen_name]
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to update decimal places configuration: {str(e)}"}), 500

# Hàm mới để kiểm tra chất lượng ảnh
def check_image_quality(image):
    """
    Kiểm tra chất lượng ảnh và trả về kết quả đánh giá cùng với thông tin chất lượng
    
    Args:
        image: Ảnh cần kiểm tra (grayscale)
        
    Returns:
        dict: {
            'is_good_quality': True/False,
            'issues': [],
            'blurriness': float,
            'brightness': float,
            'contrast': float,
            'has_glare': bool,
            'has_moire': bool
        }
    """
    # Kết quả mặc định
    result = {
        'is_good_quality': True,
        'issues': [],
        'blurriness': 0,
        'brightness': 0,
        'contrast': 0,
        'has_glare': False,
        'has_moire': False
    }
    
    # Kiểm tra ảnh null hoặc rỗng
    if image is None or image.size == 0:
        result['is_good_quality'] = False
        result['issues'].append('empty_image')
        return result
    
    # Đảm bảo ảnh đã chuyển sang grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Tính độ sáng trung bình
    brightness = np.mean(gray)
    result['brightness'] = brightness
    
    # Tính độ tương phản bằng độ lệch chuẩn
    contrast = np.std(gray)
    result['contrast'] = contrast
    # 4. Kiểm tra Moiré pattern (sử dụng FFT)
    # Chuyển đổi ảnh sang không gian tần số bằng FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    
    # Tính phổ biên độ
    magnitude_spectrum = np.abs(fft_shift)
    magnitude_spectrum = 20 * np.log(magnitude_spectrum + 1e-10)  # Tránh log(0)
    
    # Chuẩn hóa phổ biên độ về khoảng [0, 1]
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
    
    # Tính ngưỡng thích nghi bằng percentile
    threshold = np.percentile(magnitude_spectrum, 99)
    
    # Lọc dải tần số: Tập trung vào tần số trung
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    r = min(rows, cols) // 4  # Bán kính cho dải tần số trung
    mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if r / 2 < dist < r:  # Chỉ giữ tần số trung
                mask[i, j] = True
    filtered_magnitude = magnitude_spectrum * mask
    
    # Đếm số đỉnh trong phổ đã lọc
    peaks = np.sum(filtered_magnitude > threshold)
    peak_ratio = peaks / np.sum(mask)  # Tỷ lệ so với vùng được lọc
    
    # Quyết định
    if (brightness > 200) and (contrast > 20) and (0.077 >peak_ratio > 0.005):
        print(peak_ratio)
        result['has_moire'] = True
        result['is_good_quality'] = False
        result['issues'].append('moire_pattern')
        return result
    # 1. Kiểm tra độ sắc nét (blurriness) bằng biến đổi Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result['blurriness'] = laplacian_var
    
    # Ngưỡng blurriness: thấp đồng nghĩa với ảnh mờ
    blur_threshold = 7.0  # Ngưỡng này có thể điều chỉnh
    if laplacian_var < blur_threshold:
        result['is_good_quality'] = False
        result['issues'].append('blurry')
        return result
    # 2. Kiểm tra độ sáng và độ tương phản
    # Tính histogram của ảnh
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / gray.size
    
    # Kiểm tra độ sáng quá thấp hoặc quá cao
    if brightness > 220:
        result['is_good_quality'] = False
        result['issues'].append('too_bright')
        return result
    # Kiểm tra độ tương phản quá thấp
    if contrast < 16:
        result['is_good_quality'] = False
        result['issues'].append('low_contrast')

        return result
    # 3. Kiểm tra hiện tượng chói (glare)
    # Tìm vùng sáng quá mức (gần trắng)
    bright_threshold = 250
    bright_pixels = np.sum(gray > bright_threshold)
    bright_ratio = bright_pixels / gray.size
    
    # Nếu tỷ lệ pixel sáng quá cao, coi là có glare
    if bright_ratio > 0.2:  # 20% pixel quá sáng
        result['has_glare'] = True
        result['is_good_quality'] = False
        result['issues'].append('glare')
        return result


# Hàm mới để cải thiện chất lượng ảnh dựa trên kết quả kiểm tra
def enhance_image_quality(image, quality_info):
    """
    Cải thiện chất lượng ảnh dựa trên các vấn đề được phát hiện
    
    Args:
        image: Ảnh cần cải thiện
        quality_info: Thông tin về chất lượng ảnh từ hàm check_image_quality
        
    Returns:
        enhanced_image: Ảnh đã được cải thiện
    """
    # Đảm bảo ảnh đã chuyển sang grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ảnh sau khi cải thiện
    enhanced = gray.copy()
    
    if quality_info['has_moire']:
        # Áp dụng bộ lọc fastNlMeansDenoisingColored
        # denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)

        # Tạo hiệu ứng Unsharp Mask:
        # Công thức: sharpened = (1 + amount)*img - amount*blurred
        # amount = 0.3  # điều chỉnh mức tăng nét (có thể từ 0.3 đến 1.0)
        # blurred = cv2.GaussianBlur(denoised, (9, 9), 10)
        # sharpened = cv2.addWeighted(denoised, 1 + amount, blurred, -amount, 0)
        # enhanced = sharpened
        pass
    # 1. Xử lý khi ảnh bị mờ
    if 'blurry' in quality_info['issues']:
        # Áp dụng bộ lọc làm sắc nét (sharpening filter)
        kernel = np.array([[-1, -1, -1],
                           [-1, 10, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # 2. Xử lý khi ảnh quá sáng
    if 'too_bright' in quality_info['issues']:
        # Giảm độ sáng bằng cách giảm giá trị pixel
        alpha = 0.9  # Điều chỉnh độ sáng (< 1 làm tối, > 1 làm sáng)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=0)
    
    # 3. Xử lý khi ảnh có độ tương phản thấp
    if 'low_contrast' in quality_info['issues']:
        # Tăng độ tương phản bằng CLAHE
        # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(18, 18))
        # enhanced = clahe.apply(enhanced)
        
        # Convert numpy array to PIL Image
        # enhanced_pil = Image.fromarray(enhanced)
        # enhancer = ImageEnhance.Contrast(enhanced_pil)
        # enhanced_pil = enhancer.enhance(2.0)
        # Convert back to numpy array
        # enhanced = np.array(enhanced_pil)
        pass
    # 4. Xử lý khi ảnh bị chói (glare)
    if quality_info['has_glare']:
        # Áp dụng ngưỡng thích ứng để giảm tác động của vùng quá sáng
        enhanced = cv2.adaptiveThreshold(
            enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            blockSize=11, 
            C=2
        )
    
    # 5. Xử lý khi ảnh có moire pattern
    
    return enhanced

# Thêm các hàm phát hiện màn hình HMI từ hmi_image_detector.py
def enhance_image(image):
    """Cải thiện chất lượng ảnh trước khi phát hiện cạnh"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Tăng độ tương phản với PIL
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_pil = enhancer.enhance(2.0)  # Tăng độ tương phản lên 100%
    
    # Chuyển lại về định dạng OpenCV
    enhanced_image = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
    
    # Tiếp tục quy trình xử lý ảnh như trước
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    # Tăng clip limit để cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(11, 11))  # Tăng từ 3.0 lên 4.0
    enhanced = clahe.apply(gray)
    
    # Tăng độ tương phản
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=0)  # Thêm bước tăng contrast
    
    # Làm mịn ảnh với kernel nhỏ hơn để giữ nguyên cạnh sắc nét hơn
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)  # Giảm từ (7, 7) xuống (5, 5)
    return blurred, enhanced

def adaptive_edge_detection(image):
    """Phát hiện cạnh với nhiều phương pháp và kết hợp kết quả"""
    median_val = np.median(image)
    # Giảm ngưỡng để tăng độ nhạy cảm phát hiện cạnh
    lower = int(max(0, (1.0 - 0.33) * median_val))  # Giảm từ 0.25 xuống 0.33
    upper = int(min(255, (1.0 + 0.33) * median_val))  # Tăng từ 0.25 lên 0.33
    canny_edges = cv2.Canny(image, lower, upper)
    
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Giảm ngưỡng sobel để bắt được nhiều cạnh hơn
    _, sobel_edges = cv2.threshold(sobel_edges, 80, 255, cv2.THRESH_BINARY)  # Giảm từ 50 xuống 40
    
    combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
    
    # Tăng số lần giãn nở để kết nối các cạnh bị đứt đoạn
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=2)  # Tăng từ 1 lên 2
    final_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    
    return canny_edges, sobel_edges, final_edges

def process_lines(lines, img_shape, min_length=20, max_lines_per_direction=30):
    """Xử lý và nhóm các đường thẳng theo hướng ngang/dọc, giới hạn số lượng đường"""
    if lines is None:
        return [], []
    
    horizontal_lines = []
    vertical_lines = []
    
    all_h_lines = []
    all_v_lines = []
    
    height, width = img_shape[:2]
    min_dimension = min(height, width)
    
    # Giảm độ dài tối thiểu để phát hiện nhiều đường hơn
    min_length = max(min_length, int(min_dimension * 0.02))  # Giảm từ 0.03 xuống 0.02
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if length < min_length:
            continue
        
        if x2 != x1:
            angle = degrees(atan2(y2-y1, x2-x1))
        else:
            angle = 90
        
        # Mở rộng phạm vi phân loại đường ngang/dọc
        if abs(angle) < 40 or abs(angle) > 140:  # Đường ngang (mở rộng phạm vi từ 35 lên 40)
            all_h_lines.append([x1, y1, x2, y2, angle, length])
        elif abs(angle - 90) < 40 or abs(angle + 90) < 40:  # Đường dọc (mở rộng phạm vi từ 35 lên 40)
            all_v_lines.append([x1, y1, x2, y2, angle, length])
    
    all_h_lines.sort(key=lambda x: x[5], reverse=True)
    all_v_lines.sort(key=lambda x: x[5], reverse=True)
    
    # Đảm bảo có đủ số lượng đường ngang và dọc tối thiểu
    min_lines = min(4, len(all_h_lines))  # Tăng số lượng dòng tối thiểu từ 3 lên 4
    horizontal_lines = [line[:5] for line in all_h_lines[:max(min_lines, max_lines_per_direction)]]
    
    min_lines = min(4, len(all_v_lines))  # Tăng số lượng dòng tối thiểu từ 3 lên 4
    vertical_lines = [line[:5] for line in all_v_lines[:max(min_lines, max_lines_per_direction)]]
    
    return horizontal_lines, vertical_lines

def find_largest_rectangle(intersections, img_shape):
    """Tìm hình chữ nhật lớn nhất từ các giao điểm"""
    if len(intersections) < 4:
        return None
    
    left_point = min(intersections, key=lambda p: p[0])
    right_point = max(intersections, key=lambda p: p[0])
    top_point = min(intersections, key=lambda p: p[1])
    bottom_point = max(intersections, key=lambda p: p[1])
    
    top_left = (left_point[0], top_point[1])
    top_right = (right_point[0], top_point[1])
    bottom_left = (left_point[0], bottom_point[1])
    bottom_right = (right_point[0], bottom_point[1])
    
    threshold = 30
    
    def find_nearest_intersection(point):
        nearest = min(intersections, key=lambda p: (p[0]-point[0])**2 + (p[1]-point[1])**2)
        distance = sqrt((nearest[0]-point[0])**2 + (nearest[1]-point[1])**2)
        if distance < threshold:
            return nearest
        return point
    
    refined_top_left = find_nearest_intersection(top_left)
    refined_top_right = find_nearest_intersection(top_right)
    refined_bottom_left = find_nearest_intersection(bottom_left)
    refined_bottom_right = find_nearest_intersection(bottom_right)
    
    width = refined_top_right[0] - refined_top_left[0]
    height = refined_bottom_left[1] - refined_top_left[1]
    area = width * height
    
    height_img, width_img = img_shape[:2]
    total_area = height_img * width_img
    
    if area < 0.01 * total_area or area > 0.9 * total_area:
        return None
    
    if width <= 0 or height <= 0:
        return None
    
    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
    if aspect_ratio > 5:
        return None
    
    return (refined_top_left, refined_top_right, refined_bottom_right, refined_bottom_left, area)

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def find_rectangle_from_classified_lines(horizontal_lines, vertical_lines, img_shape):
    """Tìm hình chữ nhật từ các đường đã phân loại ngang và dọc"""
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
    
    top_left_x = max(min(top_line[0], top_line[2]), left_x)
    top_right_x = min(max(top_line[0], top_line[2]), right_x)
    bottom_left_x = max(min(bottom_line[0], bottom_line[2]), left_x)
    bottom_right_x = min(max(bottom_line[0], bottom_line[2]), right_x)
    
    left_top_y = max(min(left_line[1], left_line[3]), top_y)
    left_bottom_y = min(max(left_line[1], left_line[3]), bottom_y)
    right_top_y = max(min(right_line[1], right_line[3]), top_y)
    right_bottom_y = min(max(right_line[1], right_line[3]), bottom_y)
    
    if (top_right_x - top_left_x < 10 or bottom_right_x - bottom_left_x < 10 or
        left_bottom_y - left_top_y < 10 or right_bottom_y - right_top_y < 10):
        return None
    
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
    
    top_left = (int(left_x), int(top_y))
    top_right = (int(right_x), int(top_y))
    bottom_right = (int(right_x), int(bottom_y))
    bottom_left = (int(left_x), int(bottom_y))
    
    area = rect_width * rect_height
    
    total_area = height * width
    if area < 0.01 * total_area or area > 0.9 * total_area:
        return None
    
    return (top_left, top_right, bottom_right, bottom_left, area)

def extend_lines(lines, width, height):
    """Kéo dài các đường thẳng đến biên của ảnh"""
    extended_lines = []
    
    for x1, y1, x2, y2, angle in lines:
        # Xử lý đường dọc (x không đổi)
        if abs(x2 - x1) < 5:  # Đường dọc hoặc gần dọc
            extended_lines.append([x1, 0, x1, height - 1, angle])
            continue
            
        # Xử lý đường ngang (y không đổi)
        if abs(y2 - y1) < 5:  # Đường ngang hoặc gần ngang
            extended_lines.append([0, y1, width - 1, y1, angle])
            continue
        
        # Xử lý các đường xiên
        m = (y2 - y1) / (x2 - x1)  # Hệ số góc
        b = y1 - m * x1  # Hệ số tự do
        
        # Tính toán giao điểm với các cạnh của ảnh
        intersections = []
        
        # Giao với cạnh trái (x=0)
        y_left = m * 0 + b
        if 0 <= y_left < height:
            intersections.append((0, int(y_left)))
            
        # Giao với cạnh phải (x=width-1)
        y_right = m * (width - 1) + b
        if 0 <= y_right < height:
            intersections.append((width - 1, int(y_right)))
            
        # Giao với cạnh trên (y=0)
        if abs(m) > 1e-10:  # Tránh chia cho số quá nhỏ
            x_top = (0 - b) / m
            if 0 <= x_top < width:
                intersections.append((int(x_top), 0))
            
        # Giao với cạnh dưới (y=height-1)
        if abs(m) > 1e-10:  # Tránh chia cho số quá nhỏ
            x_bottom = ((height - 1) - b) / m
            if 0 <= x_bottom < width:
                intersections.append((int(x_bottom), height - 1))
        
        # Nếu có đủ hai giao điểm, tạo đường kéo dài
        if len(intersections) >= 2:
            # Lấy hai giao điểm đầu tiên
            p1, p2 = intersections[:2]
            extended_lines.append([p1[0], p1[1], p2[0], p2[1], angle])
    
    return extended_lines

def find_intersections(horizontal_lines, vertical_lines, max_intersections=200):
    """Tìm giao điểm của các đường ngang và dọc"""
    intersections = []
    
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            if len(intersections) >= max_intersections:
                break
                
            x1_h, y1_h, x2_h, y2_h, _ = h_line
            x1_v, y1_v, x2_v, y2_v, _ = v_line
            
            # Xử lý trường hợp đặc biệt của đường ngang và dọc
            if abs(y1_h - y2_h) < 5 and abs(x1_v - x2_v) < 5:
                # Giao điểm của đường ngang thuần túy và đường dọc thuần túy
                intersections.append((int(x1_v), int(y1_h)))
                continue
            
            try:
                # Chuyển sang float để tránh tràn số
                x1_h, y1_h, x2_h, y2_h = float(x1_h), float(y1_h), float(x2_h), float(y2_h)
                x1_v, y1_v, x2_v, y2_v = float(x1_v), float(y1_v), float(x2_v), float(y2_v)
                
                # Kiểm tra nếu đường ngang gần như ngang
                if abs(y2_h - y1_h) < 1e-10:
                    if abs(x2_v - x1_v) < 1e-10:
                        x_intersect = x1_v
                    else:
                        t = (y1_h - y1_v) / (y2_v - y1_v)
                        x_intersect = x1_v + t * (x2_v - x1_v)
                    
                    intersections.append((int(x_intersect), int(y1_h)))
                    continue
                
                # Kiểm tra nếu đường dọc gần như dọc
                if abs(x2_v - x1_v) < 1e-10:
                    if abs(x2_h - x1_h) < 1e-10:
                        y_intersect = y1_h
                    else:
                        t = (x1_v - x1_h) / (x2_h - x1_h)
                        y_intersect = y1_h + t * (y2_h - y1_h)
                    
                    intersections.append((int(x1_v), int(y_intersect)))
                    continue
                
                denom = (y2_v - y1_v) * (x2_h - x1_h) - (x2_v - x1_v) * (y2_h - y1_h)
                
                if abs(denom) < 1e-10:
                    continue
                
                # Tính tham số t cho đường 1
                ua = ((x2_v - x1_v) * (y1_h - y1_v) - (y2_v - y1_v) * (x1_h - x1_v)) / denom
                
                # Tính tọa độ giao điểm
                x_intersect = x1_h + ua * (x2_h - x1_h)
                y_intersect = y1_h + ua * (y2_h - y1_h)
                
                # Kiểm tra giao điểm có nằm trong đoạn đường không
                if (min(x1_h, x2_h) - 10 <= x_intersect <= max(x1_h, x2_h) + 10 and
                    min(y1_v, y2_v) - 10 <= y_intersect <= max(y1_v, y2_v) + 10):
                    intersections.append((int(x_intersect), int(y_intersect)))
            
            except (ValueError, OverflowError, ZeroDivisionError) as e:
                continue
    
    return intersections

def detect_hmi_screen(image):
    """Phát hiện màn hình HMI trong ảnh và trả về vùng đã cắt"""
    # Tạo bản sao để vẽ kết quả
    result_image = image.copy()
    
    # Bước 1: Tăng cường chất lượng ảnh
    enhanced_img, enhanced_clahe = enhance_image(image)
    
    # Bước 2: Phát hiện cạnh
    canny_edges, sobel_edges, edges = adaptive_edge_detection(enhanced_clahe)
    
    # Bước 3: Tìm contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contour theo diện tích
    min_contour_area = image.shape[0] * image.shape[1] * 0.001
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Tạo contour mask
    contour_mask = np.zeros_like(edges)
    cv2.drawContours(contour_mask, large_contours, -1, 255, 2)
    
    # Bước 4: Phát hiện đường thẳng - Điều chỉnh các tham số
    lines = cv2.HoughLinesP(contour_mask, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=30)  # Giảm threshold, minLineLength và tăng maxLineGap

    # Nếu không tìm được đường thẳng, thử điều chỉnh tham số
    if lines is None or len(lines) < 2:
        # Thử với các tham số dễ dàng hơn
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=10, maxLineGap=40)
        
        if lines is None or len(lines) < 2:
            # Thử lần cuối với các tham số rất dễ dàng
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=50)
    
    if lines is None:
        print("Không tìm thấy đường thẳng trong ảnh.")
        return None, result_image, None
    
    # Bước 5: Phân loại đường ngang/dọc
    height, width = image.shape[:2]
    horizontal_lines, vertical_lines = process_lines(lines, image.shape, min_length=20)
    
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        print("Không tìm thấy đủ đường ngang và dọc.")
        return None, result_image, None
    
    # PHẦN MỚI: Thử tìm hình chữ nhật từ các đường đã phân loại
    largest_rectangle = find_rectangle_from_classified_lines(horizontal_lines, vertical_lines, image.shape)
    
    # Nếu không tìm được hình chữ nhật từ các đường đã phân loại, tiếp tục với quy trình thông thường
    if largest_rectangle is None:
        print("Không tìm thấy hình chữ nhật từ các đường đã phân loại, tiếp tục với quy trình thông thường...")
        
        # Bước 6: Kéo dài đường
        extended_h_lines = extend_lines(horizontal_lines, width, height)
        extended_v_lines = extend_lines(vertical_lines, width, height)
        
        # Bước 7: Tìm giao điểm
        intersections = find_intersections(extended_h_lines, extended_v_lines)
        
        if len(intersections) < 4:
            print("Không tìm thấy đủ giao điểm để tạo hình chữ nhật.")
            return None, result_image, None
        
        # Bước 8: Tìm hình chữ nhật lớn nhất từ các giao điểm xa nhất
        largest_rectangle = find_largest_rectangle(intersections, image.shape)
        
        if largest_rectangle is None:
            print("Không tìm thấy hình chữ nhật phù hợp.")
            return None, result_image, None
    
    # Lấy các góc của hình chữ nhật
    top_left, top_right, bottom_right, bottom_left, _ = largest_rectangle
    
    # Tính tọa độ của vùng HMI
    x_min = min(top_left[0], bottom_left[0])
    y_min = min(top_left[1], top_right[1])
    x_max = max(top_right[0], bottom_right[0])
    y_max = max(bottom_left[1], bottom_right[1])
    
    # Kiểm tra biên
    if x_min < 0: x_min = 0
    if y_min < 0: y_min = 0
    if x_max >= image.shape[1]: x_max = image.shape[1] - 1
    if y_max >= image.shape[0]: y_max = image.shape[0] - 1
    
    # Cắt vùng HMI
    hmi_screen = None
    roi_coords = None
    
    if x_max > x_min and y_max > y_min:
        roi_coords = (x_min, y_min, x_max, y_max)
        
        # Vẽ hình chữ nhật lên ảnh kết quả
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Cắt vùng HMI thô
        roi = image[y_min:y_max, x_min:x_max]
        
        # THÊM MỚI: Tinh chỉnh và trải phẳng vùng HMI
        warped_roi, refined_coords = fine_tune_hmi_screen(image, roi_coords)
        
        # Sử dụng ảnh đã tinh chỉnh
        hmi_screen = warped_roi
        
        # Lưu kết quả phát hiện
        print(f"Đã phát hiện và tinh chỉnh màn hình HMI: x={x_min}, y={y_min}, width={x_max-x_min}, height={y_max-y_min}")
    
    return hmi_screen, result_image, roi_coords

def extract_content_region(img):
    """
    Trích xuất vùng nội dung (không phải vùng đen xung quanh màn hình sử dụng gradient và kernel theo chiều dọc
    """
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tăng độ tương phản để làm nổi bật đường viền màn hình
    enhanced_contrast = cv2.convertScaleAbs(gray, alpha=1.3, beta=5)
    
    # Làm mịn ảnh để giảm nhiễu nhưng vẫn giữ được cạnh
    blurred = cv2.GaussianBlur(enhanced_contrast, (3, 3), 0)  # Kernel nhỏ hơn để giữ được cạnh
    
    # Phân tích gradient để tìm các vùng có độ tương phản cao
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobel_x, sobel_y)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Ngưỡng gradient - sử dụng ngưỡng thấp hơn để bắt được nhiều cạnh hơn
    _, gradient_thresh = cv2.threshold(gradient_mag, 20, 255, cv2.THRESH_BINARY)  # Giảm ngưỡng từ 30 xuống 20
    
    # Tạo kernel theo chiều dọc cao hơn để bắt được toàn bộ các cạnh dọc
    vertical_kernel = np.ones((11, 3), np.uint8)  # Tăng từ (9, 3) lên (11, 3)
    
    # Mở rộng các cạnh theo chiều dọc
    gradient_dilated = cv2.dilate(gradient_thresh, vertical_kernel, iterations=3)  # Tăng iterations từ 2 lên 3
    
    # Đảm bảo kết nối tốt theo chiều ngang
    horizontal_kernel = np.ones((3, 9), np.uint8)  # Tăng từ (3, 7) lên (3, 9)
    gradient_dilated = cv2.dilate(gradient_dilated, horizontal_kernel, iterations=2)  # Tăng iterations từ 1 lên 2
    
    # Làm mịn và loại bỏ nhiễu
    kernel = np.ones((5, 5), np.uint8)
    gradient_final = cv2.morphologyEx(gradient_dilated, cv2.MORPH_CLOSE, kernel, iterations=3)  # Tăng từ 2 lên 3
    
    # Tìm contour trực tiếp từ ảnh gradient
    contours, _ = cv2.findContours(gradient_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu không tìm thấy contour, thử với phương pháp ngưỡng
    if not contours:
        print("Không tìm thấy contour từ gradient, chuyển sang phương pháp ngưỡng")
        # Áp dụng phương pháp ngưỡng tự động bằng Otsu
        # Trước tiên, tăng độ tương phản
        enhanced_for_threshold = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        _, thresh = cv2.threshold(enhanced_for_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc ra các contour lớn (diện tích > 0.5% của ảnh)
    min_area = img.shape[0] * img.shape[1] * 0.005  # Giảm từ 1% xuống 0.5%
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Tạo mask từ contour lớn nhất
    mask = np.zeros_like(gray)
    if large_contours:
        largest_contour = max(large_contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)  # Vẽ đầy contour
    else:
        print("Không tìm thấy contour lớn, trả về mask đầy")
        mask.fill(255)  # Trả về mask đầy nếu không tìm thấy contour
    
    return mask, large_contours[0] if large_contours else None

def fine_tune_hmi_screen(image, roi_coords):
    """
    Tinh chỉnh vùng màn hình HMI đã phát hiện:
    1. Loại bỏ vùng đen xung quanh màn hình sử dụng gradient và kernel theo chiều dọc
    2. Áp dụng Warp Perspective trực tiếp trên contour lớn nhất
    """
    x_min, y_min, x_max, y_max = roi_coords
    roi = image[y_min:y_max, x_min:x_max]
    
    # THAY ĐỔI: Tìm vùng nội dung và lấy contour lớn nhất trực tiếp
    content_mask, largest_contour = extract_content_region(roi)
    
    # Kiểm tra nếu không tìm được contour
    if largest_contour is None:
        print("Không tìm thấy contour lớn trong ROI")
        return roi, roi_coords
    
    # Kiểm tra diện tích contour
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < 0.1 * roi.shape[0] * roi.shape[1]:
        print("Vùng nội dung quá nhỏ, có thể không phải là màn hình HMI")
        return roi, roi_coords
    
    # Xấp xỉ contour thành đa giác
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Nếu không có đúng 4 điểm, điều chỉnh để có 4 điểm
    if len(approx) != 4:
        # Sử dụng hình chữ nhật bao quanh tối thiểu
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = np.array(box, dtype=np.int32)
    
    # Chuyển đổi sang mảng điểm
    points = approx.reshape(-1, 2)
    
    # Sắp xếp các điểm để chuẩn bị cho biến đổi phối cảnh
    points = order_points(points)
    
    # Tính toán chiều rộng và chiều cao của màn hình đích
    # Sử dụng khoảng cách Euclidean
    width_a = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    width_b = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    height_b = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Đảm bảo kích thước hợp lý
    if max_width < 10 or max_height < 10:
        print("Kích thước màn hình HMI quá nhỏ")
        return roi, roi_coords
    
    # Tạo điểm đích cho biến đổi phối cảnh
    dst_points = np.array([
        [0, 0],                     # top-left
        [max_width - 1, 0],         # top-right
        [max_width - 1, max_height - 1],  # bottom-right
        [0, max_height - 1]         # bottom-left
    ], dtype=np.float32)
    
    # Chuyển đổi points sang float32
    src_points = points.astype(np.float32)
    
    # Thực hiện biến đổi phối cảnh
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(roi, M, (max_width, max_height))
    
    # Tính toán tọa độ mới
    new_roi_coords = (x_min, y_min, x_min + warped.shape[1], y_min + warped.shape[0])
    
    return warped, new_roi_coords

@app.route('/api/history', methods=['GET'])
def get_ocr_history():
    """
    API để truy vấn kết quả OCR trong khoảng thời gian chỉ định
    Tham số truy vấn:
    - ExpectedStartTime: Thời gian bắt đầu theo định dạng YYYY-MM-DD
    - ExpectedEndTime: Thời gian kết thúc theo định dạng YYYY-MM-DD
    """
    # Lấy tham số truy vấn
    start_time_str = request.args.get('ExpectedStartTime')
    end_time_str = request.args.get('ExpectedEndTime')
    
    # Kiểm tra tham số
    if not start_time_str or not end_time_str:
        return jsonify({"error": "ExpectedStartTime và ExpectedEndTime là bắt buộc"}), 400
    
    try:
        # Chuyển đổi chuỗi thời gian thành đối tượng datetime
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d")
        end_time = datetime.strptime(end_time_str, "%Y-%m-%d")
        
        # Đảm bảo thời gian bắt đầu không lớn hơn thời gian kết thúc
        if start_time > end_time:
            return jsonify({"error": "ExpectedStartTime không thể lớn hơn ExpectedEndTime"}), 400
    except ValueError:
        return jsonify({"error": "Định dạng thời gian không hợp lệ. Vui lòng sử dụng định dạng YYYY-MM-DD"}), 400
    
    # Đọc dữ liệu thông tin máy từ machine_screens.json
    machine_info = {}
    try:
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if os.path.exists(machine_screens_path):
            with open(machine_screens_path, 'r', encoding='utf-8') as f:
                machines_data = json.load(f)
                for machine_code, machine_data in machines_data.get('machines', {}).items():
                    machine_info[machine_code] = machine_data.get('name', f"Máy {machine_code}")
    except Exception as e:
        print(f"Lỗi khi đọc file machine_screens.json: {str(e)}")
    
    # Thư mục chứa kết quả OCR
    ocr_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_results')
    
    # Danh sách kết quả OCR
    ocr_results = []
    
    # Duyệt qua tất cả các file trong thư mục ocr_results
    for filename in os.listdir(ocr_results_dir):
        file_path = os.path.join(ocr_results_dir, filename)
        
        # Kiểm tra nếu là file json
        if os.path.isfile(file_path) and filename.endswith('.json'):
            try:
                # Lấy thời gian tạo file
                file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                
                # Kiểm tra xem thời gian tạo file có nằm trong khoảng thời gian chỉ định không
                if start_time <= file_creation_time <= end_time:
                    # Đọc nội dung file JSON
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Thêm thông tin thời gian và tên file vào dữ liệu
                    data['timestamp'] = file_creation_time.strftime("%Y-%m-%d %H:%M:%S")
                    data['datetime_obj'] = file_creation_time  # Thêm đối tượng datetime để sắp xếp
                    data['filename'] = filename
                    
                    # Thêm machine_name từ thông tin máy đã đọc
                    if 'machine_code' in data and data['machine_code'] in machine_info:
                        data['machine_name'] = machine_info[data['machine_code']]
                    
                    # Thêm vào danh sách kết quả
                    ocr_results.append(data)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Lỗi khi đọc file {filename}: {str(e)}")
    
    # Sắp xếp kết quả theo khoảng cách đến ExpectedEndTime (gần nhất trước)
    ocr_results.sort(key=lambda x: abs((x['datetime_obj'] - end_time).total_seconds()))
    
    # Tạo đối tượng kết quả mới với key là các số từ 0 trở đi
    indexed_results = {}
    for i, result in enumerate(ocr_results):
        # Loại bỏ trường datetime_obj trước khi trả về kết quả
        del result['datetime_obj']
        indexed_results[str(i)] = result
    
    return jsonify({
        **indexed_results  # Thêm các kết quả đã đánh số vào response
    })

# API mới: Cập nhật máy và màn hình
@app.route('/api/update_machine_screen', methods=['POST'])
def update_machine_screen():
    """
    Cập nhật machine_order và parameter_order dựa trên mã máy và tên màn hình
    
    Cấu trúc dữ liệu đầu vào:
    Form data với các key:
    - machine_code: Mã máy (ví dụ: "F1")
    - screen_id: Tên của màn hình (chuỗi, ví dụ: "Production")
    """
    try:
        # Kiểm tra đầu vào
        if 'machine_code' not in request.form or 'screen_id' not in request.form:
            return jsonify({
                "error": "Missing required fields. Please provide machine_code and screen_id in form-data"
            }), 400
        
        machine_code = request.form['machine_code'].strip().upper()
        screen_name = request.form['screen_id'].strip()  # screen_id giờ là tên màn hình
        area = request.form.get('area', '').strip().upper()
        
        # Nếu không có area, thử lấy từ machine_code
        if not area:
            area = get_area_for_machine(machine_code)
            if not area:
                return jsonify({
                    "error": "Could not determine area for this machine_code. Please provide area parameter."
                }), 400
        
        # Kiểm tra tính hợp lệ của khu vực, mã máy và tên màn hình
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if area not in data.get('areas', {}):
            return jsonify({"error": f"Area {area} not found"}), 404
            
        if machine_code not in data['areas'][area].get('machines', {}):
            return jsonify({"error": f"Machine {machine_code} not found in area {area}"}), 404
        
        # Lấy loại máy
        machine_type = data['areas'][area]['machines'][machine_code].get('type')
        if not machine_type or machine_type not in data.get('machine_types', {}):
            return jsonify({"error": f"Machine type not found for machine {machine_code}"}), 404
        
        # Tìm màn hình có tên trùng khớp và lấy ID số
        screen_numeric_id = None
        selected_screen = None
        for screen in data['machine_types'][machine_type].get('screens', []):
            if screen['screen_id'] == screen_name:
                screen_numeric_id = screen['id']
                selected_screen = screen
                break
        
        if not screen_numeric_id:
            return jsonify({"error": f"Screen '{screen_name}' not found for machine {machine_code} (type: {machine_type})"}), 404
        
        # Cập nhật parameter_order_value.txt với ID số của màn hình
        parameter_order_file_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'parameter_order_value.txt')
        with open(parameter_order_file_path, 'w', encoding='utf-8') as f:
            f.write(str(screen_numeric_id))
        
        return jsonify({
            "message": "Machine and screen selection updated successfully",
            "area": {
                "area_code": area,
                "name": data['areas'][area]['name']
            },
            "machine": {
                "machine_code": machine_code,
                "name": data['areas'][area]['machines'][machine_code]['name'],
                "type": machine_type
            },
            "screen": {
                "id": screen_numeric_id,
                "screen_id": selected_screen['screen_id'],
                "description": selected_screen.get('description', '')
            }
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to update machine and screen selection: {str(e)}"}), 500

# Thêm các hàm compare_images từ hmi_image_detector.py
def compare_histograms(img1, img2):
    """So sánh histogram giữa hai ảnh (legacy function)"""
    # Chuyển sang không gian màu HSV để giảm ảnh hưởng của độ sáng
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Tính histogram cho hai ảnh
    hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Chuẩn hóa histogram
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # So sánh histogram
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # Correlation (1 = hoàn hảo)
    
    return correlation

def compare_histograms_optimized(img1, img2):
    """
    [*] Optimized histogram comparison for auto detection
    
    Improvements:
    - Multi-channel histogram analysis
    - Reduced bins for faster computation
    - Combined color and texture features
    """
    try:
        # Convert to HSV for better color representation
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Reduce bins for faster computation while maintaining accuracy
        # H: 32 bins, S: 32 bins (original was 180, 256)
        hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Use correlation (best for template matching)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Ensure result is in [0, 1] range
        return max(0, correlation)
        
    except Exception as e:
        print(f"Error in optimized histogram comparison: {e}")
        return 0

def compare_features_orb(img1, img2, max_features=500):
    """So sánh hai ảnh dựa trên đặc trưng ORB (Oriented FAST và Rotated BRIEF)"""
    # Chuyển sang grayscale để phát hiện đặc trưng
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Kiểm tra ORB detector global có khả dụng không
    if orb_detector is None:
        print("ORB detector not available")
        return 0
    
    # Tìm keypoints và descriptors sử dụng ORB detector global
    kp1, des1 = orb_detector.detectAndCompute(img1_gray, None)
    kp2, des2 = orb_detector.detectAndCompute(img2_gray, None)
    
    # Kiểm tra nếu không tìm thấy đặc trưng
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0
    
    # Kiểm tra BFMatcher global có khả dụng không
    if bf_matcher is None:
        print("BFMatcher not available")
        return 0
    
    # Tìm các matches sử dụng BFMatcher global
    matches = bf_matcher.match(des1, des2)
    
    # Sắp xếp các matches theo khoảng cách (thấp hơn = tốt hơn)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Tính điểm: số lượng matches tốt và chất lượng của chúng
    good_matches = [m for m in matches if m.distance < 50]  # Chỉ lấy matches có khoảng cách < 50
    
    # Tỷ lệ matches tốt so với tổng số đặc trưng
    match_ratio = len(good_matches) / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
    
    # Điểm chất lượng (0-1): tỷ lệ matches tốt
    return match_ratio

def compare_phash(img1, img2, hash_size=16):
    """So sánh hai ảnh dựa trên perceptual hash"""
    from PIL import Image
    import numpy as np
    from scipy.fftpack import dct
    
    # Hàm tính perceptual hash
    def calculate_phash(image, hash_size=16):
        # Chuyển sang grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize ảnh
        resized = cv2.resize(image, (hash_size, hash_size))
        
        # Chuyển sang float và tính DCT
        dct_data = dct(dct(resized, axis=0), axis=1)
        
        # Lấy vùng tần số thấp (góc trên bên trái)
        dct_low_freq = dct_data[:8, :8]
        
        # Tính trung bình của vùng tần số thấp (bỏ qua DC coefficient)
        med = np.median(dct_low_freq)
        
        # Tạo hash từ so sánh với giá trị trung bình
        hash_bits = (dct_low_freq > med).flatten()
        
        # Chuyển sang số nguyên 64-bit
        hash_value = 0
        for bit in hash_bits:
            hash_value = (hash_value << 1) | int(bit)
            
        return hash_bits
    
    # Tính hash cho hai ảnh
    hash1 = calculate_phash(img1, hash_size)
    hash2 = calculate_phash(img2, hash_size)
    
    # Tính khoảng cách Hamming
    hamming_distance = np.sum(hash1 != hash2)
    
    # Chuyển đổi khoảng cách thành độ tương đồng (0-1)
    similarity = 1 - (hamming_distance / (hash_size * hash_size))
    
    return similarity

def find_best_matching_template(hmi_image, reference_dir, machine_type=None):
    """
    🔄 LEGACY: Tìm template phù hợp nhất với ảnh HMI (deprecated)
    
    [WARNING] Function này được giữ lại để tương thích ngược, 
        nhưng auto_detect_machine_and_screen() đã được tối ưu hóa hoàn toàn
    
    Args:
        hmi_image: Ảnh cần so sánh
        reference_dir: Đường dẫn thư mục chứa ảnh tham chiếu
        machine_type: Loại máy để lọc ảnh tham chiếu
        
    Returns:
        Tuple (best_match_path, best_match_screen_id, similarity_score)
    """
    print("[WARNING] WARNING: Using legacy find_best_matching_template(). Consider using optimized auto_detect_machine_and_screen()")
    
    if not os.path.exists(reference_dir):
        print(f"Thư mục reference không tồn tại: {reference_dir}")
        return None, None, 0
    
    # Lọc các file theo loại máy (nếu có)
    template_files = []
    for filename in os.listdir(reference_dir):
        # Chỉ xử lý file ảnh
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Nếu có chỉ định machine_type, chỉ lấy các template tương ứng
        if machine_type and f"template_{machine_type}_" not in filename:
            continue
            
        template_files.append(filename)
    
    if not template_files:
        print(f"Không tìm thấy template phù hợp với loại máy {machine_type}")
        return None, None, 0
    
    best_match = None
    best_score = -1
    best_screen_id = None
    
    print(f"Bắt đầu so sánh với {len(template_files)} template...")
    
    for template_file in template_files:
        template_path = os.path.join(reference_dir, template_file)
        
        # Đọc ảnh template
        template_img = cv2.imread(template_path)
        if template_img is None:
            print(f"Không thể đọc file template: {template_path}")
            continue
        
        # So sánh kích thước của ảnh
        img_height, img_width = hmi_image.shape[:2]
        templ_height, templ_width = template_img.shape[:2]
        
        # Nếu kích thước quá khác nhau, điều chỉnh template
        if abs(img_height/img_width - templ_height/templ_width) > 0.3:
            print(f"Tỷ lệ khung hình quá khác biệt cho {template_file}, điều chỉnh...")
            template_img = cv2.resize(template_img, (img_width, img_height))
        
        # Kết hợp các phương pháp so sánh
        hist_score = compare_histograms(hmi_image, template_img)
        feature_score = compare_features_orb(hmi_image, template_img)
        phash_score = compare_phash(hmi_image, template_img)
        
        # Tính điểm tổng hợp (có thể điều chỉnh trọng số)
        combined_score = 0.3 * hist_score + 0.4 * feature_score + 0.3 * phash_score
        
        print(f"Template {template_file}: hist={hist_score:.2f}, feature={feature_score:.2f}, phash={phash_score:.2f}, combined={combined_score:.2f}")
        
        # Cập nhật best match
        if combined_score > best_score:
            best_score = combined_score
            best_match = template_path
            
            # Trích xuất screen_id từ tên file template
            # Format: template_{machine_type}_{screen_name}.png
            parts = template_file.split('_')
            if len(parts) >= 3:
                # Lấy tất cả phần từ index 2 trở đi và bỏ phần mở rộng
                screen_name = '_'.join(parts[2:]).rsplit('.', 1)[0]
                best_screen_id = screen_name
    
    print(f"Best match: {os.path.basename(best_match) if best_match else 'None'} với điểm {best_score:.2f}, screen_id: {best_screen_id}")
    return best_match, best_screen_id, best_score

def detect_screen_by_template_matching(image, machine_type):
    """
    Phát hiện loại màn hình dựa trên so sánh với ảnh template
    
    Args:
        image: Ảnh cần phân tích
        machine_type: Loại máy (ví dụ: F1, F41, F42)
        
    Returns:
        Tuple (screen_id, screen_numeric_id, template_path)
    """
    # Đường dẫn thư mục chứa ảnh template mẫu
    reference_dir = app.config['REFERENCE_IMAGES_FOLDER']
    
    # Tìm template phù hợp nhất
    best_template, best_screen_id, similarity = find_best_matching_template(image, reference_dir, machine_type)
    
    if best_template is None or similarity < 0.4:  # Ngưỡng tương đồng tối thiểu
        print(f"Không tìm thấy template phù hợp với mức tương đồng đủ cao (similarity={similarity})")
        return None, None, None
    
    # Tìm screen_numeric_id từ screen_id
    screen_numeric_id = get_screen_numeric_id(machine_type, best_screen_id)
    if screen_numeric_id is None:
        print(f"Không tìm thấy screen_numeric_id cho screen_id={best_screen_id}")
    
    return best_screen_id, screen_numeric_id, best_template

# Thêm hàm get_decimal_places để lấy thông tin số thập phân theo machine_type và screen_id
def get_decimal_places(machine_type, screen_id):
    """
    Lấy thông tin số chữ số thập phân cho các ROI của một màn hình cụ thể
    
    Args:
        machine_type: Loại máy (ví dụ: F1, F41, F42)
        screen_id: Tên màn hình (ví dụ: "Faults", "Production Data")
        
    Returns:
        Dict chứa thông tin số chữ số thập phân cho các ROI, hoặc {} nếu không tìm thấy
    """
    try:
        # Đọc file cấu hình decimal places
        decimal_config_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'decimal_places.json')
        if not os.path.exists(decimal_config_path):
            print(f"Decimal places configuration file not found at {decimal_config_path}")
            return {}
        
        with open(decimal_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Kiểm tra xem machine_type có trong cấu hình không
        if machine_type not in config:
            print(f"Machine type {machine_type} not found in decimal places config")
            return {}
        
        # Kiểm tra xem screen_id có trong cấu hình của machine_type không
        if screen_id not in config[machine_type]:
            print(f"Screen ID {screen_id} not found in decimal places config for machine type {machine_type}")
            return {}
        
        # Trả về cấu hình decimal places cho screen_id
        return config[machine_type][screen_id]
    
    except Exception as e:
        print(f"Error getting decimal places: {str(e)}")
        traceback.print_exc()
        return {}

    
    except Exception as e:
        return jsonify({"error": f"Failed to update machine and screen selection: {str(e)}"}), 500

def find_best_allowed_value_match(ocr_text, allowed_values, debug_roi_name=""):
    """
    Tìm giá trị phù hợp nhất từ allowed_values cho kết quả OCR
    
    Args:
        ocr_text (str): Kết quả OCR gốc
        allowed_values (list): Danh sách giá trị cho phép
        debug_roi_name (str): Tên ROI để debug
    
    Returns:
        tuple: (best_match, confidence_score, method_used)
    """
    if not allowed_values or not ocr_text:
        return None, 0.0, "no_data"
    
    ocr_upper = ocr_text.upper().strip()
    print(f"\n=== FINDING BEST MATCH FOR ROI '{debug_roi_name}' ===")
    print(f"OCR text: '{ocr_text}' -> normalized: '{ocr_upper}'")
    print(f"Allowed values: {allowed_values}")
    
    methods = []
    
    for value in allowed_values:
        value_upper = value.upper().strip()
        
        # 1. Exact match (highest priority)
        if ocr_upper == value_upper:
            print(f"  [OK] EXACT MATCH: '{value}' (score: 1.0)")
            return value, 1.0, "exact_match"
        
        # 2. Levenshtein distance (character-level similarity)
        lev_distance = Levenshtein.distance(ocr_upper, value_upper)
        max_len = max(len(ocr_upper), len(value_upper))
        lev_similarity = 1.0 - (lev_distance / max_len) if max_len > 0 else 0.0
        
        # 3. SequenceMatcher (difflib similarity)
        seq_similarity = SequenceMatcher(None, ocr_upper, value_upper).ratio()
        
        # 4. Substring matching
        substring_score = 0.0
        if ocr_upper in value_upper:
            substring_score = len(ocr_upper) / len(value_upper)
        elif value_upper in ocr_upper:
            substring_score = len(value_upper) / len(ocr_upper)
        
        # 5. Character set overlap (tỷ lệ ký tự chung)
        ocr_chars = set(ocr_upper)
        value_chars = set(value_upper)
        char_overlap = len(ocr_chars.intersection(value_chars)) / len(ocr_chars.union(value_chars)) if ocr_chars.union(value_chars) else 0.0
        
        # 6. Prefix/Suffix matching
        prefix_score = 0.0
        suffix_score = 0.0
        min_len = min(len(ocr_upper), len(value_upper))
        if min_len > 0:
            # Prefix matching
            prefix_match_len = 0
            for i in range(min_len):
                if ocr_upper[i] == value_upper[i]:
                    prefix_match_len += 1
                else:
                    break
            prefix_score = prefix_match_len / min_len
            
            # Suffix matching  
            suffix_match_len = 0
            for i in range(1, min_len + 1):
                if ocr_upper[-i] == value_upper[-i]:
                    suffix_match_len += 1
                else:
                    break
            suffix_score = suffix_match_len / min_len
        
        # 7. Common character confusion handling (I/T, G/C, L/I, etc.)
        confusion_score = calculate_ocr_confusion_similarity(ocr_upper, value_upper)
        
        # 8. Weighted character matching (ưu tiên các ký tự khớp quan trọng)
        weighted_char_score = calculate_weighted_character_similarity(ocr_upper, value_upper)
        
        # 9. Length similarity (tương đối về độ dài)
        length_similarity = 1.0 - abs(len(ocr_upper) - len(value_upper)) / max(len(ocr_upper), len(value_upper))
        
        # 10. Abbreviation matching (text ngắn có thể là viết tắt của text dài)
        abbreviation_score = calculate_abbreviation_similarity(ocr_upper, value_upper)
        
        # 11. Phonetic similarity (âm thanh tương tự)
        phonetic_score = calculate_phonetic_similarity(ocr_upper, value_upper)
        
        # Tính điểm tổng hợp với trọng số điều chỉnh
        composite_score = (
            lev_similarity * 0.18 +       # Levenshtein distance
            seq_similarity * 0.18 +       # Sequence matcher
            substring_score * 0.10 +      # Substring matching
            char_overlap * 0.10 +         # Character overlap
            prefix_score * 0.08 +         # Prefix matching
            suffix_score * 0.03 +         # Suffix matching
            confusion_score * 0.10 +      # OCR confusion handling
            weighted_char_score * 0.07 +  # Weighted character matching
            length_similarity * 0.03 +    # Length similarity
            abbreviation_score * 0.08 +   # Abbreviation matching
            phonetic_score * 0.05         # Phonetic similarity
        )
        
        methods.append({
            'value': value,
            'value_upper': value_upper,
            'lev_similarity': lev_similarity,
            'seq_similarity': seq_similarity,
            'substring_score': substring_score,
            'char_overlap': char_overlap,
            'prefix_score': prefix_score,
            'suffix_score': suffix_score,
            'confusion_score': confusion_score,
            'weighted_char_score': weighted_char_score,
            'length_similarity': length_similarity,
            'abbreviation_score': abbreviation_score,
            'phonetic_score': phonetic_score,
            'composite_score': composite_score
        })
        
        print(f"  📊 '{value}': lev={lev_similarity:.3f}, seq={seq_similarity:.3f}, abbrev={abbreviation_score:.3f}, phone={phonetic_score:.3f} -> composite={composite_score:.3f}")
    
    # Sắp xếp theo composite score
    methods.sort(key=lambda x: x['composite_score'], reverse=True)
    
    if methods:
        best_match = methods[0]
        best_value = best_match['value']
        best_score = best_match['composite_score']
        
        print(f"  🏆 BEST MATCH: '{best_value}' (composite score: {best_score:.3f})")
        
        # Giảm threshold để chấp nhận nhiều match hơn
        if best_score >= 0.20:  
            return best_value, best_score, "composite_similarity"
        else:
            print(f"  [ERROR] SCORE TOO LOW ({best_score:.3f} < 0.20), rejecting match")
            return None, best_score, "low_confidence"
    
    return None, 0.0, "no_match"

def calculate_ocr_confusion_similarity(text1, text2):
    """
    Tính similarity dựa trên các lỗi OCR phổ biến
    I/T, G/C, L/I, O/0, etc.
    """
    # Mapping các ký tự dễ nhầm lẫn trong OCR
    confusion_map = {
        'I': ['1', 'l', 'T', '|'],
        'T': ['I', '1', 'l', '7'],
        'G': ['C', '6', '0', 'O'],
        'C': ['G', '0', 'O'],
        'L': ['I', '1', 'l', '7'],
        'O': ['0', 'Q', 'D'],
        '0': ['O', 'Q', 'D'],
        'S': ['5', '8'],
        '5': ['S', '8'],
        'B': ['8', '6'],
        '8': ['B', '6', 'S'],
        '6': ['G', 'B'],
        'N': ['H'],
        'H': ['N'],
        'A': ['4'],
        '4': ['A'],
    }
    
    if len(text1) != len(text2):
        return 0.0
    
    matches = 0
    for i in range(len(text1)):
        c1, c2 = text1[i], text2[i]
        if c1 == c2:
            matches += 1
        elif c1 in confusion_map and c2 in confusion_map[c1]:
            matches += 0.8  # Partial match for confusable characters
        elif c2 in confusion_map and c1 in confusion_map[c2]:
            matches += 0.8
    
    return matches / len(text1) if text1 else 0.0

def calculate_weighted_character_similarity(text1, text2):
    """
    Tính similarity với trọng số cho các ký tự quan trọng
    Ví dụ: ký tự đầu và ký tự cuối có trọng số cao hơn
    """
    if not text1 or not text2:
        return 0.0
    
    max_len = max(len(text1), len(text2))
    min_len = min(len(text1), len(text2))
    
    total_weight = 0
    matched_weight = 0
    
    # Tính trọng số cho từng vị trí
    for i in range(max_len):
        # Ký tự đầu và cuối có trọng số cao hơn
        if i == 0 or i == max_len - 1:
            weight = 2.0  # Trọng số cao cho ký tự đầu/cuối
        else:
            weight = 1.0  # Trọng số bình thường
        
        total_weight += weight
        
        # Kiểm tra khớp nếu cả hai text có ký tự ở vị trí này
        if i < len(text1) and i < len(text2):
            if text1[i] == text2[i]:
                matched_weight += weight
    
    return matched_weight / total_weight if total_weight > 0 else 0.0

def calculate_abbreviation_similarity(short_text, long_text):
    """
    Tính similarity dựa trên khả năng short_text là viết tắt của long_text
    Ví dụ: "GLNI" có thể là viết tắt của "DAGIANHIET" (các ký tự G, L, N, I xuất hiện)
    """
    if len(short_text) >= len(long_text):
        return 0.0  # Chỉ áp dụng khi text ngắn hơn rõ rệt
    
    if len(short_text) <= 2:
        return 0.0  # Text quá ngắn không tin cậy
        
    # Kiểm tra xem các ký tự của short_text có xuất hiện theo thứ tự trong long_text không
    long_idx = 0
    matched_chars = 0
    
    for char in short_text:
        # Tìm ký tự này trong long_text từ vị trí hiện tại
        found = False
        for i in range(long_idx, len(long_text)):
            if long_text[i] == char:
                matched_chars += 1
                long_idx = i + 1  # Tiếp tục tìm từ vị trí sau
                found = True
                break
        if not found:
            # Thử tìm với ký tự tương tự OCR confusion
            for i in range(long_idx, len(long_text)):
                if are_similar_chars(char, long_text[i]):
                    matched_chars += 0.7  # Partial match
                    long_idx = i + 1
                    found = True
                    break
    
    # Tính điểm dựa trên tỷ lệ ký tự khớp
    score = matched_chars / len(short_text)
    return min(score, 1.0)  # Giới hạn tối đa 1.0

def calculate_phonetic_similarity(text1, text2):
    """
    Tính similarity dựa trên âm thanh tương tự (đơn giản)
    """
    # Mapping âm thanh tương tự
    phonetic_map = {
        'G': 'C', 'C': 'G',
        'I': 'E', 'E': 'I', 
        'L': 'N', 'N': 'L',
        'T': 'D', 'D': 'T',
        'H': 'K', 'K': 'H'
    }
    
    matches = 0
    total = max(len(text1), len(text2))
    
    for i in range(min(len(text1), len(text2))):
        c1, c2 = text1[i], text2[i]
        if c1 == c2:
            matches += 1
        elif phonetic_map.get(c1) == c2 or phonetic_map.get(c2) == c1:
            matches += 0.5
    
    return matches / total if total > 0 else 0.0

def are_similar_chars(char1, char2):
    """Kiểm tra hai ký tự có tương tự nhau không (OCR confusion)"""
    similar_groups = [
        {'G', 'C', '6', '0', 'O'},
        {'I', 'L', '1', 'l', 'T'},
        {'N', 'H'},
        {'A', '4'},
        {'S', '5'},
        {'B', '8'}
    ]
    
    for group in similar_groups:
        if char1 in group and char2 in group:
            return True
    return False

def save_roi_image_with_result(roi, roi_name, original_filename, detected_text, confidence, original_value, is_text_result=False):
    """
    Lưu ảnh ROI với kết quả OCR được overlay lên ảnh
    
    Args:
        roi: Ảnh ROI đã cắt
        roi_name: Tên của ROI
        original_filename: Tên file gốc
        detected_text: Text đã được detect
        confidence: Độ tin cậy của kết quả OCR
        original_value: Giá trị gốc trước khi xử lý
        is_text_result: True nếu kết quả là text, False nếu là số
    """
    try:
        processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_roi')
        os.makedirs(processed_folder, exist_ok=True)
        
        base_filename = os.path.splitext(original_filename)[0]
        result_type = "text" if is_text_result else "number"
        roi_result_filename = f"{base_filename}_{roi_name}_{result_type}_detected.png"
        roi_result_path = os.path.join(processed_folder, roi_result_filename)
        
        # Tạo ảnh kết quả với overlay text
        result_img = roi.copy()
        if len(result_img.shape) == 2:  # Nếu là grayscale, chuyển sang BGR
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        
        # Vẽ bounding box quanh ROI
        cv2.rectangle(result_img, (2, 2), (result_img.shape[1]-2, result_img.shape[0]-2), 
                     (0, 255, 0) if is_text_result else (255, 0, 0), 2)
        
        # Tính toán kích thước font phù hợp với ảnh
        font_scale = max(0.4, min(result_img.shape[0], result_img.shape[1]) / 120)
        
        # Vẽ kết quả detected lên ảnh (màu đỏ cho kết quả chính)
        cv2.putText(result_img, f"Detected: '{detected_text}'", 
                  (5, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
        
        # Vẽ confidence (màu trắng)
        cv2.putText(result_img, f"Confidence: {confidence:.3f}", 
                  (5, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        # Vẽ giá trị gốc nếu khác với kết quả đã xử lý (màu xanh lá)
        if original_value and original_value != detected_text:
            cv2.putText(result_img, f"Original: '{original_value}'", 
                      (5, int(result_img.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale, (0, 255, 0), 1)
        
        # Vẽ loại kết quả (màu vàng)
        # cv2.putText(result_img, f"Type: {result_type.upper()}", 
        #           (5, 75), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
        
        # Lưu ảnh kết quả
        # cv2.imwrite(roi_result_path, result_img)
        # print(f"💾 Saved {result_type} detection image to: {roi_result_path}")
        
        return roi_result_path
        
    except Exception as e:
        print(f"[ERROR] Error saving ROI image with result: {str(e)}")
        return None

# Sửa lại hàm perform_ocr_on_roi để sử dụng ảnh đã căn chỉnh
auto_detect_machine_and_screen = auto_detect_machine_and_screen_smart

if __name__ == '__main__':
    # Khởi tạo tất cả cache trước khi start server để tăng tốc độ API response
    initialize_all_caches()
    
    print("DEBUG INFO:")
    print(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
    print(f"API Routes configured:")
    print("- / (GET): Test endpoint")
    print("- /debug (GET): Debug information")
    print("- /api/images (GET): List all images")
    print("- /api/images (POST): Upload image với area, machine_code và file")
    print("- /api/images/<filename> (GET): Get image")
    print("- /api/images/<filename> (DELETE): Delete image")
    print("- /api/machines (GET): Get machine information (with optional machineid parameter)")
    print("- /api/set_machine_screen (POST): Set current machine and screen")
    print("- /api/current_machine_screen (GET): Get current machine and screen")
    print("- /api/parameter_order (GET/POST): Get or update parameter order value")
    print("- /api/images/processed_roi/<filename> (GET): Get processed ROI image")
    print("- /api/decimal_places (POST): Update decimal places configuration")
    print("- /api/decimal_places (GET): Get decimal places configuration")
    print("- /api/decimal_places/<machine_code> (GET): Get decimal places configuration for a specific machine")
    print("- /api/decimal_places/<machine_code>/<screen_name> (GET): Get decimal places configuration for a specific screen")
    print("- /api/decimal_places/<machine_code>/<screen_name> (POST): Update decimal places for a specific screen")
    print("- /api/set_decimal_value (POST): Set decimal places value based on current machine, screen and ROI index")
    print("- /api/machine_screen_status (GET): Check machine and screen status")
    print("- /api/set_all_decimal_values (POST): Set all decimal places for a specific screen")
    print("- /api/history (GET): Get OCR history")
    print("- /api/reference_images (POST): Upload ảnh tham chiếu với machine_type và screen_id")
    print("- /api/reference_images (GET): Lấy danh sách ảnh tham chiếu, có thể lọc theo machine_type và screen_id")
    print("Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=True) 
