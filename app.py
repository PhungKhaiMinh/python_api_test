"""
Main Flask Application
HMI OCR API Server - Refactored for better maintainability
"""

from flask import Flask, request, jsonify
import os
import cv2
import concurrent.futures
from multiprocessing import cpu_count
import threading

# Import utils modules
from utils import initialize_all_caches
from utils.image_processor import init_cv2_detectors
from utils.ocr_processor import init_ocr_globals

# Import route blueprints
from routes import image_bp, machine_bp, decimal_bp, reference_bp
from routes.image_routes import init_image_routes
from routes.reference_routes import init_reference_routes

# [*] Import GPU Accelerator và Parallel Processor modules
try:
    from gpu_accelerator import get_gpu_accelerator, is_gpu_available, get_gpu_info
    from parallel_processor import (
        get_ocr_thread_pool, get_image_thread_pool, get_roi_processor,
        parallel_map, get_system_stats, ParallelROIProcessor
    )
    OPTIMIZATION_MODULES_AVAILABLE = True
    print("[OK] GPU Accelerator và Parallel Processor modules loaded")
except ImportError as e:
    OPTIMIZATION_MODULES_AVAILABLE = False
    print(f"[WARNING] Optimization modules not available: {e}")

# Import EasyOCR
try:
    import easyocr
    HAS_EASYOCR = True
    reader = easyocr.Reader(['en'], gpu=True)
    print("[OK] EasyOCR initialized with GPU")
except ImportError:
    print("[WARNING] EasyOCR not installed")
    HAS_EASYOCR = False
    reader = None

# Initialize OpenCV detectors
try:
    sift_detector = cv2.SIFT_create()
    print("[OK] SIFT detector initialized")
except Exception as e:
    print(f"[ERROR] SIFT detector failed: {e}")
    sift_detector = None

try:
    orb_detector = cv2.ORB_create(nfeatures=500)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    print("[OK] ORB detector and BFMatcher initialized")
except Exception as e:
    print(f"[ERROR] ORB detector failed: {e}")
    orb_detector = None
    bf_matcher = None

try:
    FLANN_INDEX_KDTREE = 1
    flann_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    flann_search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
    print("[OK] FLANN matcher initialized")
except Exception as e:
    print(f"[ERROR] FLANN matcher failed: {e}")
    flann_matcher = None

# Thread pools
if OPTIMIZATION_MODULES_AVAILABLE:
    _ocr_thread_pool = get_ocr_thread_pool()
    _image_thread_pool = get_image_thread_pool()
    _roi_processor = get_roi_processor()
    print("[OK] Enhanced thread pools initialized")
else:
    _ocr_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(4, cpu_count()))
    _image_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(4, cpu_count()))
    _roi_processor = None
    print(f"[WARNING] Standard thread pool with {max(4, cpu_count())} workers")

# GPU Accelerator
_gpu_accelerator = None
if OPTIMIZATION_MODULES_AVAILABLE and is_gpu_available():
    _gpu_accelerator = get_gpu_accelerator()
    print(f"[OK] GPU Accelerator ready: {get_gpu_info()}")

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ROI_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data')
OCR_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_results')
REFERENCE_IMAGES_FOLDER = os.path.join(ROI_DATA_FOLDER, 'reference_images')
HMI_REFINED_FOLDER = os.path.join(UPLOAD_FOLDER, 'hmi_refined')

# Create directories
for folder in [UPLOAD_FOLDER, ROI_DATA_FOLDER, OCR_RESULTS_FOLDER, REFERENCE_IMAGES_FOLDER, HMI_REFINED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# App config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROI_DATA_FOLDER'] = ROI_DATA_FOLDER
app.config['OCR_RESULTS_FOLDER'] = OCR_RESULTS_FOLDER
app.config['REFERENCE_IMAGES_FOLDER'] = REFERENCE_IMAGES_FOLDER
app.config['HMI_REFINED_FOLDER'] = HMI_REFINED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Initialize modules
init_cv2_detectors(sift_detector, flann_matcher)
init_ocr_globals(HAS_EASYOCR, reader, _gpu_accelerator, _ocr_thread_pool)
init_image_routes(UPLOAD_FOLDER, HMI_REFINED_FOLDER, app.config['ALLOWED_EXTENSIONS'])
init_reference_routes(REFERENCE_IMAGES_FOLDER, app.config['ALLOWED_EXTENSIONS'])

# Register blueprints
app.register_blueprint(image_bp)
app.register_blueprint(machine_bp)
app.register_blueprint(decimal_bp)
app.register_blueprint(reference_bp)

# Basic routes
@app.route('/')
def home():
    return jsonify({
        "status": "Server is running",
        "version": "2.0 - Refactored",
        "endpoints": [
            "/api/images",
            "/api/machines",
            "/api/decimal_places",
            "/api/reference_images"
        ]
    }), 200


@app.route('/debug')
def debug_info():
    """Debug information"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": [m for m in rule.methods if m not in ['OPTIONS', 'HEAD']],
            "route": str(rule)
        })
    
    return jsonify({
        "server_info": {
            "upload_folder": UPLOAD_FOLDER,
            "roi_data_folder": ROI_DATA_FOLDER,
            "optimization_enabled": OPTIMIZATION_MODULES_AVAILABLE,
            "gpu_available": is_gpu_available() if OPTIMIZATION_MODULES_AVAILABLE else False,
            "easyocr_available": HAS_EASYOCR
        },
        "routes": routes
    })


@app.route('/api/performance', methods=['GET'])
def get_performance_stats():
    """Get performance statistics"""
    try:
        import time
        stats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_available": False,
            "optimization_enabled": OPTIMIZATION_MODULES_AVAILABLE
        }
        
        if OPTIMIZATION_MODULES_AVAILABLE and is_gpu_available():
            stats["gpu_available"] = True
            stats["gpu_info"] = get_gpu_info()
            if _gpu_accelerator:
                stats["gpu_memory"] = _gpu_accelerator.get_memory_info()
        
        if OPTIMIZATION_MODULES_AVAILABLE:
            stats["system"] = get_system_stats()
        else:
            stats["system"] = {"cpu_count": cpu_count()}
        
        stats["ocr"] = {
            "easyocr_available": HAS_EASYOCR,
            "gpu_enabled": HAS_EASYOCR and reader is not None
        }
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get OCR history"""
    try:
        limit = int(request.args.get('limit', 10))
        
        if not os.path.exists(OCR_RESULTS_FOLDER):
            return jsonify({"history": []}), 200
        
        files = [f for f in os.listdir(OCR_RESULTS_FOLDER) if f.endswith('.json')]
        files.sort(reverse=True)  # Most recent first
        files = files[:limit]
        
        history = []
        for filename in files:
            try:
                import json
                with open(os.path.join(OCR_RESULTS_FOLDER, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history.append(data)
            except:
                continue
        
        return jsonify({"history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Initialize caches at startup
    initialize_all_caches()
    
    print("\n" + "="*70)
    print("HMI OCR API SERVER - REFACTORED v2.0")
    print("="*70)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"ROI data folder: {ROI_DATA_FOLDER}")
    print(f"GPU available: {is_gpu_available() if OPTIMIZATION_MODULES_AVAILABLE else False}")
    print(f"EasyOCR available: {HAS_EASYOCR}")
    print("="*70)
    print("Starting server on http://0.0.0.0:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

