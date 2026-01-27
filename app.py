"""
Main Flask Application
HMI OCR API Server - PaddleOCR Edition
Uses PaddleOCR exclusively for OCR processing
"""

from flask import Flask, request, jsonify
import os
import concurrent.futures
from multiprocessing import cpu_count
from datetime import datetime
import json

# Import utils modules
from utils import initialize_all_caches
from utils.ocr_processor import init_ocr_globals
from utils.paddleocr_engine import get_paddleocr_instance, init_paddleocr_globals

# Import route blueprints
from routes import image_bp, machine_bp, decimal_bp, reference_bp
from routes.image_routes import init_image_routes
from routes.reference_routes import init_reference_routes

# [*] Import GPU Accelerator và Parallel Processor modules
try:
    from gpu_accelerator import get_gpu_accelerator, is_gpu_available, get_gpu_info
    from parallel_processor import (
        get_ocr_thread_pool, get_image_thread_pool,
        get_system_stats
    )
    OPTIMIZATION_MODULES_AVAILABLE = True
    print("[OK] GPU Accelerator and Parallel Processor modules loaded")
except ImportError as e:
    OPTIMIZATION_MODULES_AVAILABLE = False
    print(f"[WARNING] Optimization modules not available: {e}")

# Initialize PaddleOCR
print("[*] Initializing PaddleOCR...")
try:
    paddle_reader = get_paddleocr_instance()
    HAS_PADDLEOCR = paddle_reader is not None
    if HAS_PADDLEOCR:
        print("[OK] PaddleOCR initialized successfully")
    else:
        print("[WARNING] PaddleOCR initialization failed")
except Exception as e:
    print(f"[WARNING] PaddleOCR not available: {e}")
    HAS_PADDLEOCR = False
    paddle_reader = None

# Thread pools
if OPTIMIZATION_MODULES_AVAILABLE:
    _ocr_thread_pool = get_ocr_thread_pool()
    _image_thread_pool = get_image_thread_pool()
    print("[OK] Enhanced thread pools initialized")
else:
    _ocr_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(4, cpu_count()))
    _image_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(4, cpu_count()))
    print(f"[OK] Standard thread pool with {max(4, cpu_count())} workers")

# GPU Accelerator
_gpu_accelerator = None
if OPTIMIZATION_MODULES_AVAILABLE and is_gpu_available():
    _gpu_accelerator = get_gpu_accelerator()
    print(f"[OK] GPU Accelerator ready")

# Initialize Flask app
app = Flask(__name__)

# Initialize Swagger UI
try:
    from utils.swagger_config import init_swagger
    from utils.swagger_specs import get_history_spec
    swagger = init_swagger(app)
    SWAGGER_AVAILABLE = True
except ImportError:
    SWAGGER_AVAILABLE = False

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ROI_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data')
OCR_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_results')
HMI_REFINED_FOLDER = os.path.join(UPLOAD_FOLDER, 'hmi_refined')

# Create directories
for folder in [UPLOAD_FOLDER, ROI_DATA_FOLDER, OCR_RESULTS_FOLDER, HMI_REFINED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# App config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROI_DATA_FOLDER'] = ROI_DATA_FOLDER
app.config['OCR_RESULTS_FOLDER'] = OCR_RESULTS_FOLDER
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
init_paddleocr_globals(_gpu_accelerator)
init_ocr_globals(HAS_PADDLEOCR, paddle_reader, _gpu_accelerator, _ocr_thread_pool)
init_image_routes(UPLOAD_FOLDER, HMI_REFINED_FOLDER, app.config['ALLOWED_EXTENSIONS'])
init_reference_routes(ROI_DATA_FOLDER, app.config['ALLOWED_EXTENSIONS'])

# Inject Swagger docstrings before registering blueprints
if SWAGGER_AVAILABLE:
    try:
        from utils.swagger_specs import (
            # Machine routes
            get_machines_spec, get_machines_by_area_spec, get_machine_screens_spec,
            get_set_machine_screen_spec, get_current_machine_screen_spec,
            get_machine_screen_status_spec, get_update_machine_screen_spec,
            # Reference routes
            get_reference_images_post_spec, get_reference_images_list_spec,
            get_reference_image_spec, get_delete_reference_image_spec,
            # Image routes
            get_upload_image_spec, get_images_list_spec, get_image_spec,
            get_delete_image_spec, get_hmi_detection_spec,
            # Decimal routes (UNIFIED API)
            get_decimal_places_all_spec, get_decimal_places_post_spec,
            get_decimal_places_unified_spec, get_decimal_places_unified_post_spec,
            get_set_decimal_value_spec, get_set_all_decimal_values_spec
        )
        
        # Inject docstrings directly
        from routes import machine_routes, reference_routes, image_routes, decimal_routes
        
        # Machine routes
        machine_routes.get_machine_info_route.__doc__ = get_machines_spec().strip()
        machine_routes.get_machines_by_area.__doc__ = get_machines_by_area_spec().strip()
        machine_routes.get_machine_screens.__doc__ = get_machine_screens_spec().strip()
        machine_routes.set_machine_screen.__doc__ = get_set_machine_screen_spec().strip()
        machine_routes.get_current_machine_screen.__doc__ = get_current_machine_screen_spec().strip()
        machine_routes.check_machine_screen_status.__doc__ = get_machine_screen_status_spec().strip()
        machine_routes.update_machine_screen.__doc__ = get_update_machine_screen_spec().strip()
        
        # Reference routes
        reference_routes.upload_reference_image.__doc__ = get_reference_images_post_spec().strip()
        reference_routes.get_reference_images.__doc__ = get_reference_images_list_spec().strip()
        reference_routes.get_reference_image.__doc__ = get_reference_image_spec().strip()
        reference_routes.delete_reference_image.__doc__ = get_delete_reference_image_spec().strip()
        
        # Image routes
        image_routes.upload_image.__doc__ = get_upload_image_spec().strip()
        image_routes.get_images.__doc__ = get_images_list_spec().strip()
        image_routes.get_image.__doc__ = get_image_spec().strip()
        image_routes.delete_image.__doc__ = get_delete_image_spec().strip()
        image_routes.get_hmi_detection_image.__doc__ = get_hmi_detection_spec().strip()
        
        # Decimal routes (UNIFIED API)
        decimal_routes.get_decimal_places.__doc__ = get_decimal_places_all_spec().strip()
        decimal_routes.update_decimal_places.__doc__ = get_decimal_places_post_spec().strip()
        decimal_routes.get_decimal_places_unified.__doc__ = get_decimal_places_unified_spec().strip()
        decimal_routes.update_decimal_places_unified.__doc__ = get_decimal_places_unified_post_spec().strip()
        decimal_routes.set_decimal_value.__doc__ = get_set_decimal_value_spec().strip()
        decimal_routes.set_all_decimal_values.__doc__ = get_set_all_decimal_values_spec().strip()
        
        print("[OK] Swagger docstrings injected")
    except Exception as e:
        import traceback
        print(f"[WARNING] Failed to inject Swagger docs: {e}")

# Register blueprints
app.register_blueprint(image_bp)
app.register_blueprint(machine_bp)
app.register_blueprint(decimal_bp)
app.register_blueprint(reference_bp)

# Basic routes
@app.route('/')
def home():
    """Get server status and available endpoints"""
    return jsonify({
        "status": "Server is running",
        "version": "3.0 - PaddleOCR Edition",
        "ocr_engine": "PaddleOCR",
        "endpoints": [
            "/api/images",
            "/api/images/ocr",
            "/api/machines",
            "/api/decimal_places"
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
            "paddleocr_available": HAS_PADDLEOCR,
            "ocr_engine": "PaddleOCR"
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
            "optimization_enabled": OPTIMIZATION_MODULES_AVAILABLE,
            "ocr_engine": "PaddleOCR"
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
            "paddleocr_available": HAS_PADDLEOCR,
            "engine": "PaddleOCR"
        }
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get OCR history with filtering support"""
    try:
        # Required parameters
        start_time_str = request.args.get('start_time', '').strip()
        end_time_str = request.args.get('end_time', '').strip()
        
        if not start_time_str or not end_time_str:
            return jsonify({
                "error": "start_time and end_time are required parameters",
                "example": "/api/history?start_time=2025-10-03&end_time=2025-10-05"
            }), 400
        
        # Optional parameters
        machine_code_filter = request.args.get('machine_code', '').strip().upper()
        area_filter = request.args.get('area', '').strip().upper()
        screen_id_filter = request.args.get('screen_id', '').strip()
        limit = int(request.args.get('limit', 100))
        
        # Parse time filters
        start_time = None
        end_time = None
        
        try:
            try:
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({
                "error": "Invalid start_time format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
                "received": start_time_str
            }), 400
        
        try:
            try:
                end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                end_time = datetime.strptime(end_time_str, "%Y-%m-%d")
                end_time = end_time.replace(hour=23, minute=59, second=59)
        except ValueError:
            return jsonify({
                "error": "Invalid end_time format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
                "received": end_time_str
            }), 400
        
        if start_time > end_time:
            return jsonify({
                "error": "start_time cannot be greater than end_time",
                "start_time": start_time_str,
                "end_time": end_time_str
            }), 400
        
        if not os.path.exists(OCR_RESULTS_FOLDER):
            return jsonify({
                "history": [],
                "count": 0,
                "filters_applied": {
                    "start_time": start_time_str,
                    "end_time": end_time_str
                }
            }), 200
        
        # Get all JSON files
        files = [f for f in os.listdir(OCR_RESULTS_FOLDER) if f.endswith('.json')]
        
        history = []
        for filename in files:
            try:
                file_path = os.path.join(OCR_RESULTS_FOLDER, filename)
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                file_timestamp_str = data.get('timestamp', '')
                file_timestamp = None
                
                if file_timestamp_str:
                    try:
                        try:
                            file_timestamp = datetime.strptime(file_timestamp_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            file_timestamp = datetime.strptime(file_timestamp_str, "%Y-%m-%d")
                    except (ValueError, TypeError):
                        pass
                
                if file_timestamp is None:
                    try:
                        file_timestamp = datetime.fromtimestamp(os.path.getctime(file_path))
                    except:
                        continue
                
                if file_timestamp < start_time or file_timestamp > end_time:
                    continue
                
                if machine_code_filter:
                    file_machine_code = data.get('machine_code', '').upper()
                    if machine_code_filter not in file_machine_code and file_machine_code != machine_code_filter:
                        continue
                
                if area_filter:
                    file_area = data.get('area', '').upper()
                    if file_area != area_filter:
                        continue
                
                if screen_id_filter:
                    file_screen_id = data.get('screen_id', '')
                    if file_screen_id != screen_id_filter:
                        continue
                
                data['filename'] = filename
                history.append(data)
                
            except (json.JSONDecodeError, Exception):
                continue
        
        # Sort by timestamp (most recent first)
        def get_sort_key(item):
            timestamp_str = item.get('timestamp', '')
            if timestamp_str:
                try:
                    try:
                        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        return datetime.strptime(timestamp_str, "%Y-%m-%d")
                except:
                    pass
            return datetime.min
        
        history.sort(key=get_sort_key, reverse=True)
        history = history[:limit]
        
        filters_applied = {
            "start_time": start_time_str,
            "end_time": end_time_str
        }
        
        if machine_code_filter:
            filters_applied['machine_code'] = machine_code_filter
        if area_filter:
            filters_applied['area'] = area_filter
        if screen_id_filter:
            filters_applied['screen_id'] = screen_id_filter
        
        return jsonify({
            "history": history,
            "count": len(history),
            "limit": limit,
            "filters_applied": filters_applied
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Inject Swagger docstrings for System and History routes
if SWAGGER_AVAILABLE:
    try:
        from utils.swagger_specs import (
            get_home_spec, get_debug_spec, get_performance_spec, get_history_spec
        )
        
        home.__doc__ = get_home_spec().strip()
        debug_info.__doc__ = get_debug_spec().strip()
        get_performance_stats.__doc__ = get_performance_spec().strip()
        get_history.__doc__ = get_history_spec().strip()
        
        print("[OK] Swagger docstrings injected for System routes")
    except Exception as e:
        print(f"[WARNING] Failed to inject System Swagger docs: {e}")


if __name__ == '__main__':
    # Initialize caches at startup
    initialize_all_caches()
    
    print("\n" + "="*70)
    print("HMI OCR API SERVER - v3.0 PaddleOCR Edition")
    print("="*70)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"ROI data folder: {ROI_DATA_FOLDER}")
    print(f"GPU available: {is_gpu_available() if OPTIMIZATION_MODULES_AVAILABLE else False}")
    print(f"PaddleOCR available: {HAS_PADDLEOCR}")
    print(f"OCR Engine: PaddleOCR (exclusively)")
    print("="*70)
    print("Starting server on http://0.0.0.0:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
