"""
Image Routes Module
Handles image upload, retrieval, and deletion endpoints
Uses PaddleOCR exclusively for OCR processing
"""

from flask import Blueprint, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import cv2
import time
from utils import (
    get_roi_info_cached, get_machine_type, perform_full_image_ocr,
    detect_hmi_screen
)
from utils.paddleocr_engine import (
    read_image_with_paddleocr, extract_ocr_data, find_matching_screen,
    filter_ocr_by_roi, post_process_ocr_text, get_paddleocr_instance
)
from utils.ocr_processor import apply_decimal_places_format

image_bp = Blueprint('image', __name__)

# Will be set by app
UPLOAD_FOLDER = None
HMI_REFINED_FOLDER = None
ALLOWED_EXTENSIONS = None


def init_image_routes(upload_folder, hmi_folder, allowed_ext):
    """Initialize route config from app"""
    global UPLOAD_FOLDER, HMI_REFINED_FOLDER, ALLOWED_EXTENSIONS
    UPLOAD_FOLDER = upload_folder
    HMI_REFINED_FOLDER = hmi_folder
    ALLOWED_EXTENSIONS = allowed_ext


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@image_bp.route('/api/images', methods=['POST'])
def upload_image():
    """
    Upload image and perform OCR using PaddleOCR
    
    Uses the new PaddleOCR algorithm:
    1. Detect and extract HMI screen
    2. Full image OCR with PaddleOCR
    3. Match screen based on Special_rois
    4. Filter OCR results by IoU with ROIs
    """
    try:
        from utils.swagger_specs import get_upload_image_spec
        upload_image.__doc__ = get_upload_image_spec().strip()
    except:
        pass
    
    try:
        # Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get params
        area = request.form.get('area')
        machine_code = request.form.get('machine_code')
        
        if not area or not machine_code:
            return jsonify({"error": "Missing area or machine_code"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename_with_timestamp = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename_with_timestamp)
        file.save(filepath)
        
        # Read image
        uploaded_image = cv2.imread(filepath)
        if uploaded_image is None:
            return jsonify({"error": "Failed to read image"}), 400
        
        start_time = time.time()
        
        # STEP 1: DETECT AND EXTRACT HMI SCREEN
        print(f"[*] Step 1: Detecting and extracting HMI screen...")
        hmi_screen, hmi_time = detect_hmi_screen(uploaded_image)
        
        hmi_detected = False
        hmi_image = uploaded_image  # Default: use original image
        
        if hmi_screen is not None and hmi_screen.size > 0:
            hmi_detected = True
            hmi_image = hmi_screen
            print(f"[OK] HMI screen extracted in {hmi_time:.2f}s, size: {hmi_image.shape}")
        else:
            print(f"[WARN] Could not extract HMI screen, using original image")
        
        img_height, img_width = hmi_image.shape[:2]
        
        # STEP 2: FULL IMAGE OCR WITH PADDLEOCR
        print(f"[*] Step 2: Performing full image OCR with PaddleOCR...")
        ocr_start = time.time()
        results, _, _ = read_image_with_paddleocr(hmi_image)
        ocr_data = extract_ocr_data(results)
        ocr_time = time.time() - ocr_start
        print(f"[OK] PaddleOCR found {len(ocr_data)} text items in {ocr_time:.2f}s")
        
        # STEP 3: MATCH SCREEN BASED ON SPECIAL_ROIS
        print(f"[*] Step 3: Matching screen for {area}/{machine_code}...")
        roi_info = get_roi_info_cached()
        
        match_start = time.time()
        matched_machine_type, matched_machine, screen_name, sub_page, sub_page_data, match_count, match_percentage = find_matching_screen(
            ocr_data, roi_info,
            selected_area=area,
            selected_machine=machine_code,
            debug=True
        )
        match_time = time.time() - match_start
        
        # STEP 4: FILTER OCR RESULTS BY IOI WITH ROIS
        ocr_results = []
        
        if screen_name and sub_page_data:
            print(f"[OK] Matched: {matched_machine_type}/{matched_machine}/{screen_name} (sub-page {sub_page})")
            print(f"[OK] Match: {match_count} Special_rois ({match_percentage:.1f}%)")
            
            print(f"[*] Step 4: Filtering OCR results by IoU with ROIs...")
            filter_start = time.time()
            filtered_results = filter_ocr_by_roi(ocr_data, sub_page_data, img_width, img_height)
            filter_time = time.time() - filter_start
            
            print(f"[OK] Filtered to {len(filtered_results)} results matching ROIs")
            
            # Get machine_type
            machine_type = matched_machine_type if matched_machine_type else get_machine_type(matched_machine)
            
            # Convert to standard output format
            for item in filtered_results:
                text = post_process_ocr_text(item['text'])
                text = apply_decimal_places_format(text, item['matched_roi'], machine_type, screen_name, matched_machine, sub_page)
                
                ocr_results.append({
                    "roi_index": item['matched_roi'],
                    "text": text,
                    "confidence": float(item['confidence']),
                    "has_text": True,
                    "original_value": item['text'],
                    "iou": float(item['iou'])
                })
        else:
            print(f"[WARN] No matching screen found - returning all OCR results")
            
            # Return all OCR results without filtering
            for i, item in enumerate(ocr_data):
                text = post_process_ocr_text(item['text'])
                ocr_results.append({
                    "roi_index": f"OCR_{i}",
                    "text": text,
                    "confidence": float(item['confidence']),
                    "has_text": True,
                    "original_value": item['text']
                })
            
            machine_type = get_machine_type(machine_code) if machine_code else area
        
        total_time = time.time() - start_time
        
        # Build response
        response_data = {
            "success": True,
            "filename": filename_with_timestamp,
            "machine_code": matched_machine or machine_code,
            "machine_type": machine_type,
            "screen_id": screen_name,
            "area": matched_machine_type or area,
            "hmi_detection": {
                "hmi_extracted": hmi_detected,
                "hmi_size": f"{img_width}x{img_height}",
                "extraction_time": round(hmi_time, 2)
            },
            "screen_matching": {
                "matched": screen_name is not None,
                "match_count": match_count,
                "match_percentage": round(match_percentage, 1)
            },
            "ocr_results": ocr_results,
            "roi_count": len(ocr_results),
            "ocr_engine": "PaddleOCR",
            "processing_time": {
                "hmi_detection": round(hmi_time, 2),
                "ocr": round(ocr_time, 2),
                "matching": round(match_time, 2),
                "total": round(total_time, 2)
            }
        }
        
        # Add sub-page info if available
        if sub_page:
            response_data["sub_page"] = sub_page
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"[ERROR] upload_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@image_bp.route('/api/images', methods=['GET'])
def get_images():
    """Get list of uploaded images"""
    try:
        from utils.swagger_specs import get_images_list_spec
        get_images.__doc__ = get_images_list_spec().strip()
    except:
        pass
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            return jsonify({"images": []}), 200
        
        files = [f for f in os.listdir(UPLOAD_FOLDER) 
                if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)) and allowed_file(f)]
        
        return jsonify({"images": files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@image_bp.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    """Get specific image file"""
    try:
        from utils.swagger_specs import get_image_spec
        get_image.__doc__ = get_image_spec().strip()
    except:
        pass
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except:
        abort(404)


@image_bp.route('/api/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    """Delete image file"""
    try:
        from utils.swagger_specs import get_delete_image_spec
        delete_image.__doc__ = get_delete_image_spec().strip()
    except:
        pass
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"message": f"Deleted {filename}"}), 200
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@image_bp.route('/api/images/hmi_detection/<filename>', methods=['GET'])
def get_hmi_detection_image(filename):
    """Get HMI detection image"""
    try:
        from utils.swagger_specs import get_hmi_detection_spec
        get_hmi_detection_image.__doc__ = get_hmi_detection_spec().strip()
    except:
        pass
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except:
        abort(404)


@image_bp.route('/api/images/ocr', methods=['POST'])
def perform_ocr_on_image():
    """
    Perform full OCR on image with automatic screen detection and ROI matching
    Uses PaddleOCR algorithm exclusively
    
    Request body:
        - file: Image file (required)
        - area: Area code (optional, e.g., "F1", "F4")
        - machine_code: Machine code (optional, e.g., "IE-F1-CWA01")
    
    Returns:
        JSON with OCR results, screen detection info, and matched ROIs
    """
    try:
        # Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get optional params
        area = request.form.get('area')
        machine_code = request.form.get('machine_code')
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename_with_timestamp = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename_with_timestamp)
        file.save(filepath)
        
        # Read image
        uploaded_image = cv2.imread(filepath)
        if uploaded_image is None:
            return jsonify({"error": "Failed to read image"}), 400
        
        # Step 1: HMI Detection
        print(f"[*] Detecting HMI screen...")
        hmi_screen, hmi_time = detect_hmi_screen(uploaded_image)
        
        if hmi_screen is not None and hmi_screen.size > 0:
            image_for_ocr = hmi_screen
            hmi_detected = True
            print(f"[OK] HMI extracted in {hmi_time:.2f}s")
        else:
            image_for_ocr = uploaded_image
            hmi_detected = False
            print(f"[WARN] No HMI detected, using original image")
        
        # Step 2: Full Image OCR with screen matching
        roi_info = get_roi_info_cached()
        
        ocr_result = perform_full_image_ocr(
            image_for_ocr, 
            roi_info, 
            area=area, 
            machine_code=machine_code
        )
        
        # Add additional info to response
        ocr_result["filename"] = filename_with_timestamp
        ocr_result["hmi_detection"] = {
            "hmi_extracted": hmi_detected,
            "hmi_size": f"{image_for_ocr.shape[1]}x{image_for_ocr.shape[0]}",
            "extraction_time": round(hmi_time, 2)
        }
        ocr_result["ocr_engine"] = "PaddleOCR"
        
        return jsonify(ocr_result), 200
        
    except Exception as e:
        print(f"[ERROR] perform_ocr_on_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500
