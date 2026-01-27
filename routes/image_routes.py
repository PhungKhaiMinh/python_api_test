"""
Image Routes Module - OPTIMIZED FOR SPEED
Handles image upload, retrieval, and deletion endpoints
Uses PaddleOCR with GPU acceleration and parallel processing
"""

from flask import Blueprint, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from utils import (
    get_roi_info_cached, get_machine_type, perform_full_image_ocr,
    detect_hmi_screen
)
from utils.paddleocr_engine import (
    read_image_with_paddleocr, extract_ocr_data, find_matching_screen,
    filter_ocr_by_roi, filter_ocr_by_roi_parallel, post_process_ocr_text, 
    get_paddleocr_instance, get_ocr_performance_stats
)
import re
from utils.ocr_processor import apply_decimal_places_format

image_bp = Blueprint('image', __name__)

# Thread pool for parallel post-processing
_post_process_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="PostProcess")

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
    import sys
    
    # Force flush all print statements
    def log(msg):
        print(msg, flush=True)
        sys.stdout.flush()
    
    try:
        from utils.swagger_specs import get_upload_image_spec
        upload_image.__doc__ = get_upload_image_spec().strip()
    except:
        pass
    
    try:
        log(f"\n{'='*60}")
        log(f"[REQUEST] POST /api/images received")
        log(f"{'='*60}")
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
        log(f"[*] Step 1: Detecting and extracting HMI screen...")
        hmi_screen, hmi_time = detect_hmi_screen(uploaded_image)
        
        hmi_detected = False
        hmi_image = uploaded_image  # Default: use original image
        
        if hmi_screen is not None and hmi_screen.size > 0:
            hmi_detected = True
            hmi_image = hmi_screen
            log(f"[OK] HMI screen extracted in {hmi_time:.2f}s, size: {hmi_image.shape}")
        else:
            log(f"[WARN] Could not extract HMI screen, using original image")
        
        img_height, img_width = hmi_image.shape[:2]
        
        # STEP 2: FULL IMAGE OCR WITH PADDLEOCR
        log(f"[*] Step 2: Performing full image OCR with PaddleOCR...")
        log(f"[DEBUG] HMI image shape: {hmi_image.shape}, dtype: {hmi_image.dtype}")
        
        # Check PaddleOCR instance before calling
        from utils.paddleocr_engine import get_paddleocr_instance
        ocr_instance = get_paddleocr_instance()
        log(f"[DEBUG] PaddleOCR instance: {type(ocr_instance)}, is None: {ocr_instance is None}")
        
        ocr_start = time.time()
        results, img_width_ocr, img_height_ocr = read_image_with_paddleocr(hmi_image)
        
        log(f"[DEBUG] OCR returned: results type={type(results)}, img_size={img_width_ocr}x{img_height_ocr}")
        log(f"[DEBUG] HMI image size: {img_width}x{img_height}")
        
        # Validate OCR results
        if results is None:
            log(f"[ERROR] PaddleOCR returned None results")
            ocr_data = []
        else:
            log(f"[DEBUG] PaddleOCR returned results: {type(results)}")
            if isinstance(results, list):
                log(f"[DEBUG] Results list length: {len(results)}")
            log(f"[DEBUG] Extracting data from results...")
            ocr_data = extract_ocr_data(results)
            
            # Debug: Show sample OCR data
            if ocr_data:
                log(f"[DEBUG] Sample OCR data (first 5):")
                for i, item in enumerate(ocr_data[:5]):
                    bbox = item.get('bbox', [])
                    bbox_str = f"bbox={bbox[:2] if len(bbox) > 0 and isinstance(bbox[0], (list, tuple)) else bbox[:2] if bbox else 'None'}..." if bbox else "bbox=None"
                    log(f"[DEBUG]   {i+1}. text='{item.get('text', '')[:30]}', conf={item.get('confidence', 0):.2f}, {bbox_str}")
            else:
                log(f"[WARNING] extract_ocr_data returned empty list!")
        
        ocr_time = time.time() - ocr_start
        log(f"[OK] PaddleOCR found {len(ocr_data)} text items in {ocr_time:.2f}s")
        
        # If no OCR data found, log warning but continue processing
        if len(ocr_data) == 0:
            log(f"[WARNING] No OCR text detected in image. This may indicate:")
            log(f"   - Image quality is too low")
            log(f"   - Image does not contain readable text")
            log(f"   - PaddleOCR model needs warm-up (first call may fail)")
        
        # STEP 3: MATCH SCREEN BASED ON SPECIAL_ROIS
        log(f"[*] Step 3: Matching screen for {area}/{machine_code}...")
        roi_info = get_roi_info_cached()
        
        if not roi_info:
            log(f"[ERROR] ROI info is empty or None")
            roi_info = {}
        else:
            # Debug: Show available machines in roi_info
            if 'machines' in roi_info:
                log(f"[DEBUG] Available machine_types in roi_info: {list(roi_info['machines'].keys())}")
        
        # Get machine_type from machine_code for better matching
        from utils.config_manager import get_machine_type as _get_machine_type
        detected_machine_type = _get_machine_type(machine_code)
        log(f"[DEBUG] Machine type from machine_code '{machine_code}': {detected_machine_type}")
        
        match_start = time.time()
        matched_machine_type, matched_machine, screen_name, sub_page, sub_page_data, match_count, match_percentage = find_matching_screen(
            ocr_data, roi_info,
            selected_area=area,
            selected_machine=machine_code,
            debug=True
        )
        match_time = time.time() - match_start
        
        if screen_name:
            print(f"[OK] Screen matched: {matched_machine_type}/{matched_machine}/{screen_name} (sub-page {sub_page})")
            print(f"[OK] Match: {match_count} Special_rois ({match_percentage:.1f}%)")
        else:
            print(f"[WARNING] No screen matched. OCR data count: {len(ocr_data)}")
            if len(ocr_data) == 0:
                print(f"[WARNING] No OCR text detected - cannot match screen without text")
            elif detected_machine_type:
                print(f"[DEBUG] Expected machine_type: {detected_machine_type}")
                if detected_machine_type in roi_info.get('machines', {}):
                    mt_data = roi_info['machines'][detected_machine_type]
                    if machine_code in mt_data:
                        screens = mt_data[machine_code].get('screens', {})
                        print(f"[DEBUG] Available screens for {machine_code}: {list(screens.keys())}")
        
        # STEP 4: FILTER OCR RESULTS BY IoU WITH ROIs - OPTIMIZED
        # According to API_IMAGES_LOGIC.md - Step 4: Filter OCR Results bằng IoU với ROIs
        ocr_results = []
        
        if screen_name and sub_page_data:
            print(f"[OK] Matched: {matched_machine_type}/{matched_machine}/{screen_name} (sub-page {sub_page})")
            print(f"[OK] Match: {match_count} Special_rois ({match_percentage:.1f}%)")
            
            log(f"[*] Step 4: Filtering OCR results by IoU with ROIs...")
            log(f"[DEBUG] Before filtering: {len(ocr_data)} OCR items, image size: {img_width}x{img_height}")
            log(f"[DEBUG] HMI image actual size: {hmi_image.shape[1]}x{hmi_image.shape[0]}")
            log(f"[DEBUG] sub_page_data keys: {list(sub_page_data.keys()) if sub_page_data else 'None'}")
            
            # Count items with bbox
            items_with_bbox = sum(1 for item in ocr_data if item.get('bbox'))
            log(f"[DEBUG] OCR items with bbox: {items_with_bbox}/{len(ocr_data)}")
            
            if ocr_data and len(ocr_data) > 0:
                first_item = ocr_data[0]
                bbox = first_item.get('bbox', [])
                log(f"[DEBUG] First OCR item: text='{first_item.get('text', '')[:30]}', has_bbox={bool(bbox)}, bbox_type={type(bbox)}, bbox_len={len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'}")
                if bbox and len(bbox) > 0:
                    log(f"[DEBUG] First bbox sample: {bbox[0] if isinstance(bbox[0], (list, tuple)) else bbox[:2]}")
            
            filter_start = time.time()
            
            # Use parallel filtering for large datasets
            # IMPORTANT: Use img_width, img_height from HMI image (not from OCR)
            # OCR bounding boxes are relative to the HMI image size
            if len(ocr_data) > 50:
                filtered_results = filter_ocr_by_roi_parallel(ocr_data, sub_page_data, img_width, img_height)
            else:
                filtered_results = filter_ocr_by_roi(ocr_data, sub_page_data, img_width, img_height)
            
            filter_time = time.time() - filter_start
            
            log(f"[OK] Filtered to {len(filtered_results)} results matching ROIs in {filter_time:.3f}s")
            
            # Debug: If no results, log detailed info
            if len(filtered_results) == 0 and len(ocr_data) > 0:
                log(f"[WARNING] No OCR results matched ROIs!")
                log(f"[DEBUG] Total OCR items: {len(ocr_data)}")
                log(f"[DEBUG] Items with bbox: {items_with_bbox}")
                log(f"[DEBUG] ROIs in sub_page_data: {len(sub_page_data.get('Rois', sub_page_data.get('rois', [])))}")
                # Sample first few OCR items
                for i, item in enumerate(ocr_data[:3]):
                    bbox = item.get('bbox', [])
                    log(f"[DEBUG] OCR item {i+1}: text='{item.get('text', '')[:30]}', has_bbox={bool(bbox)}, bbox_len={len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'}")
            
            # Get machine_type
            machine_type = matched_machine_type if matched_machine_type else get_machine_type(matched_machine)
            
            # STEP 5: POST-PROCESSING AND FORMATTING
            # According to API_IMAGES_LOGIC.md - Step 5: Post-Processing và Formatting
            print(f"[*] Step 5: Post-processing and formatting OCR results...")
            
            # STEP 5.0: Pre-process - Group items by original_value to detect merged numbers
            from utils.paddleocr_engine import split_merged_numbers_by_decimal_places
            from utils.cache_manager import get_decimal_places_config_cached
            
            # Group filtered_results by original_value (use original OCR text, not processed)
            items_by_original = {}
            for item in filtered_results:
                orig_val = item.get('text', '')  # Use original OCR text before post-processing
                if orig_val not in items_by_original:
                    items_by_original[orig_val] = []
                items_by_original[orig_val].append(item)
            
            # Check for merged numbers and prepare split mapping
            decimal_config = get_decimal_places_config_cached()
            merged_splits = {}  # Map: original_value -> {roi_name: split_text}
            
            print(f"[DEBUG] Checking {len(items_by_original)} unique original values for merged numbers...")
            
            for orig_val, items_group in items_by_original.items():
                if len(items_group) < 2:
                    continue
                
                # Check if original_value looks like merged numbers
                text_clean = re.sub(r'[^\d.-]', '', str(orig_val))
                if not text_clean or len(text_clean) < 6:
                    continue
                
                print(f"[DEBUG] Potential merged numbers detected: '{orig_val}' for {len(items_group)} ROIs: {[item['matched_roi'] for item in items_group]}")
                
                # Get decimal_places config for each ROI in this group
                roi_decimal_configs = []
                for item in items_group:
                    roi_name = item['matched_roi']
                    decimal_places = None
                    
                    # Get decimal_places from config
                    # Try multiple structures in order:
                    # 1. F42 > IE-F4-WBI01 > Overview > "1" > roi_name (with machine_code)
                    # 2. F1 > Reject_Summary > IE-F1-CWA01 > "1" > roi_name (Reject_Summary)
                    # 3. F42 > Overview > "1" > roi_name (without machine_code, with sub_page)
                    # 4. F41 > Injection > roi_name (standard, no sub_page)
                    
                    if machine_type in decimal_config:
                        machine_type_config = decimal_config[machine_type]
                        
                        # Structure 1: With machine_code (F42 > IE-F4-WBI01 > Overview > "1" > roi_name)
                        if matched_machine and matched_machine in machine_type_config:
                            machine_code_config = machine_type_config[matched_machine]
                            if screen_name in machine_code_config:
                                screen_config = machine_code_config[screen_name]
                                # Check if it's a dict with sub_page
                                if isinstance(screen_config, dict):
                                    if sub_page and sub_page in screen_config:
                                        if isinstance(screen_config[sub_page], dict) and roi_name in screen_config[sub_page]:
                                            decimal_places = int(screen_config[sub_page][roi_name])
                                    # Also try direct roi_name (in case no sub_page structure)
                                    elif roi_name in screen_config:
                                        value = screen_config[roi_name]
                                        if isinstance(value, (int, float)):
                                            decimal_places = int(value)
                        
                        # Structure 2: Reject_Summary with machine_code and sub_page
                        if decimal_places is None and screen_name == "Reject_Summary" and sub_page and matched_machine:
                            if screen_name in machine_type_config:
                                screen_config = machine_type_config[screen_name]
                                if isinstance(screen_config, dict) and matched_machine in screen_config:
                                    if sub_page in screen_config[matched_machine]:
                                        if isinstance(screen_config[matched_machine][sub_page], dict) and roi_name in screen_config[matched_machine][sub_page]:
                                            decimal_places = int(screen_config[matched_machine][sub_page][roi_name])
                        
                        # Structure 3: Screen with sub_page but no machine_code (F42 > Overview > "1" > roi_name)
                        if decimal_places is None and screen_name in machine_type_config:
                            screen_config = machine_type_config[screen_name]
                            if isinstance(screen_config, dict):
                                if sub_page and sub_page in screen_config:
                                    if isinstance(screen_config[sub_page], dict) and roi_name in screen_config[sub_page]:
                                        decimal_places = int(screen_config[sub_page][roi_name])
                                # Structure 4: Standard structure (direct roi_name, no sub_page)
                                elif roi_name in screen_config:
                                    value = screen_config[roi_name]
                                    if isinstance(value, (int, float)):
                                        decimal_places = int(value)
                    
                    print(f"[DEBUG] ROI '{roi_name}': decimal_places={decimal_places}")
                    
                    if decimal_places is not None:
                        roi_coords = item.get('roi_coords', [0, 0, 0, 0])
                        roi_decimal_configs.append({
                            'roi_name': roi_name,
                            'decimal_places': decimal_places,
                            'roi_coords': roi_coords
                        })
                
                if len(roi_decimal_configs) >= 2:
                    # Sort by X coordinate (left to right)
                    roi_decimal_configs.sort(key=lambda x: x['roi_coords'][0])
                    print(f"[DEBUG] Attempting to split '{orig_val}' for {len(roi_decimal_configs)} ROIs with decimal_places: {[r['decimal_places'] for r in roi_decimal_configs]}")
                    
                    # Try to split merged numbers
                    split_numbers = split_merged_numbers_by_decimal_places(orig_val, roi_decimal_configs)
                    
                    if split_numbers and len(split_numbers) == len(roi_decimal_configs):
                        # Successfully split! Store mapping
                        print(f"[OK] Successfully split merged numbers '{orig_val}' -> {split_numbers} for ROIs: {[r['roi_name'] for r in roi_decimal_configs]}")
                        for i, roi_config in enumerate(roi_decimal_configs):
                            roi_name = roi_config['roi_name']
                            if orig_val not in merged_splits:
                                merged_splits[orig_val] = {}
                            merged_splits[orig_val][roi_name] = split_numbers[i]
                    else:
                        print(f"[WARN] Failed to split '{orig_val}' - split_numbers={split_numbers}, expected {len(roi_decimal_configs)}")
            
            # Now process each item - OPTIMIZED WITH PARALLEL PROCESSING
            from utils.paddleocr_engine import match_text_with_allowed_values, extract_number_from_text
            
            def process_single_ocr_item(item):
                """Process a single OCR item - can be run in parallel"""
                orig_val = item.get('text', '')
                roi_name = item['matched_roi']
                
                # Check if this ROI has a split number
                if orig_val in merged_splits and roi_name in merged_splits[orig_val]:
                    text = merged_splits[orig_val][roi_name]
                else:
                    # Normal processing
                    text = post_process_ocr_text(item['text'])
                    
                    allowed_values = item.get('allowed_values', [])
                    if allowed_values and len(allowed_values) > 0:
                        text = match_text_with_allowed_values(text, allowed_values)
                    else:
                        text = extract_number_from_text(text)
                    
                    text = apply_decimal_places_format(text, item['matched_roi'], machine_type, screen_name, matched_machine, sub_page)
                
                return {
                    "roi_index": item['matched_roi'],
                    "text": text,
                    "confidence": float(item['confidence']),
                    "has_text": True,
                    "original_value": orig_val,
                    "iou": float(item['iou'])
                }
            
            # Process items - use parallel for large datasets
            if len(filtered_results) > 10:
                # Parallel processing for many items
                from concurrent.futures import as_completed
                futures = [_post_process_executor.submit(process_single_ocr_item, item) for item in filtered_results]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        ocr_results.append(result)
                    except Exception as e:
                        print(f"[WARNING] Post-process error: {e}")
            else:
                # Sequential for small datasets (faster due to no thread overhead)
                for item in filtered_results:
                    ocr_results.append(process_single_ocr_item(item))
            
            # STEP 6: DEDUPLICATION - Loại Bỏ Trùng Lặp
            # According to API_IMAGES_LOGIC.md - Step 6: Deduplication
            # Đảm bảo mỗi roi_index chỉ có 1 kết quả duy nhất với IOU cao nhất
            print(f"[*] Step 6: Deduplicating results (keep highest IOU for each roi_index)...")
            roi_index_map = {}
            for result in ocr_results:
                roi_index = result['roi_index']
                if roi_index not in roi_index_map:
                    roi_index_map[roi_index] = result
                else:
                    # Keep the one with higher IOU
                    if result['iou'] > roi_index_map[roi_index]['iou']:
                        roi_index_map[roi_index] = result
            
            # Convert back to list
            ocr_results = list(roi_index_map.values())
            print(f"[OK] Deduplicated to {len(ocr_results)} unique ROI results")
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
        
        # Get OCR performance stats
        try:
            ocr_stats = get_ocr_performance_stats()
        except:
            ocr_stats = {}
        
        # Prepare debug info
        debug_info = {
            "ocr_items_total": len(ocr_data) if 'ocr_data' in locals() else 0,
            "ocr_items_with_bbox": 0,
            "rois_count": 0,
            "filtered_results_count": 0,
            "image_size": f"{img_width}x{img_height}",
            "screen_name": screen_name,
            "sub_page": None
        }
        
        # Try to get debug info safely
        try:
            if 'ocr_data' in locals() and ocr_data:
                debug_info["ocr_items_with_bbox"] = sum(1 for item in ocr_data if item.get('bbox'))
            if screen_name and 'sub_page_data' in locals() and sub_page_data:
                debug_info["rois_count"] = len(sub_page_data.get('Rois', sub_page_data.get('rois', [])))
            if 'filtered_results' in locals():
                debug_info["filtered_results_count"] = len(filtered_results)
            if 'sub_page' in locals():
                debug_info["sub_page"] = sub_page
        except:
            pass
        
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
            "debug": debug_info,
            "processing_time": {
                "hmi_detection": round(hmi_time, 2),
                "ocr": round(ocr_time, 2),
                "matching": round(match_time, 2),
                "filtering": round(filter_time, 2) if screen_name and sub_page_data else 0,
                "total": round(total_time, 2)
            },
            "performance": {
                "ocr_avg_time": round(ocr_stats.get('avg_time', 0), 2),
                "ocr_calls": ocr_stats.get('ocr_calls', 0)
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
