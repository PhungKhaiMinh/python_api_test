"""
OCR Processor Module
Handles all OCR processing operations using PaddleOCR,
parallel processing, and result formatting

Refactored to use PaddleOCR instead of EasyOCR for better accuracy
"""

import cv2
import numpy as np
import re
import os
import time
from .cache_manager import get_decimal_places_config_cached, get_roi_info_cached, get_machine_info_cached
from .config_manager import get_machine_type

# Import PaddleOCR engine
from .paddleocr_engine import (
    get_paddleocr_instance, 
    is_paddleocr_available,
    read_image_with_paddleocr,
    extract_ocr_data,
    post_process_ocr_text,
    find_matching_screen,
    filter_ocr_by_roi,
    normalize_text,
    fuzzy_match,
    calculate_iou,
    polygon_to_normalized_bbox,
    IOU_THRESHOLD
)


# Global variables - will be set by app
HAS_PADDLEOCR = False
_gpu_accelerator = None
_ocr_thread_pool = None


def init_ocr_globals(paddleocr_available=True, ocr_reader=None, gpu_acc=None, thread_pool=None):
    """Initialize global OCR variables from app"""
    global HAS_PADDLEOCR, _gpu_accelerator, _ocr_thread_pool
    
    # Try to initialize PaddleOCR
    try:
        paddle_instance = get_paddleocr_instance()
        HAS_PADDLEOCR = paddle_instance is not None
    except Exception as e:
        print(f"[WARNING] PaddleOCR initialization failed: {e}")
        HAS_PADDLEOCR = False
    
    _gpu_accelerator = gpu_acc
    _ocr_thread_pool = thread_pool
    
    if HAS_PADDLEOCR:
        print("[OK] OCR Processor initialized with PaddleOCR")
    else:
        print("[WARNING] PaddleOCR not available")


def process_single_roi_paddleocr(args):
    """
    Process OCR for a single ROI using PaddleOCR
    
    Args:
        args: tuple containing (roi_image, roi_name, machine_type, allowed_values, 
              is_special_on_off_case, screen_id, machine_code, sub_page)
    
    Returns:
        dict: OCR result with text, confidence, and metadata
    """
    # Unpack args
    if len(args) == 8:
        (roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id, machine_code, sub_page) = args
    else:
        (roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id) = args
        machine_code = None
        sub_page = None
    
    try:
        # Special ON/OFF case - color analysis
        if is_special_on_off_case and machine_type != "F1":
            b, g, r = cv2.split(roi_image)
            
            if _gpu_accelerator:
                avg_blue = _gpu_accelerator.mean(b)
                avg_red = _gpu_accelerator.mean(r)
            else:
                avg_blue = np.mean(b)
                avg_red = np.mean(r)
            
            best_text = "OFF" if avg_blue > avg_red else "ON"
            
            return {
                "roi_index": roi_name,
                "text": best_text,
                "confidence": 1.0,
                "has_text": True
            }
        
        if not HAS_PADDLEOCR:
            return {
                "roi_index": roi_name,
                "text": "OCR_NOT_AVAILABLE",
                "confidence": 0,
                "has_text": False,
                "original_value": ""
            }
        
        # Resize if ROI is too small
        height, width = roi_image.shape[:2]
        if height < 30 or width < 30:
            scale_factor = max(30/height, 30/width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            if _gpu_accelerator:
                roi_image = _gpu_accelerator.resize_gpu(roi_image, (new_width, new_height), cv2.INTER_CUBIC)
            else:
                roi_image = cv2.resize(roi_image, (new_width, new_height), cv2.INTER_CUBIC)
        
        # Preprocess image for better OCR
        if len(roi_image.shape) == 3:
            if _gpu_accelerator:
                roi_processed = _gpu_accelerator.cvt_color_gpu(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                roi_processed = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_processed = roi_image.copy()
        
        # Gaussian Blur
        if _gpu_accelerator:
            roi_processed = _gpu_accelerator.gaussian_blur_gpu(roi_processed, (3, 3), 0)
        else:
            roi_processed = cv2.GaussianBlur(roi_processed, (3, 3), 0)
        
        # Convert back to BGR for PaddleOCR
        roi_for_ocr = cv2.cvtColor(roi_processed, cv2.COLOR_GRAY2BGR) if len(roi_processed.shape) == 2 else roi_processed
        
        # Perform OCR with PaddleOCR
        results, img_width, img_height = read_image_with_paddleocr(roi_for_ocr)
        ocr_data = extract_ocr_data(results)
        
        if ocr_data and len(ocr_data) > 0:
            # Find best result by confidence
            best_result = max(ocr_data, key=lambda x: x['confidence'])
            best_text = best_result['text']
            best_confidence = best_result['confidence']
            original_value = best_text
            has_text = True
            
            # Post-processing
            best_text = post_process_ocr_text(best_text, allowed_values)
            
            # Apply decimal places formatting
            best_text = apply_decimal_places_format(best_text, roi_name, machine_type, screen_id, machine_code, sub_page)
            
            # Handle working hours format
            if "working hours" in roi_name.lower():
                best_text = format_working_hours(best_text)
            
            return {
                "roi_index": roi_name,
                "text": best_text,
                "confidence": float(best_confidence),
                "has_text": has_text,
                "original_value": original_value
            }
        else:
            return {
                "roi_index": roi_name,
                "text": "",
                "confidence": 0.0,
                "has_text": False,
                "original_value": ""
            }
            
    except Exception as e:
        print(f"[ERROR] process_single_roi_paddleocr: {str(e)}")
        return {
            "roi_index": roi_name,
            "text": "ERROR",
            "confidence": 0.0,
            "has_text": False,
            "original_value": "",
            "error": str(e)
        }


def apply_decimal_places_format(text, roi_name, machine_type, screen_id, machine_code=None, sub_page=None):
    """Apply decimal places formatting based on configuration"""
    try:
        # Get decimal config
        decimal_config = get_decimal_places_config_cached()
        
        # Clean text - remove non-numeric except . and -
        text_clean = re.sub(r'[^\d.-]', '', str(text))
        is_negative = text_clean.startswith('-')
        text_clean = text_clean.lstrip('-')
        
        # Check if numeric
        if not text_clean.replace('.', '').isdigit():
            return text
        
        # Get decimal places config for this ROI
        decimal_places = None
        
        # Try machine_info if machine_code not provided
        if machine_code is None:
            machine_info = get_machine_info_cached()
            machine_code = machine_info.get('machine_code')
        
        # New structure for Reject_Summary with sub-page:
        # machine_type > screen_id > machine_code > sub_page > roi_name
        if screen_id == "Reject_Summary" and sub_page and machine_code:
            if machine_type in decimal_config:
                if screen_id in decimal_config[machine_type]:
                    if machine_code in decimal_config[machine_type][screen_id]:
                        if sub_page in decimal_config[machine_type][screen_id][machine_code]:
                            if roi_name in decimal_config[machine_type][screen_id][machine_code][sub_page]:
                                decimal_places = decimal_config[machine_type][screen_id][machine_code][sub_page][roi_name]
        else:
            # Standard structure: machine_type > screen_id > roi_name
            if machine_type in decimal_config:
                if screen_id in decimal_config[machine_type]:
                    if roi_name in decimal_config[machine_type][screen_id]:
                        decimal_places = decimal_config[machine_type][screen_id][roi_name]
        
        # Apply formatting
        if decimal_places is not None:
            if decimal_places == 0:
                formatted = text_clean.replace('.', '')
            else:
                if '.' in text_clean:
                    int_part, dec_part = text_clean.split('.')
                    all_digits = int_part + dec_part
                else:
                    all_digits = text_clean
                
                if len(all_digits) <= decimal_places:
                    padded = all_digits.zfill(decimal_places)
                    formatted = f"0.{padded}"
                else:
                    insert_pos = len(all_digits) - decimal_places
                    formatted = f"{all_digits[:insert_pos]}.{all_digits[insert_pos:]}"
            
            return ('-' + formatted) if is_negative else formatted
        
        return text
        
    except Exception as e:
        print(f"[ERROR] apply_decimal_format: {e}")
        return text


def format_working_hours(text):
    """Format working hours as HH:MM:SS"""
    try:
        digits_only = re.sub(r'[^0-9]', '', text)
        
        if len(digits_only) >= 2:
            result = ""
            for i in range(len(digits_only) - 1, -1, -1):
                result = digits_only[i] + result
                if i > 0 and (len(digits_only) - i) % 2 == 0:
                    result = ":" + result
            return result
        
        return text
    except:
        return text


def process_roi_with_retry_logic_optimized(roi_args, original_filename):
    """Process ROI with retry logic - optimized version using PaddleOCR"""
    return process_single_roi_paddleocr(roi_args)


def perform_ocr_on_roi_optimized(image, roi_coordinates, original_filename, 
                                 template_path=None, roi_names=None, machine_code=None, screen_id=None, sub_page=None):
    """
    Optimized OCR on ROIs using PaddleOCR with parallel processing
    
    This is the main entry point for performing OCR on detected ROIs.
    Uses PaddleOCR engine for text recognition.
    """
    try:
        if roi_coordinates is None or len(roi_coordinates) == 0:
            return []
        
        if roi_names is None or len(roi_names) != len(roi_coordinates):
            roi_names = [f"ROI_{i}" for i in range(len(roi_coordinates))]
        
        if not HAS_PADDLEOCR:
            print("[WARNING] PaddleOCR not available, returning empty results")
            return [{"roi_index": roi_names[i], "text": "OCR_NOT_AVAILABLE", "confidence": 0} 
                   for i in range(len(roi_coordinates))]
        
        img_height, img_width = image.shape[:2]
        
        # Get machine info
        machine_info = get_machine_info_cached()
        machine_code = machine_code or machine_info.get('machine_code', 'F41')
        screen_id = screen_id or machine_info.get('screen_id', 'Main')
        sub_page = sub_page or machine_info.get('sub_page')
        
        roi_info = get_roi_info_cached()
        machine_type = get_machine_type(machine_code)
        
        # Prepare ROI args for processing
        roi_args = []
        
        for i, coords in enumerate(roi_coordinates):
            try:
                if len(coords) != 4:
                    continue
                
                # Convert coordinates
                is_normalized = any(isinstance(v, float) and 0 <= v <= 1 for v in coords)
                
                if is_normalized:
                    x1, y1, x2, y2 = coords
                    x1, x2 = int(x1 * img_width), int(x2 * img_width)
                    y1, y2 = int(y1 * img_height), int(y2 * img_height)
                else:
                    x1, y1, x2, y2 = [int(float(c)) for c in coords]
                
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
                    continue
                
                roi_image = image[y1:y2, x1:x2]
                roi_name = roi_names[i]
                
                # Get allowed values
                allowed_values = []
                is_special_on_off_case = False
                
                # Check in ROI info
                for key in [machine_code, machine_type]:
                    if key in roi_info.get("machines", {}):
                        screens = roi_info["machines"][key].get("screens", {})
                        if screen_id in screens:
                            roi_list = screens[screen_id]
                            for roi_item in roi_list:
                                if isinstance(roi_item, dict) and roi_item.get("name") == roi_name:
                                    allowed_values = roi_item.get("allowed_values", [])
                                    if "ON" in allowed_values and "OFF" in allowed_values:
                                        is_special_on_off_case = True
                                    break
                
                roi_args.append((roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id, machine_code, sub_page))
                
            except Exception as e:
                print(f"[ERROR] Preparing ROI {i}: {e}")
                continue
        
        # Process ROIs in parallel if thread pool available
        if _ocr_thread_pool:
            results = list(_ocr_thread_pool.map(process_single_roi_paddleocr, roi_args))
        else:
            results = [process_single_roi_paddleocr(args) for args in roi_args]
        
        return results
        
    except Exception as e:
        print(f"[ERROR] perform_ocr_on_roi_optimized: {e}")
        return []


def perform_full_image_ocr(image, roi_info, area=None, machine_code=None):
    """
    Perform full image OCR with screen detection and ROI filtering
    Uses PaddleOCR algorithm from paddleocr_reader.py
    
    Args:
        image: OpenCV image (numpy array BGR)
        roi_info: ROI data from roi_info.json
        area: Selected area (F1, F4, etc.) - optional
        machine_code: Selected machine code - optional
    
    Returns:
        dict: OCR results with screen detection info
    """
    try:
        if not HAS_PADDLEOCR:
            return {
                "success": False,
                "error": "PaddleOCR not available",
                "ocr_results": []
            }
        
        start_time = time.time()
        img_height, img_width = image.shape[:2]
        
        # Step 1: Full image OCR
        print("[*] Performing full image OCR with PaddleOCR...")
        results, _, _ = read_image_with_paddleocr(image)
        ocr_data = extract_ocr_data(results)
        
        ocr_time = time.time() - start_time
        print(f"[OK] PaddleOCR found {len(ocr_data)} text items in {ocr_time:.2f}s")
        
        # Step 2: Find matching screen
        # Returns: (machine_type, machine_code, screen_name, sub_page, sub_page_data, match_count, match_percentage)
        match_start = time.time()
        matched_machine_type, matched_machine, screen_name, sub_page, sub_page_data, match_count, match_percentage = find_matching_screen(
            ocr_data, roi_info,
            selected_area=area,
            selected_machine=machine_code,
            debug=True
        )
        match_time = time.time() - match_start
        
        if screen_name:
            print(f"[OK] Matched screen: {matched_machine_type}/{matched_machine}/{screen_name} (sub-page {sub_page})")
            print(f"[OK] Match: {match_count} Special_rois ({match_percentage:.1f}%)")
            
            # Step 3: Filter OCR results by IoU with ROIs
            filter_start = time.time()
            filtered_results = filter_ocr_by_roi(ocr_data, sub_page_data, img_width, img_height)
            filter_time = time.time() - filter_start
            
            print(f"[OK] Filtered to {len(filtered_results)} results matching ROIs (IoU >= {IOU_THRESHOLD:.0%})")
            
            # Use matched_machine_type from find_matching_screen
            machine_type = matched_machine_type if matched_machine_type else get_machine_type(matched_machine)
            
            # Convert to standard output format
            ocr_results = []
            for item in filtered_results:
                # Apply post-processing and decimal formatting
                text = post_process_ocr_text(item['text'])
                
                # Check if ROI has allowed_values - if yes, match with allowed_values first
                allowed_values = item.get('allowed_values', [])
                if allowed_values and len(allowed_values) > 0:
                    from .paddleocr_engine import match_text_with_allowed_values
                    text = match_text_with_allowed_values(text, allowed_values)
                else:
                    # Extract number from text if it contains label + value + unit
                    from .paddleocr_engine import extract_number_from_text
                    text = extract_number_from_text(text)
                
                text = apply_decimal_places_format(text, item['matched_roi'], machine_type, screen_name, matched_machine, sub_page)
                
                ocr_results.append({
                    "roi_index": item['matched_roi'],
                    "text": text,
                    "confidence": float(item['confidence']),
                    "has_text": True,
                    "original_value": item['text'],
                    "iou": float(item['iou'])
                })
            
            # Deduplicate: Keep only the result with highest IOU for each roi_index
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
            
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "area": matched_machine_type,  # machine_type serves as area identifier
                "machine_code": matched_machine,
                "machine_type": machine_type,
                "screen_id": screen_name,
                "sub_page": sub_page,
                "match_count": match_count,
                "match_percentage": match_percentage,
                "ocr_results": ocr_results,
                "roi_count": len(ocr_results),
                "processing_time": {
                    "ocr": ocr_time,
                    "matching": match_time,
                    "filtering": filter_time,
                    "total": total_time
                }
            }
        else:
            print("[WARNING] No matching screen found")
            
            # Return all OCR results without filtering
            ocr_results = []
            for item in ocr_data:
                text = post_process_ocr_text(item['text'])
                ocr_results.append({
                    "roi_index": f"OCR_{len(ocr_results)}",
                    "text": text,
                    "confidence": float(item['confidence']),
                    "has_text": True,
                    "original_value": item['text']
                })
            
            return {
                "success": True,
                "area": area,
                "machine_code": machine_code,
                "screen_id": None,
                "sub_page": None,
                "match_count": 0,
                "match_percentage": 0,
                "ocr_results": ocr_results,
                "roi_count": len(ocr_results),
                "warning": "No matching screen found - returning all OCR results"
            }
            
    except Exception as e:
        print(f"[ERROR] perform_full_image_ocr: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "ocr_results": []
        }


# Backward compatibility aliases
process_single_roi_optimized = process_single_roi_paddleocr
perform_ocr_on_roi = perform_ocr_on_roi_optimized
process_roi_with_retry_logic = process_roi_with_retry_logic_optimized


def save_roi_image_with_result(roi, roi_name, original_filename, detected_text, 
                               confidence, original_value, is_text_result=False, upload_folder='uploads'):
    """Save ROI image with OCR result overlay"""
    try:
        processed_folder = os.path.join(upload_folder, 'processed_roi')
        os.makedirs(processed_folder, exist_ok=True)
        
        base_filename = os.path.splitext(original_filename)[0]
        result_type = "text" if is_text_result else "number"
        roi_result_filename = f"{base_filename}_{roi_name}_{result_type}_detected.png"
        roi_result_path = os.path.join(processed_folder, roi_result_filename)
        
        # Create result image
        result_img = roi.copy()
        if len(result_img.shape) == 2:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding box
        cv2.rectangle(result_img, (2, 2), (result_img.shape[1]-2, result_img.shape[0]-2),
                     (0, 255, 0) if is_text_result else (255, 0, 0), 2)
        
        # Calculate font scale
        font_scale = max(0.4, min(result_img.shape[0], result_img.shape[1]) / 120)
        
        # Draw detected text
        cv2.putText(result_img, f"Detected: '{detected_text}'",
                  (5, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
        
        # Draw confidence
        cv2.putText(result_img, f"Confidence: {confidence:.3f}",
                  (5, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        # Draw original value if different
        if original_value and original_value != detected_text:
            cv2.putText(result_img, f"Original: '{original_value}'",
                      (5, int(result_img.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale, (0, 255, 0), 1)
        
        return roi_result_path
        
    except Exception as e:
        print(f"[ERROR] save_roi_image_with_result: {e}")
        return None
