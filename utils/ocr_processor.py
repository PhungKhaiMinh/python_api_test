"""
OCR Processor Module
Handles all OCR processing operations including EasyOCR integration,
parallel processing, and result formatting
"""

import cv2
import numpy as np
import re
import os
from .cache_manager import get_decimal_places_config_cached, get_roi_info_cached, get_machine_info_cached
from .config_manager import get_machine_type


# Global variables - will be set by app
HAS_EASYOCR = False
reader = None
_gpu_accelerator = None
_ocr_thread_pool = None


def init_ocr_globals(easyocr_available, ocr_reader, gpu_acc, thread_pool):
    """Initialize global OCR variables from app"""
    global HAS_EASYOCR, reader, _gpu_accelerator, _ocr_thread_pool
    HAS_EASYOCR = easyocr_available
    reader = ocr_reader
    _gpu_accelerator = gpu_acc
    _ocr_thread_pool = thread_pool


def process_single_roi_optimized(args):
    """Xử lý OCR cho một ROI đơn lẻ - GPU accelerated"""
    # Unpack args - có thể có hoặc không có machine_code và sub_page
    if len(args) == 8:
        (roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id, machine_code, sub_page) = args
    else:
        (roi_image, roi_name, machine_type, allowed_values, is_special_on_off_case, screen_id) = args
        machine_code = None
        sub_page = None
    
    try:
        # Trường hợp đặc biệt ON/OFF - phân tích màu sắc
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
        
        if not HAS_EASYOCR or reader is None:
            return {
                "roi_index": roi_name,
                "text": "OCR_NOT_AVAILABLE",
                "confidence": 0,
                "has_text": False,
                "original_value": ""
            }
        
        # Resize nếu ROI quá nhỏ
        height, width = roi_image.shape[:2]
        if height < 30 or width < 30:
            scale_factor = max(30/height, 30/width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            if _gpu_accelerator:
                roi_image = _gpu_accelerator.resize_gpu(roi_image, (new_width, new_height), cv2.INTER_CUBIC)
            else:
                roi_image = cv2.resize(roi_image, (new_width, new_height), cv2.INTER_CUBIC)
        
        # Convert to grayscale
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
        
        # Thực hiện OCR
        ocr_results = reader.readtext(
            roi_processed,
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
            best_result = max(ocr_results, key=lambda x: x[2])
            best_text = best_result[1]
            best_confidence = best_result[2]
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
        print(f"Error in process_single_roi_optimized: {str(e)}")
        return {
            "roi_index": roi_name,
            "text": "ERROR",
            "confidence": 0.0,
            "has_text": False,
            "original_value": "",
            "error": str(e)
        }


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
            
            # Check if mostly digit-like characters
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
        
        # Cấu trúc mới cho Reject Summary với sub-page:
        # machine_type > screen_id > machine_code > sub_page > roi_name
        if screen_id == "Reject Summary" and sub_page and machine_code:
            if machine_type in decimal_config:
                if screen_id in decimal_config[machine_type]:
                    if machine_code in decimal_config[machine_type][screen_id]:
                        if sub_page in decimal_config[machine_type][screen_id][machine_code]:
                            if roi_name in decimal_config[machine_type][screen_id][machine_code][sub_page]:
                                decimal_places = decimal_config[machine_type][screen_id][machine_code][sub_page][roi_name]
        else:
            # Cấu trúc thường: machine_type > screen_id > roi_name
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
        print(f"Error applying decimal format: {e}")
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
    """Process ROI with retry logic - optimized version"""
    # For now, call single process directly
    # Can add retry logic later if needed
    return process_single_roi_optimized(roi_args)


def perform_ocr_on_roi_optimized(image, roi_coordinates, original_filename, 
                                 template_path=None, roi_names=None, machine_code=None, screen_id=None, sub_page=None):
    """
    Phiên bản tối ưu của perform_ocr_on_roi với parallel processing
    """
    try:
        if roi_coordinates is None or len(roi_coordinates) == 0:
            return []
        
        if roi_names is None or len(roi_names) != len(roi_coordinates):
            roi_names = [f"ROI_{i}" for i in range(len(roi_coordinates))]
        
        if not HAS_EASYOCR or reader is None:
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
        
        # Prepare ROI args for parallel processing
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
                print(f"Error preparing ROI {i}: {e}")
                continue
        
        # Process ROIs in parallel
        if _ocr_thread_pool:
            results = list(_ocr_thread_pool.map(process_single_roi_optimized, roi_args))
        else:
            results = [process_single_roi_optimized(args) for args in roi_args]
        
        return results
        
    except Exception as e:
        print(f"Error in perform_ocr_on_roi_optimized: {e}")
        return []


# Alias for backward compatibility
perform_ocr_on_roi = perform_ocr_on_roi_optimized
process_roi_with_retry_logic = process_roi_with_retry_logic_optimized


def save_roi_image_with_result(roi, roi_name, original_filename, detected_text, 
                               confidence, original_value, is_text_result=False, upload_folder='uploads'):
    """Lưu ảnh ROI với kết quả OCR overlay"""
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
        print(f"Error saving ROI image: {e}")
        return None

