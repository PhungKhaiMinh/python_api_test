"""
Image Routes Module
Handles image upload, retrieval, and deletion endpoints
"""

from flask import Blueprint, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import cv2
import time
from smart_detection_functions import auto_detect_machine_and_screen_smart
from utils import (
    get_roi_coordinates, get_roi_coordinates_with_subpage, get_special_region_coordinates,
    get_template_image_cached, get_reference_template_path, get_reference_template_path_with_subpage,
    ImageAligner, perform_ocr_on_roi_optimized, detect_hmi_screen
)

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


def detect_sub_page_from_special_region(image, machine_type, machine_code, screen_name):
    """
    Phát hiện sub-page dựa trên vùng đặc biệt
    
    Args:
        image: Ảnh HMI
        machine_type: Loại máy (F1, F41, F42)
        machine_code: Mã máy cụ thể (IE-F1-CWA01)
        screen_name: Tên màn hình
        
    Returns:
        str: Sub-page ("1", "2") hoặc None nếu không phát hiện được
    """
    try:
        import time
        from difflib import SequenceMatcher
        
        def similarity_ratio(a, b):
            """Tính độ tương đồng giữa 2 chuỗi (0.0 - 1.0)"""
            return SequenceMatcher(None, a.upper(), b.upper()).ratio()
        
        # Định nghĩa điều kiện cho từng máy (keywords quan trọng)
        # Mỗi sub-page có nhiều keywords, chỉ cần match một số là được
        sub_page_conditions = {
            "IE-F1-CWA01": {
                "1": ["ST02", "WING", "PRESENCE"],  # ST02 - WING PRESENCE CHECK
                "2": ["ST14", "LEAK", "TEST"]       # ST14 - LEAK TEST CHECK 1
            },
            # TODO: Thêm điều kiện cho máy khác
        }
        
        if machine_code not in sub_page_conditions:
            print(f"[WARNING] Sub-page detection not configured for machine: {machine_code}")
            return None
        
        conditions = sub_page_conditions[machine_code]
        best_match = None
        best_score = 0.0
        ocr_texts = {}
        
        # Thử đọc special_region của cả 2 sub-pages
        for sub_page in ["1", "2"]:
            try:
                # Lấy tọa độ vùng đặc biệt của sub-page này
                special_regions = get_special_region_coordinates(machine_type, machine_code, screen_name, sub_page)
                
                if not special_regions:
                    print(f"[DEBUG] No special regions found for sub-page {sub_page}")
                    continue
                
                # Sử dụng vùng đầu tiên
                region = special_regions[0]
                
                # Chuyển đổi tọa độ chuẩn hóa sang pixel
                height, width = image.shape[:2]
                x1 = int(region[0] * width)
                y1 = int(region[1] * height)
                x2 = int(region[2] * width)
                y2 = int(region[3] * height)
                
                # Lưu ảnh ROI special_regions để debug
                timestamp = int(time.time())
                roi_filename = f"special_region_{machine_code}_page{sub_page}_{timestamp}.jpg"
                roi_save_path = os.path.join("D:\\python_WREMBLY_test-main\\anhHMI", roi_filename)
                
                roi = image[y1:y2, x1:x2]
                
                try:
                    cv2.imwrite(roi_save_path, roi)
                    print(f"[DEBUG] Saved special region ROI (page {sub_page}) to: {roi_save_path}")
                    print(f"[DEBUG] ROI coordinates: ({x1}, {y1}) to ({x2}, {y2}), size: {roi.shape}")
                except Exception as e:
                    print(f"[ERROR] Failed to save ROI: {e}")
                
                # OCR trên vùng đặc biệt - Sử dụng toàn bộ image và để perform_ocr_on_roi_optimized xử lý
                # Không cắt ROI trước, để hàm OCR tự cắt và xử lý như ROI bình thường
                ocr_results = perform_ocr_on_roi_optimized(
                    image, [region], f"special_region_{machine_code}_page{sub_page}",
                    roi_names=[f"special_region_page{sub_page}"]
                )
                
                # Lấy kết quả OCR
                if ocr_results and len(ocr_results) > 0:
                    ocr_result = ocr_results[0]
                    text = ocr_result.get('text', '').upper().strip()
                    ocr_texts[sub_page] = text
                    print(f"[DEBUG] OCR text from sub-page {sub_page}: '{text}'")
                    
                    if not text:
                        continue
                    
                    # Tính điểm tương đồng với điều kiện của sub-page này
                    page_conditions = conditions.get(sub_page, [])
                    total_score = 0.0
                    matched_keywords = []
                    
                    for keyword in page_conditions:
                        # Kiểm tra xem có chứa keyword không (exact match)
                        if keyword in text:
                            total_score += 1.0
                            matched_keywords.append(keyword)
                        else:
                            # Tính độ tương đồng fuzzy
                            # Kiểm tra từng từ trong text
                            words = text.split()
                            max_word_similarity = 0.0
                            for word in words:
                                similarity = similarity_ratio(word, keyword)
                                max_word_similarity = max(max_word_similarity, similarity)
                            
                            # Chỉ tính điểm nếu similarity > 0.6 (60%)
                            if max_word_similarity > 0.6:
                                total_score += max_word_similarity
                                matched_keywords.append(f"{keyword}~{max_word_similarity:.2f}")
                    
                    # Normalize score (chia cho số keywords)
                    avg_score = total_score / len(page_conditions) if page_conditions else 0.0
                    
                    print(f"[DEBUG] Sub-page {sub_page}: score={avg_score:.3f}, matched={matched_keywords}")
                    
                    # Cập nhật best match nếu điểm cao hơn
                    if avg_score > best_score:
                        best_score = avg_score
                        best_match = sub_page
                
            except Exception as e:
                print(f"[ERROR] Error processing sub-page {sub_page}: {e}")
                continue
        
        # Quyết định sub-page dựa trên best score
        if best_match and best_score > 0.3:  # Ngưỡng tối thiểu 30% tương đồng
            print(f"[OK] Detected sub-page {best_match} with confidence {best_score:.3f}")
            print(f"[INFO] OCR results: {ocr_texts}")
            return best_match
        else:
            print(f"[WARNING] Cannot determine sub-page. Best score: {best_score:.3f}")
            print(f"[INFO] OCR results: {ocr_texts}")
            return None
        
    except Exception as e:
        print(f"[ERROR] Error detecting sub-page: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


@image_bp.route('/api/images', methods=['POST'])
def upload_image():
    """Upload image và thực hiện OCR"""
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
        
        # BƯỚC 1: PHÁT HIỆN VÀ TÁCH HMI SCREEN (quan trọng!)
        # ROI info được vẽ trên ảnh HMI đã tách, nên phải tách HMI trước khi OCR
        print(f"[*] Detecting and extracting HMI screen from image...")
        hmi_screen, processing_time = detect_hmi_screen(uploaded_image)
        
        hmi_detected = False
        hmi_image = uploaded_image  # Default: dùng ảnh gốc
        
        if hmi_screen is not None and hmi_screen.size > 0:
            hmi_detected = True
            hmi_image = hmi_screen  # ← QUAN TRỌNG: Sử dụng HMI đã tách
            print(f"[OK] HMI screen extracted successfully in {processing_time:.2f}s, size: {hmi_image.shape}")
        else:
            print(f"[WARN] Could not extract HMI screen in {processing_time:.2f}s, using original image")
        
        # BƯỚC 2: Auto detect machine and screen (trên HMI đã tách)
        print(f"[*] Auto-detecting machine and screen for {area}/{machine_code}...")
        detection_result = auto_detect_machine_and_screen_smart(
            hmi_image, area=area, machine_code=machine_code
        )
        
        machine_type = detection_result.get('machine_type', 'F41')
        screen_id = detection_result.get('screen_id', 'Main')
        template_path = detection_result.get('template_path')
        
        print(f"[OK] Detected: {machine_type} - {screen_id}")
        
        # BƯỚC 3: Detect sub-page nếu là màn hình "Reject_Summary"
        sub_page = None
        if screen_id == "Reject_Summary":
            print("[*] Detecting sub-page for Reject_Summary...")
            # ← QUAN TRỌNG: Sử dụng hmi_image (HMI đã tách) để phát hiện sub-page
            sub_page = detect_sub_page_from_special_region(hmi_image, machine_type, machine_code, screen_id)
            if sub_page:
                print(f"[OK] Detected sub-page: {sub_page}")
            else:
                print("[WARN] Could not detect sub-page, using default")
                sub_page = "1"  # Default to page 1
        
        # BƯỚC 4: Get ROI coordinates (với sub-page nếu có)
        if sub_page:
            roi_coordinates, roi_names = get_roi_coordinates_with_subpage(machine_type, screen_id, sub_page, machine_code)
        else:
            roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_id, machine_type)
        
        if not roi_coordinates:
            error_msg = f"No ROI coordinates for {machine_code}/{screen_id}"
            if sub_page:
                error_msg += f" (sub_page: {sub_page})"
            return jsonify({"error": error_msg}), 404
        
        # BƯỚC 5: Align image if template available
        # ← QUAN TRỌNG: Sử dụng hmi_image (HMI đã tách) cho alignment và OCR
        image = hmi_image
        if template_path:
            # Sử dụng template riêng cho từng trang nếu có sub_page
            if sub_page:
                subpage_template_path = get_reference_template_path_with_subpage(machine_type, screen_id, sub_page)
                if subpage_template_path:
                    print(f"[*] Using sub-page template: {subpage_template_path}")
                    template_img = get_template_image_cached(subpage_template_path)
                else:
                    print(f"[WARN] No sub-page template found, using general template")
                    template_img = get_template_image_cached(template_path)
            else:
                template_img = get_template_image_cached(template_path)
            
            if template_img is not None:
                aligner = ImageAligner(template_img, image)
                aligned_image = aligner.align_images()
                image = aligned_image
        
        # BƯỚC 6: Lưu ảnh từng vùng ROI để kiểm tra (trước khi OCR)
        print(f"[*] Saving individual ROI images for verification...")
        import time as time_module
        timestamp = int(time_module.time())
        
        # Tạo thư mục nếu chưa có
        roi_debug_dir = "D:\\python_WREMBLY_test-main\\anhHMI"
        os.makedirs(roi_debug_dir, exist_ok=True)
        
        # Lưu từng ROI
        for i, roi_coord in enumerate(roi_coordinates):
            try:
                # Tính toán tọa độ thực tế
                height, width = image.shape[:2]
                x1 = int(roi_coord[0] * width)
                y1 = int(roi_coord[1] * height)
                x2 = int(roi_coord[2] * width)
                y2 = int(roi_coord[3] * height)
                
                # Cắt ROI từ ảnh đã aligned
                roi_img = image[y1:y2, x1:x2]
                
                # Tạo tên file
                roi_name = roi_names[i] if i < len(roi_names) else f"ROI_{i}"
                roi_filename = f"{roi_name}_{machine_code}_{screen_id}"
                if sub_page:
                    roi_filename += f"_page{sub_page}"
                roi_filename += f"_{timestamp}_{i}.jpg"
                
                # Lưu ảnh
                roi_save_path = os.path.join(roi_debug_dir, roi_filename)
                cv2.imwrite(roi_save_path, roi_img)
                
                print(f"[DEBUG] Saved ROI {i+1}/{len(roi_coordinates)}: {roi_filename}")
                print(f"[DEBUG] ROI coordinates: ({x1}, {y1}) to ({x2}, {y2}), size: {roi_img.shape}")
                
            except Exception as e:
                print(f"[ERROR] Failed to save ROI {i+1}: {e}")
        cv2.imwrite(os.path.join(roi_debug_dir, f"hmi_aligned_{timestamp}.jpg"), image)
        print(f"[OK] All ROI images saved to: {roi_debug_dir}")
        
        # BƯỚC 7: Perform OCR (trên HMI đã tách và đã aligned)
        ocr_results = perform_ocr_on_roi_optimized(
            image, roi_coordinates, filename_with_timestamp,
            template_path=template_path,
            roi_names=roi_names,
            machine_code=machine_code,
            screen_id=screen_id,
            sub_page=sub_page
        )
        
        response_data = {
            "success": True,
            "filename": filename_with_timestamp,
            "machine_code": machine_code,
            "machine_type": machine_type,
            "screen_id": screen_id,
            "area": area,
            "detection_method": detection_result.get('detection_method'),
            "similarity_score": detection_result.get('similarity_score', 0),
            "hmi_detection": {
                "hmi_extracted": hmi_detected,
                "hmi_size": f"{hmi_image.shape[1]}x{hmi_image.shape[0]}",
                "extraction_status": "HMI screen extracted successfully" if hmi_detected else "Using original image"
            },
            "ocr_results": ocr_results,
            "roi_count": len(ocr_results)
        }
        
        # Thêm thông tin sub-page nếu có
        if sub_page:
            response_data["sub_page"] = sub_page
            response_data["sub_page_detected"] = True
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error in upload_image: {e}")
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

