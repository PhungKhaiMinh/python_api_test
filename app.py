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

# Thêm try-except khi import EasyOCR
try:
    import easyocr
    HAS_EASYOCR = True
    # Khởi tạo đối tượng reader ở cấp độ global
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR initialized successfully")
except ImportError:
    print("EasyOCR not installed. OCR functionality will be limited.")
    HAS_EASYOCR = False
    reader = None

app = Flask(__name__)

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
            "upload_folder": UPLOAD_FOLDER,
            "roi_data_folder": ROI_DATA_FOLDER,
            "ocr_results_folder": OCR_RESULTS_FOLDER,
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

# Đảm bảo thư mục uploads tồn tại
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROI_DATA_FOLDER'] = ROI_DATA_FOLDER
app.config['OCR_RESULTS_FOLDER'] = OCR_RESULTS_FOLDER
app.config['REFERENCE_IMAGES_FOLDER'] = REFERENCE_IMAGES_FOLDER  # Thêm cấu hình cho thư mục reference_images
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn kích thước file 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Thêm hàm căn chỉnh ảnh từ wrap_perspective.py
class ImageAligner:
    def __init__(self, template_img, source_img):
        """Khởi tạo với ảnh mẫu và ảnh nguồn đã được đọc bởi OpenCV"""
        self.template_img = template_img.copy()
        self.source_img = source_img.copy()
        
        # Store warped image
        self.warped_img = None
        
        # Initialize feature detector (SIFT works well for this type of image)
        self.detector = cv2.SIFT_create()
        
    def align_images(self):
        """Căn chỉnh ảnh nguồn để khớp với ảnh mẫu bằng feature matching và homography."""
        # Convert images to grayscale for feature detection
        template_gray = cv2.cvtColor(self.template_img, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(self.source_img, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        template_keypoints, template_descriptors = self.detector.detectAndCompute(template_gray, None)
        source_keypoints, source_descriptors = self.detector.detectAndCompute(source_gray, None)
        
        # Match features using FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(source_descriptors, template_descriptors, k=2)
        
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
    aligned_filename = f"aligned_{original_filename}"
    aligned_path = os.path.join(aligned_folder, aligned_filename)
    cv2.imwrite(aligned_path, aligned_image)
    
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
def get_roi_coordinates(machine_code, screen_id=None):
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
        
        if screen_id is not None:
            # Lấy tên màn hình từ screen_id
            screen_name = None
            machine_screens_path = 'roi_data/machine_screens.json'
            if not os.path.exists(machine_screens_path):
                machine_screens_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_data/machine_screens.json')
            
            with open(machine_screens_path, 'r', encoding='utf-8') as f:
                machine_screens = json.load(f)
                
            if machine_code in machine_screens.get("machines", {}):
                for screen in machine_screens["machines"][machine_code].get("screens", []):
                    if screen.get("id") == screen_id:
                        screen_name = screen.get("screen_id")
                        break
            
            print(f"Looking for ROIs in machine: {machine_code}, screen: {screen_name} (id: {screen_id})")
            
            if screen_name and machine_code in roi_data.get("machines", {}) and "screens" in roi_data["machines"][machine_code]:
                if screen_name in roi_data["machines"][machine_code]["screens"]:
                    roi_list = roi_data["machines"][machine_code]["screens"][screen_name]
                    
                    # Xử lý định dạng ROI mới (dictionary với name và coordinates)
                    roi_coordinates = []
                    roi_names = []
                    
                    for roi_item in roi_list:
                        if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                            # Định dạng mới - dictionary với name và coordinates
                            roi_coordinates.append(roi_item["coordinates"])
                            roi_names.append(roi_item["name"])
                            print(f"Added ROI: {roi_item['name']} with coordinates: {roi_item['coordinates']}")
                        else:
                            # Định dạng cũ - mảng tọa độ
                            roi_coordinates.append(roi_item)
                            roi_names.append(f"ROI_{len(roi_names)}")
                            print(f"Added unnamed ROI with coordinates: {roi_item}")
                    
                    print(f"Found {len(roi_coordinates)} ROIs with names: {roi_names}")
                    return roi_coordinates, roi_names
                else:
                    print(f"Screen '{screen_name}' not found in roi_info.json for machine {machine_code}")
            else:
                print(f"Machine '{machine_code}' or screen path not found in roi_info.json")
        
        print(f"No ROI coordinates found for machine: {machine_code}, screen_id: {screen_id}")
        return [], []
    except Exception as e:
        print(f"Error reading ROI coordinates: {str(e)}")
        traceback.print_exc()
        return [], []

# Sửa lại hàm perform_ocr_on_roi để sử dụng ảnh đã căn chỉnh
def perform_ocr_on_roi(image, roi_coordinates, original_filename, template_path=None, roi_names=None):
    """
    Thực hiện OCR trên các vùng ROI đã xác định
    
    Args:
        image: Ảnh đầu vào
        roi_coordinates: Danh sách các tọa độ ROI
        original_filename: Tên file gốc
        template_path: Đường dẫn đến ảnh template nếu có
        roi_names: Danh sách tên của các ROI
        
    Returns:
        Danh sách kết quả OCR cho mỗi ROI
    """
    try:
        # Kiểm tra các tham số đầu vào
        if roi_coordinates is None or len(roi_coordinates) == 0:
            print("No ROI coordinates provided")
            return []
        
        print(f"Processing {len(roi_coordinates)} ROIs")
        
        # Nếu không có tên ROI được truyền vào, tạo tên mặc định
        if roi_names is None or len(roi_names) != len(roi_coordinates):
            print("Creating default ROI names")
            roi_names = [f"ROI_{i}" for i in range(len(roi_coordinates))]
        
        # Kiểm tra EasyOCR đã được khởi tạo chưa
        if not HAS_EASYOCR or reader is None:
            print("EasyOCR is not available. Cannot perform OCR.")
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
        print(f"Image dimensions: {img_width}x{img_height}")
        
        # Lấy thông tin máy hiện tại
        machine_info = get_current_machine_info()
        if not machine_info:
            print("Không thể lấy thông tin máy hiện tại")
            return []
        
        machine_code = machine_info['machine_code']
        screen_id = machine_info['screen_id']
        
        # Đọc cấu hình số thập phân
        decimal_places_config = get_decimal_places_config()
        
        # Tiền xử lý ảnh và thực hiện OCR trên mỗi ROI
        results = []
        for i, coords in enumerate(roi_coordinates):
            try:
                # Đảm bảo coords có đúng 4 giá trị
                if len(coords) != 4:
                    print(f"Invalid coordinates for ROI {i}: {coords}")
                    continue
                
                # Chuyển đổi tọa độ nếu cần
                is_normalized = False
                for value in coords:
                    if isinstance(value, float) and 0 <= value <= 1:
                        is_normalized = True
                        break
                
                if is_normalized:
                    print(f"Detected normalized coordinates for ROI {i}: {coords}")
                    # Chuyển đổi từ tọa độ chuẩn hóa sang tọa độ pixel
                    x1, y1, x2, y2 = coords
                    x1, x2 = int(x1 * img_width), int(x2 * img_width)
                    y1, y2 = int(y1 * img_height), int(y2 * img_height)
                    print(f"Converted to pixel coordinates: ({x1},{y1},{x2},{y2})")
                else:
                    # Đã là tọa độ pixel, chỉ cần chuyển sang int
                    x1, y1, x2, y2 = coords
                    x1, x2 = int(float(x1)), int(float(x2))
                    y1, y2 = int(float(y1)), int(float(y2))
                
                # Đảm bảo thứ tự tọa độ chính xác
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                roi_name = roi_names[i] if i < len(roi_names) else f"ROI_{i}"
                print(f"Processing ROI {i} ({roi_name}) with coordinates: ({x1},{y1},{x2},{y2})")
                
                # Kiểm tra tọa độ hợp lệ
                if x1 < 0 or y1 < 0 or x2 >= image.shape[1] or y2 >= image.shape[0] or x1 >= x2 or y1 >= y2:
                    print(f"Invalid coordinates: ({x1},{y1},{x2},{y2}), image shape: {image.shape}")
                    continue
                        
                # Cắt ROI
                roi = image[y1:y2, x1:x2]
                image_aligned = image
                # Tiền xử lý ROI cho OCR
                roi_processed = preprocess_roi_for_ocr(roi, i, original_filename, roi_name, image_aligned, x1, y1, x2, y2)
                
                # Kiểm tra xem ảnh đã tiền xử lý có thành công không
                if roi_processed is None:
                    print(f"ROI {i} preprocessing failed")
                    continue
                            
                # Thực hiện OCR
                has_text = False
                best_text = ""
                best_confidence = 0
                original_value = ""
                
                # Thử OCR trên vùng an toàn trước
                # safe_zone_results = check_safe_zone(roi_processed, margin_percent=10.0)
                # if safe_zone_results and len(safe_zone_results) > 0:
                #     print(f"Found text in safe zone for ROI {i} ({roi_name})")
                #     best_result = max(safe_zone_results, key=lambda x: x[2])
                #     best_text = best_result[1]
                #     best_confidence = best_result[2]
                #     original_value = best_text
                #     has_text = True
                # else:
                print(f"No text found in safe zone for ROI {i} ({roi_name}), trying full ROI")
                # Thử OCR trên toàn bộ ROI
                try:
                    # Specify the characters to read (digits only)
                    ocr_results = reader.readtext(roi_processed, allowlist='0123456789.', detail=1, paragraph=False,batch_size=1, text_threshold=0.6, link_threshold=0.3, low_text=0.4, mag_ratio=1, slope_ths=0.05)
                    if ocr_results and len(ocr_results) > 0:
                        # Lấy kết quả có confidence cao nhất
                        best_result = max(ocr_results, key=lambda x: x[2])
                        best_text = best_result[1]
                        best_confidence = best_result[2]
                        original_value = best_text
                        has_text = True
                        print(f"Found text in full ROI: '{best_text}' with confidence {best_confidence}")
                    else:
                        print(f"No text found in ROI {i} ({roi_name})")
                except Exception as ocr_error:
                    print(f"OCR error on ROI {i} ({roi_name}): {str(ocr_error)}")
                
                # Kiểm tra và áp dụng quy định decimal_places nếu text là số
                formatted_text = best_text
                if has_text and re.match(r'^-?\d+\.?\d*$', best_text):
                    try:
                        # Kiểm tra xem có cấu hình cho ROI này không
                        if (machine_code in decimal_places_config and 
                            screen_id in decimal_places_config[machine_code] and 
                            roi_name in decimal_places_config[machine_code][screen_id]):
                            
                            decimal_places = int(decimal_places_config[machine_code][screen_id][roi_name])
                            print(f"Found decimal places config for ROI {roi_name}: {decimal_places}")
                            
                            # Chuyển đổi thành số float
                            num_value = float(best_text)
                            original_str = str(num_value)
                            
                            # Xử lý các trường hợp khác nhau dựa trên decimal_places
                            if decimal_places == 0:
                                # Nếu decimal_places là 0, giữ lại tất cả các chữ số nhưng bỏ dấu chấm
                                formatted_text = str(num_value).replace('.', '')
                                print(f"Removed decimal point for ROI {roi_name}: {formatted_text}")
                            else:
                                # Đếm số chữ số thập phân hiện tại
                                current_decimal_places = 0
                                if '.' in original_str:
                                    dec_part = original_str.split('.')[1]
                                    # Đếm số thập phân có ý nghĩa (bỏ các số 0 không có ý nghĩa ở cuối)
                                    if all(d == '0' for d in dec_part):
                                        # Nếu tất cả là số 0 thì giữ nguyên số lượng số 0
                                        current_decimal_places = len(dec_part)
                                    else:
                                        # Bỏ các số 0 ở cuối
                                        dec_part_significant = dec_part.rstrip('0')
                                        current_decimal_places = len(dec_part_significant)
                                
                                # Nếu số chữ số thập phân hiện tại bằng đúng số chữ số thập phân cần có
                                if current_decimal_places == decimal_places:
                                    # Giữ nguyên số
                                    formatted_text = original_str
                                    print(f"Already correct format for ROI {roi_name}: {original_str}")
                                else:
                                    # Phần số nguyên và phần thập phân riêng biệt
                                    if '.' in original_str:
                                        int_part, dec_part = original_str.split('.')
                                        
                                        # Trường hợp đặc biệt: Số thập phân chỉ có 0 (ví dụ: 230.0, 1000.00)
                                        if all(d == '0' for d in dec_part):
                                            # Giữ nguyên phần nguyên và thêm/bớt số 0 thập phân
                                            if decimal_places > 0:
                                                # Điều chỉnh số lượng số 0 theo decimal_places
                                                formatted_text = f"{int_part}.{'0' * decimal_places}"
                                            else:
                                                # Nếu decimal_places = 0, bỏ dấu chấm
                                                formatted_text = int_part
                                        else:
                                            # Số thập phân có các chữ số khác 0
                                            # Xóa dấu chấm và lấy tất cả các chữ số để định dạng lại
                                            num_str = int_part + dec_part.rstrip('0')
                                            
                                            # Kiểm tra độ dài chuỗi số
                                            if len(num_str) <= decimal_places:
                                                # Trường hợp số chữ số ít hơn hoặc bằng số chữ số thập phân
                                                # Thêm số 0 phía trước và đặt dấu chấm sau số 0 đầu tiên
                                                padded_str = num_str.zfill(decimal_places)
                                                formatted_text = f"0.{padded_str}"
                                            else:
                                                # Đặt dấu chấm vào vị trí thích hợp: (độ dài - decimal_places)
                                                insert_pos = len(num_str) - decimal_places
                                                formatted_text = f"{num_str[:insert_pos]}.{num_str[insert_pos:]}"
                                    else:
                                        # Không có dấu chấm (số nguyên)
                                        num_str = original_str
                                        
                                        # Thêm phần thập phân nếu cần
                                        if decimal_places > 0:
                                            # Đặt dấu chấm vào vị trí thích hợp
                                            if len(num_str) <= decimal_places:
                                                # Số chữ số ít hơn hoặc bằng số thập phân cần có
                                                padded_str = num_str.zfill(decimal_places)
                                                formatted_text = f"0.{padded_str}"
                                            else:
                                                # Đặt dấu chấm vào vị trí thích hợp
                                                insert_pos = len(num_str) - decimal_places
                                                formatted_text = f"{num_str[:insert_pos]}.{num_str[insert_pos:]}"
                                        else:
                                            # Giữ nguyên số nguyên nếu không cần thập phân
                                            formatted_text = num_str
                                    
                                    print(f"Formatted value for ROI {roi_name}: Original: {num_value}, Formatted: {formatted_text}")
                    except Exception as e:
                        print(f"Error applying decimal places format for ROI {roi_name}: {str(e)}")
                
                # Thêm kết quả cho ROI này
                results.append({
                    "roi_index": roi_name,
                    "text": formatted_text,  # Trả về text đã định dạng theo quy định số chữ số thập phân
                    "confidence": best_confidence,
                    "has_text": has_text,
                    "original_value": original_value
                })
                print(f"Added result for ROI {i} ({roi_name}): Original: '{best_text}', Formatted: '{formatted_text}'")
                
                # Kiểm tra độ tin cậy của OCR
                if best_confidence < 0.3:  # Nếu độ tin cậy < 40%
                    print(f"Warning: Low confidence ({best_confidence:.2f}) for ROI {roi_name}, text: '{best_text}'")
                
            except Exception as roi_error:
                print(f"Error processing ROI {i}: {str(roi_error)}")
                traceback.print_exc()
                continue
        
        # Kiểm tra nếu kết quả có độ tin cậy cao nhất < 40%, trả về mảng rỗng
        if results and max(results, key=lambda x: x["confidence"])["confidence"] < 0.3:
            print(f"The result with highest confidence is still below 40%. Returning empty results.")
            return []
            
        return results
    
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        traceback.print_exc()
        
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

# Sửa lại route upload_image để sử dụng template reference từ thư mục mới
@app.route('/api/images', methods=['POST'])
def upload_image():
    """API để tải lên ảnh và thực hiện OCR trên các vùng ROI được định nghĩa"""
    # Kiểm tra xem có file trong request không
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Kiểm tra xem có chọn file chưa
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Kiểm tra machine_code và screen_id từ form data
    machine_code = request.form.get('machine_code')
    screen_id = request.form.get('screen_id')
    
    # Lấy template_image path từ form data nếu có
    template_image = request.form.get('template_image')
    
    # Nếu không có trong form data, thử lấy từ thông tin máy hiện tại
    if not machine_code or not screen_id:
        machine_info = get_current_machine_info()
        if machine_info:
            machine_code = machine_info['machine_code']
            screen_id = machine_info['screen_id']
        else:
            return jsonify({
                "error": "Missing machine_code and screen_id. Please provide them in form data or set them using /api/set_machine_screen first."
            }), 400
    
    print(f"Processing image for machine: {machine_code}, screen: {screen_id}")
    
    # Xác định đường dẫn template image
    template_path = None
    if template_image:
        # Sử dụng template được chỉ định trong form data
        template_path = os.path.join(app.config['REFERENCE_IMAGES_FOLDER'], template_image)
        if not os.path.exists(template_path):
            template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_image)
    else:
        # Tìm template trong thư mục reference_images dựa trên machine_code và screen_id
        template_path = get_reference_template_path(machine_code, screen_id)
    
    # Kiểm tra template_path tồn tại
    if template_path and not os.path.exists(template_path):
        print(f"Template path {template_path} does not exist")
        template_path = None
    
    if template_path:
        print(f"Using template image: {template_path}")
    else:
        print("No template image found, will not perform alignment")
    
    # Kiểm tra file có phải là hình ảnh không
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Tạo tên file cho ảnh (thêm machine_code và screen_id vào tên file)
        base_name, extension = os.path.splitext(filename)
        filename = f"{base_name}_{machine_code}_{screen_id}{extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Kiểm tra xem file đã tồn tại chưa và xóa nếu có
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Lưu file
        file.save(file_path)
        print(f"Saved image to: {file_path}")
        
        # Xử lý ảnh và thực hiện OCR
        try:
            # Đọc ảnh bằng OpenCV
            image = cv2.imread(file_path)
            if image is None:
                return jsonify({"error": "Failed to read image with OpenCV"}), 500
            
            print(f"Image loaded successfully, shape: {image.shape}")
            
            # Tìm id màn hình dựa trên screen_id (tên màn hình)
            screen_numeric_id = get_screen_numeric_id(machine_code, screen_id)
            print(f"Found numeric screen ID: {screen_numeric_id} for screen name: {screen_id}")
            
            # Lấy tọa độ ROI và tên ROI dựa trên máy và màn hình
            roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_numeric_id)
            
            if not roi_coordinates:
                return jsonify({"error": f"Không tìm thấy tọa độ ROI cho máy {machine_code}, màn hình {screen_id}"}), 400
            
            print(f"Found {len(roi_coordinates)} ROI coordinates and {len(roi_names)} ROI names")
            
            # Cập nhật thông tin màn hình hiện tại
            try:
                # Cập nhật parameter_order_value.txt
                parameter_order_file_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'parameter_order_value.txt')
                with open(parameter_order_file_path, 'w', encoding='utf-8') as f:
                    f.write(str(screen_numeric_id))
                print(f"Updated parameter_order_value.txt with screen ID: {screen_numeric_id}")
            except Exception as e:
                print(f"Warning: Failed to update parameter_order_value.txt: {str(e)}")
            
            # Tiền xử lý ảnh với căn chỉnh nếu có template
            processed_results = []
            if template_path:
                print("Processing image with alignment...")
                # Đọc ảnh template
                template_img = cv2.imread(template_path)
                if template_img is None:
                    print(f"Failed to read template image: {template_path}")
                else:
                    # Căn chỉnh ảnh
                    aligner = ImageAligner(template_img, image)
                    aligned_image = aligner.align_images()
                    
                    # Lưu ảnh đã căn chỉnh
                    aligned_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned')
                    if not os.path.exists(aligned_folder):
                        os.makedirs(aligned_folder)
                    
                    aligned_filename = f"aligned_{filename}"
                    aligned_path = os.path.join(aligned_folder, aligned_filename)
                    cv2.imwrite(aligned_path, aligned_image)
                    print(f"Saved aligned image to: {aligned_path}")
                    
                    # Thay đổi biến image thành ảnh đã căn chỉnh
                    image = aligned_image
            
            # Thực hiện OCR trên các vùng ROI
            print(f"Performing OCR on {len(roi_coordinates)} ROIs")
            ocr_results = perform_ocr_on_roi(
                image, 
                roi_coordinates, 
                filename, 
                template_path,
                roi_names
            )
            
            # Lưu kết quả OCR vào file JSON
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ocr_result_filename = f"ocr_result_{timestamp}_{base_name}_{machine_code}_{screen_id}.json"
            ocr_result_path = os.path.join(app.config['OCR_RESULTS_FOLDER'], ocr_result_filename)
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(ocr_result_path), exist_ok=True)
            
            # Tạo cấu trúc dữ liệu giống với file OCR result
            result_data = {
                    "filename": filename,
                    "machine_code": machine_code,
                    "screen_id": screen_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "template_path": template_path if template_path else None,
                    "results": ocr_results
            }
            
            # Lưu kết quả OCR vào file JSON
            with open(ocr_result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved OCR results to: {ocr_result_path}")
            
            # Trả về kết quả với cấu trúc giống file OCR
            return jsonify(result_data), 201
            
        except Exception as e:
            print(f"Error in image processing: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Lỗi khi xử lý ảnh: {str(e)}"}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

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
        
        machine_code = machine_info['machine_code']
        screen_id = machine_info['screen_id']
        
        # Đọc thông tin máy và màn hình
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if machine_code not in data['machines']:
            return jsonify({"error": f"Machine {machine_code} not found"}), 404
        
        # Lấy thông tin màn hình đã chọn
        selected_screen = next(screen for screen in data['machines'][machine_code]['screens'] if screen['id'] == screen_id)
        
        return jsonify({
            "machine": {
                "machine_code": machine_code,
                "name": data['machines'][machine_code]['name']
            },
            "screen": selected_screen
        }), 200
    
    except ValueError as ve:
        return jsonify({"error": f"Invalid screen ID format: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to get current machine and screen: {str(e)}"}), 500

# Hàm helper để lấy thông tin máy hiện tại
def get_current_machine_info():
    """Lấy thông tin máy hiện tại từ machine_screens.json và parameter_order_value.txt"""
    try:
        parameter_order_file_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'parameter_order_value.txt')
        if not os.path.exists(parameter_order_file_path):
            return None
        
        with open(parameter_order_file_path, 'r', encoding='utf-8') as f:
            screen_numeric_id = int(f.read().strip())
        
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm máy và màn hình dựa trên screen_numeric_id
        for machine_code, machine_info in data['machines'].items():
            for screen in machine_info['screens']:
                if screen['id'] == screen_numeric_id:
                    return {
                        'machine_code': machine_code,
                        'screen_id': screen['screen_id'],  # Trả về tên màn hình
                        'screen_numeric_id': screen_numeric_id,
                        'description': screen.get('description', '')
                    }
        
        return None
    except Exception as e:
        print(f"Error getting current machine info: {str(e)}")
        return None

# API mới: Lấy thông tin máy theo ID
@app.route('/api/machines', methods=['GET'])
def get_machine_info():
    """Lấy thông tin chi tiết về một máy cụ thể và các màn hình HMI của nó"""
    try:
        machine_code = request.args.get('machine_code', '').strip().upper()
        if not machine_code:
            # Nếu không có machine_code, trả về danh sách tất cả các máy
            machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
            if not os.path.exists(machine_screens_path):
                return jsonify({"error": "Machine screens configuration not found"}), 404
            
            with open(machine_screens_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Trả về danh sách các mã máy
            machine_list = []
            for machine_code, machine_info in data['machines'].items():
                machine_list.append({
                    "machine_code": machine_code,
                    "name": machine_info['name']
                })
            
            return jsonify({"machines": machine_list}), 200
        
        # Lấy thông tin máy cụ thể
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if machine_code not in data['machines']:
            return jsonify({"error": f"Machine {machine_code} not found"}), 404
        
        # Trả về thông tin chi tiết của máy
        machine_info = data['machines'][machine_code]
        
        # Lấy thông tin tất cả các màn hình của máy này
        screens = []
        for screen in machine_info['screens']:
            screen_id = screen['id']
            roi_coordinates = get_roi_coordinates(machine_code, screen_id)
            if roi_coordinates:
                screen['roi_count'] = len(roi_coordinates)
                
                # Kiểm tra cấu hình decimal places
                decimal_config = get_decimal_places_config()
                has_decimal_config = (machine_code in decimal_config and 
                                     screen_id in decimal_config[machine_code])
                
                screen['has_decimal_config'] = has_decimal_config
                
                # Kiểm tra từng ROI có cấu hình decimal places không
                roi_info = []
                for i in range(len(roi_coordinates)):
                    roi_item = {
                        "index": i,
                        "name": roi_names[i] if i < len(roi_names) else f"ROI_{i}",
                        "coordinates": roi_coordinates[i]
                    }
                    
                    # Thêm thông tin decimal_places nếu có
                    if (machine_code in decimal_config and 
                        screen_id in decimal_config[machine_code]):
                        # Tìm decimal_places cho ROI này dựa trên tên
                        if roi_item["name"] in decimal_config[machine_code][screen_id]:
                            roi_item["decimal_places"] = decimal_config[machine_code][screen_id][roi_item["name"]]
                        # Hỗ trợ tương thích ngược với cấu trúc cũ (sử dụng chỉ số)
                        elif str(i) in decimal_config[machine_code][screen_id]:
                            roi_item["decimal_places"] = decimal_config[machine_code][screen_id][str(i)]
                    
                    roi_info.append(roi_item)
            
            screens.append(screen)
        
        response = {
            "machine_code": machine_code,
            "name": machine_info['name'],
            "screens": screens
        }
        
        return jsonify(response), 200
    
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

# API mới: Kiểm tra tình trạng cấu hình ROI và decimal places cho máy và màn hình
@app.route('/api/machine_screen_status', methods=['GET'])
def check_machine_screen_status():
    """
    Kiểm tra tình trạng cấu hình của một máy và màn hình cụ thể.
    Kiểm tra xem đã có ROI và cấu hình decimal places cho mỗi ROI hay chưa.
    
    Query parameters:
    - machine_code: Mã máy (ví dụ: "F1")
    - screen_id: ID màn hình (số nguyên)
    
    Nếu không cung cấp tham số, lấy thông tin từ máy và màn hình hiện tại.
    """
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
        
        if machine_code not in data['machines']:
            return jsonify({"error": f"Machine {machine_code} not found"}), 404
        
        # Kiểm tra screen_id có tồn tại không
        screen_exists = any(screen['id'] == screen_id for screen in data['machines'][machine_code]['screens'])
        if not screen_exists:
            return jsonify({"error": f"Screen ID {screen_id} not found for machine {machine_code}"}), 404
        
        # Kiểm tra ROI
        roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_id)
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
        
        # Lấy thông tin screen name
        screen_name = ""
        for screen in data['machines'][machine_code]['screens']:
            if screen['id'] == screen_id:
                screen_name = screen['name']
                break
        
        # Trả về kết quả tình trạng
        status = {
            "machine_code": machine_code,
            "machine_name": data['machines'][machine_code]['name'],
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
def get_screen_numeric_id(machine_code, screen_name):
    """Lấy ID số của màn hình dựa trên tên màn hình"""
    try:
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if machine_code not in data['machines']:
            return None
        
        # Tìm màn hình có screen_id (tên màn hình) trùng khớp
        for screen in data['machines'][machine_code]['screens']:
            if screen['screen_id'] == screen_name:
                return screen['id']
        
        return None
    except Exception as e:
        print(f"Error getting screen numeric ID: {str(e)}")
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
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
        processed_filename = f"{base_filename}_roi_{i}_processed.png"
        processed_path = os.path.join(processed_folder, processed_filename)
        cv2.imwrite(processed_path, closing)  # Save the processed image directly
        
    return results

# API mới: Upload/Update ảnh template mẫu
@app.route('/api/reference_images', methods=['POST'])
def upload_reference_image():
    """
    API để tải lên ảnh template mẫu dùng cho việc căn chỉnh
    
    Form data parameters:
    - file: File ảnh template mẫu
    - machine_code: Mã máy (ví dụ: F1)
    - screen_id: Mã màn hình (ví dụ: Faults)
    """
    # Kiểm tra xem có file trong request không
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Kiểm tra xem có chọn file chưa
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Kiểm tra machine_code và screen_id từ form data
    machine_code = request.form.get('machine_code')
    screen_id = request.form.get('screen_id')
    
    if not machine_code or not screen_id:
        return jsonify({
            "error": "Missing machine_code or screen_id. Both are required."
        }), 400
    
    # Kiểm tra file có phải là hình ảnh không
    if file and allowed_file(file.filename):
        # Tạo tên file cho ảnh template (format: template_{machine_code}_{screen_id}.jpg)
        extension = os.path.splitext(file.filename)[1].lower()
        reference_filename = f"template_{machine_code}_{screen_id}{extension}"
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
                "machine_code": machine_code,
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
    
    # Filter theo machine_code và screen_id nếu được cung cấp
    machine_code = request.args.get('machine_code')
    screen_id = request.args.get('screen_id')
    
    for filename in os.listdir(app.config['REFERENCE_IMAGES_FOLDER']):
        if allowed_file(filename):
            file_path = os.path.join(app.config['REFERENCE_IMAGES_FOLDER'], filename)
            
            # Trích xuất thông tin từ tên file
            file_info = filename.split('_')
            if len(file_info) >= 3 and file_info[0] == 'template':
                file_machine_code = file_info[1]
                # Trích xuất screen_id (có thể chứa dấu '_')
                file_screen_id = '_'.join(file_info[2:]).split('.')[0]
                
                # Lọc theo machine_code nếu được cung cấp
                if machine_code and file_machine_code != machine_code:
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
                    "machine_code": file_machine_code,
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
    """Xóa ảnh template mẫu"""
    file_path = os.path.join(app.config['REFERENCE_IMAGES_FOLDER'], secure_filename(filename))
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": f"Reference template {filename} has been deleted successfully"}), 200
    else:
        return jsonify({"error": "Reference template not found"}), 404

# Hàm mới: Lấy đường dẫn đến ảnh template mẫu dựa trên machine_code và screen_id
def get_reference_template_path(machine_code, screen_id):
    """
    Tìm kiếm ảnh template mẫu dựa trên machine_code và screen_id
    
    Returns:
        str: Đường dẫn đến file template nếu tìm thấy, None nếu không tìm thấy
    """
    reference_folder = app.config['REFERENCE_IMAGES_FOLDER']
    
    # Tạo pattern tên file
    file_pattern = f"template_{machine_code}_{screen_id}.*"
    
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
        
    Returns:
        Ảnh ROI đã được tiền xử lý
    """
    # Sử dụng tên ROI nếu có, nếu không sử dụng chỉ số
    x1, y1, x2, y2 = x1, y1, x2, y2
    identifier = roi_name if roi_name else f"ROI_{roi_index}"
    print(f"Preprocessing ROI {roi_index} ({identifier})")
    
    # Kiểm tra nếu ảnh rỗng
    if roi is None or roi.size == 0 or roi.shape[0] <= 5 or roi.shape[1] <= 5:
        print(f"ROI {identifier} quá nhỏ hoặc rỗng, bỏ qua")
        return None
    
    # Tạo thư mục để lưu ảnh ROI đã xử lý (nếu chưa tồn tại)
    processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_roi')
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    # Lưu ảnh ROI gốc
    base_filename = os.path.splitext(original_filename)[0]
    original_roi_filename = f"{base_filename}_{identifier}_original.png"
    original_roi_path = os.path.join(processed_folder, original_roi_filename)
    cv2.imwrite(original_roi_path, roi)
    
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    quality_info = check_image_quality(gray)
    print(f"Image quality for ROI {identifier}: {quality_info}")
    if not quality_info['is_good_quality']:
        print(f"Enhancing image quality for ROI {identifier} due to: {quality_info['issues']}")
        
        # Cải thiện chất lượng ảnh
        enhanced_gray = enhance_image_quality(gray, quality_info)
        
        # Lưu ảnh đã cải thiện
        enhanced_filename = f"{base_filename}_{identifier}_enhanced.png"
        enhanced_path = os.path.join(processed_folder, enhanced_filename)
        cv2.imwrite(enhanced_path, enhanced_gray)
        
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
    processed_filename = f"{base_filename}_{identifier}_processed1.png"
    processed_path = os.path.join(processed_folder, processed_filename)
    cv2.imwrite(processed_path, closing)
    # 5. Tìm contour
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Tính giới hạn trên và dưới cho mỗi contour
    contour_limits = []
    for cnt in contours:
        # Tính bounding rectangle cho contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Loại bỏ contour nếu chiều rộng <= 20 pixel hoặc chiều cao <= 30 pixel
        if w <= 10 or h <= 20:
            continue
        
        y_coords = [point[0][1] for point in cnt]  # Lấy tọa độ y của các điểm trong contour
        upper_limit = min(y_coords)
        lower_limit = max(y_coords)
        contour_limits.append((upper_limit, lower_limit, cnt))

    # 7. Đếm số lượng contour nằm trong giới hạn y của từng contour
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

    # 8. Gộp tất cả các contour trong vùng giới hạn y của contour tốt nhất
    if best_contour is not None and len(contour_limits) > 0:
        # Nếu chỉ có 1 contour thì sử dụng contour đó luôn mà không cần gộp
        if len(contour_limits) == 1:
            merged_contour = contour_limits[0][2]
        else:
            merged_contour = np.vstack([cnt for upper, lower, cnt in contour_limits if not (best_limits[1] < upper or best_limits[0] > lower)])
        
        # 9. Cắt (crop) vùng boundingRect của contour lớn nhất với padding
        x, y, w, h = cv2.boundingRect(merged_contour)
        
        # Mở rộng thêm 20 pixel xung quanh
        pad = 5
        x3, y3 = x1+x-pad, y1+y-pad
        x4, y4 = x1+x+w+pad, y1+y+h+pad

        # Lưu ý: Cắt từ ảnh đã xử lý (closing) để có dạng grayscale
        cropped_closing = image_aligned[y3:y4, x3:x4]

        gray = cv2.cvtColor(cropped_closing, cv2.COLOR_BGR2GRAY)

        # 10. Tiền xử lý (làm mượt, giảm nhiễu)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 11. Threshold (Otsu)
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = thresh_otsu

        # 12. Morphological Closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Tăng lên (5,5) thay vì (2,2)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Lưu ảnh đã xử lý
        processed_filename = f"{base_filename}_{identifier}_processed2.png"
        processed_path = os.path.join(processed_folder, processed_filename)
        cv2.imwrite(processed_path, closing)
        
        print(f"Saved processed ROI to: {processed_path}")
        return closing  # Trả về ảnh grayscale
    else:
        print(f"Không tìm thấy contour hợp lệ để cắt cho ROI {identifier}.")
        closing = cv2.blur(closing, (3, 3))
        # Trả về ảnh grayscale nếu không tìm thấy contour
        processed_filename = f"{base_filename}_{identifier}_processed.png"
        processed_path = os.path.join(processed_folder, processed_filename)
        cv2.imwrite(processed_path, closing)
        return closing  # Hoặc closing

def check_safe_zone(roi_processed, margin_percent=5.0):
    """
    Kiểm tra vùng an toàn trong ROI và trả về kết quả OCR
    """
    if roi_processed is None or roi_processed.size == 0:
        print("ROI is empty, cannot check safe zone")
        return None
    
    height, width = roi_processed.shape
    print(f"ROI dimensions: {width}x{height}")
    
    # Tính toán kích thước vùng an toàn
    margin_x = int(width * margin_percent / 100)
    margin_y = int(height * margin_percent / 100)
    
    # Tính toán tọa độ vùng an toàn
    safe_x1 = margin_x
    safe_y1 = margin_y
    safe_x2 = width - margin_x
    safe_y2 = height - margin_y
    
    print(f"Safe zone coordinates: ({safe_x1},{safe_y1},{safe_x2},{safe_y2})")
    
    # Kiểm tra nếu vùng an toàn quá nhỏ
    if safe_x2 <= safe_x1 or safe_y2 <= safe_y1:
        print("Safe zone is too small")
        return None
    
    # Cắt vùng an toàn
    safe_zone = roi_processed[safe_y1:safe_y2, safe_x1:safe_x2]
    
    # Đảm bảo vùng an toàn không rỗng
    if safe_zone is None or safe_zone.size == 0 or safe_zone.shape[0] < 5 or safe_zone.shape[1] < 5:
        print("Safe zone is too small or empty")
        return None
    
    try:
        # Tạo đối tượng reader cho OCR nếu chưa có
        if 'reader' not in globals():
            global reader
            reader = easyocr.Reader(['en'], gpu=False)
        
        # Sử dụng EasyOCR để nhận diện text trong vùng an toàn
        results = reader.readtext(safe_zone)
        
        if results and len(results) > 0:
            print(f"Found {len(results)} text regions in safe zone")
            for i, (bbox, text, conf) in enumerate(results):
                print(f"  Text {i+1}: '{text}' (confidence: {conf:.4f})")
        else:
            print("No text found in safe zone")
        
        return results
    except Exception as e:
        print(f"Error in safe zone OCR: {str(e)}")
        traceback.print_exc()
        return None

def is_named_roi_format(roi_list):
    """Kiểm tra xem danh sách ROI có phải là định dạng mới (có name và coordinates) hay không"""
    if not roi_list:
        return False
    
    first_item = roi_list[0]
    return isinstance(first_item, dict) and "name" in first_item and "coordinates" in first_item

# Thêm route mới cho /api/machines/<machine_code>
@app.route('/api/machines/<machine_code>', methods=['GET'])
def get_machine_screens(machine_code):
    """
    Lấy danh sách các màn hình (screens) cho một máy cụ thể
    
    Path Parameters:
    - machine_code: Mã máy (bắt buộc, ví dụ: F1, F41)
    """
    try:
        # Đọc file JSON chứa thông tin về máy và màn hình
        machine_screens_path = os.path.join(app.config['ROI_DATA_FOLDER'], 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Kiểm tra xem machine_code có tồn tại trong dữ liệu không
        if machine_code not in data['machines']:
            return jsonify({"error": f"Machine {machine_code} not found"}), 404
        
        # Lấy thông tin về các màn hình của máy này
        machine_info = data['machines'][machine_code]
        screens = machine_info['screens']
        
        # Đọc cấu hình decimal places
        decimal_config = get_decimal_places_config()
        
        # Thêm thông tin ROI cho mỗi màn hình (nếu có)
        for screen in screens:
            screen_id = screen['id']
            screen_name = screen.get('screen_id', '')
            roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_id)
            
            if roi_coordinates:
                screen['roi_count'] = len(roi_coordinates)
                
                # Kiểm tra cấu hình decimal places theo cấu trúc mới
                # Trong decimal_places.json, cấu trúc là:
                # "F1": { "Faults": { "Tgian_chu_ki": 5, "Vtri_khuon": 2 } }
                has_decimal_config = (machine_code in decimal_config and 
                                     screen_name in decimal_config[machine_code])
                
                screen['has_decimal_config'] = has_decimal_config
            else:
                screen['roi_count'] = 0
                screen['has_decimal_config'] = False
        
        return jsonify({
            "machine_code": machine_code,
            "machine_name": machine_info['name'],
            "screens_count": len(screens),
            "screens": screens
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to get screens: {str(e)}"}), 500

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
    
    # 1. Kiểm tra độ sắc nét (blurriness) bằng biến đổi Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result['blurriness'] = laplacian_var
    
    # Ngưỡng blurriness: thấp đồng nghĩa với ảnh mờ
    blur_threshold = 50.0  # Ngưỡng này có thể điều chỉnh
    if laplacian_var < blur_threshold:
        result['is_good_quality'] = False
        result['issues'].append('blurry')
        return result
    # 2. Kiểm tra độ sáng và độ tương phản
    # Tính histogram của ảnh
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / gray.size
    
    # Tính độ sáng trung bình
    brightness = np.mean(gray)
    result['brightness'] = brightness
    
    # Tính độ tương phản bằng độ lệch chuẩn
    contrast = np.std(gray)
    result['contrast'] = contrast
    
    # Kiểm tra độ sáng quá thấp hoặc quá cao
    if brightness > 220:
        result['is_good_quality'] = False
        result['issues'].append('too_bright')
        return result
    # Kiểm tra độ tương phản quá thấp
    if contrast < 20:
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
    if brightness > 200 and peak_ratio > 0.005:
        result['has_moire'] = True
        result['is_good_quality'] = False
        result['issues'].append('moire_pattern')
    
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
        denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # Tạo hiệu ứng Unsharp Mask:
        # Công thức: sharpened = (1 + amount)*img - amount*blurred
        amount = 0.5  # điều chỉnh mức tăng nét (có thể từ 0.3 đến 1.0)
        blurred = cv2.GaussianBlur(denoised, (9, 9), 10)
        sharpened = cv2.addWeighted(denoised, 1 + amount, blurred, -amount, 0)
        enhanced = sharpened
    # 1. Xử lý khi ảnh bị mờ
    if 'blurry' in quality_info['issues']:
        # Áp dụng bộ lọc làm sắc nét (sharpening filter)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
    
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

if __name__ == '__main__':
    print("DEBUG INFO:")
    print(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
    print(f"API Routes configured:")
    print("- / (GET): Test endpoint")
    print("- /debug (GET): Debug information")
    print("- /api/images (GET): List all images")
    print("- /api/images (POST): Upload image")
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
    print("Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=True) 