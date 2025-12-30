import os
import sys
import io
import warnings
import contextlib
import cv2
import numpy as np
from math import sqrt, atan2, degrees
from PIL import Image, ImageEnhance
import json
import re

# ============================================================
# CẤU HÌNH MÔI TRƯỜNG - PHẢI ĐẶT TRƯỚC KHI IMPORT PADDLEOCR
# ============================================================

# Tắt kiểm tra kết nối đến model hosters (tiết kiệm vài giây)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Tắt các log không cần thiết của Paddle  
os.environ['GLOG_minloglevel'] = '3'      # Chỉ hiện FATAL
os.environ['FLAGS_call_stack_level'] = '0'
os.environ['PADDLE_PDX_SILENT_MODE'] = '1'  # Tắt log của PaddleX

# Tắt warnings
warnings.filterwarnings('ignore')

# ============================================================
# HELPER: Suppress noisy output từ PaddleOCR
# ============================================================
@contextlib.contextmanager
def suppress_output():
    """Tạm thời tắt stdout và stderr để ẩn thông báo không cần thiết"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import PaddleOCR SAU KHI đã set env vars (với suppress output)
with suppress_output():
    from paddleocr import PaddleOCR
import tkinter as tk
from tkinter import filedialog
import time

# ============================================================
# BIẾN GLOBAL
# ============================================================
_ocr_instance = None
ROI_INFO_PATH = "roi_data/roi_info.json"
MACHINE_SCREENS_PATH = "roi_data/machine_screens.json"
IOU_THRESHOLD = 0.10  # Ngưỡng IoU 10%

# ============================================================
# MACHINE SCREENS FUNCTIONS
# ============================================================

def load_machine_screens():
    """Đọc file machine_screens.json"""
    if not os.path.exists(MACHINE_SCREENS_PATH):
        print(f"   ⚠ Không tìm thấy file: {MACHINE_SCREENS_PATH}")
        return None
    
    with open(MACHINE_SCREENS_PATH, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def select_area(machine_screens):
    """Chọn khu vực (F1, F4, ...)"""
    if not machine_screens or 'areas' not in machine_screens:
        print("   ⚠ Không tìm thấy thông tin khu vực trong machine_screens.json")
        return None
    
    areas = machine_screens['areas']
    area_list = list(areas.keys())
    
    print("\n" + "="*50)
    print("📍 CHỌN KHU VỰC")
    print("="*50)
    for i, area_code in enumerate(area_list, 1):
        area_name = areas[area_code].get('name', area_code)
        machine_count = len(areas[area_code].get('machines', {}))
        print(f"   {i}. {area_code} - {area_name} ({machine_count} máy)")
    print("   0. Thoát")
    print("-"*50)
    
    while True:
        try:
            choice = input("Nhập số thứ tự khu vực (0 để thoát): ").strip()
            if choice == '0' or choice == '':
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(area_list):
                selected_area = area_list[index]
                area_name = areas[selected_area].get('name', selected_area)
                print(f"   ✓ Đã chọn: {selected_area} - {area_name}")
                return selected_area
            else:
                print("   ⚠ Lựa chọn không hợp lệ. Vui lòng thử lại.")
        except ValueError:
            print("   ⚠ Vui lòng nhập số.")

def select_machine(machine_screens, area):
    """Chọn mã máy trong khu vực đã chọn"""
    if not machine_screens or 'areas' not in machine_screens:
        return None
    
    if area not in machine_screens['areas']:
        print(f"   ⚠ Không tìm thấy khu vực: {area}")
        return None
    
    machines = machine_screens['areas'][area].get('machines', {})
    if not machines:
        print(f"   ⚠ Không có máy nào trong khu vực {area}")
        return None
    
    machine_list = list(machines.keys())
    
    print("\n" + "="*50)
    print(f"🔧 CHỌN MÃ MÁY (Khu vực {area})")
    print("="*50)
    for i, machine_code in enumerate(machine_list, 1):
        machine_info = machines[machine_code]
        machine_name = machine_info.get('name', machine_code)
        machine_type = machine_info.get('type', 'N/A')
        screen_count = len(machine_info.get('screens', []))
        print(f"   {i}. {machine_code} - {machine_name} (Type: {machine_type}, {screen_count} màn hình)")
    print("   0. Quay lại chọn khu vực")
    print("-"*50)
    
    while True:
        try:
            choice = input("Nhập số thứ tự máy (0 để quay lại): ").strip()
            if choice == '0' or choice == '':
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(machine_list):
                selected_machine = machine_list[index]
                machine_name = machines[selected_machine].get('name', selected_machine)
                print(f"   ✓ Đã chọn: {selected_machine} - {machine_name}")
                return selected_machine
            else:
                print("   ⚠ Lựa chọn không hợp lệ. Vui lòng thử lại.")
        except ValueError:
            print("   ⚠ Vui lòng nhập số.")

# ============================================================
# ROI MATCHING & IoU FUNCTIONS
# ============================================================

def load_roi_info():
    """Đọc file roi_info.json"""
    if not os.path.exists(ROI_INFO_PATH):
        print(f"   ⚠ Không tìm thấy file: {ROI_INFO_PATH}")
        return None
    
    with open(ROI_INFO_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_text(text):
    """
    Chuẩn hóa text để so sánh fuzzy matching
    - Loại bỏ dấu : và khoảng trắng thừa
    - Chuyển uppercase
    - Chuẩn hóa định dạng "ST14 - LEAK" → "ST14-LEAK" (loại bỏ space quanh dấu gạch ngang)
    """
    if not text:
        return ""
    # Chuyển uppercase và loại bỏ khoảng trắng đầu/cuối
    normalized = text.strip().upper()
    # Loại bỏ dấu : ở cuối
    normalized = normalized.rstrip(':')
    # Loại bỏ khoảng trắng xung quanh dấu gạch ngang (ST14 - LEAK → ST14-LEAK)
    normalized = re.sub(r'\s*-\s*', '-', normalized)
    # Gộp nhiều khoảng trắng thành 1
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def fuzzy_match(text1, text2, threshold=0.75):
    """
    So sánh fuzzy giữa 2 chuỗi, trả về True nếu độ tương đồng >= threshold
    Sử dụng thuật toán Levenshtein distance
    
    LƯU Ý: threshold = 0.75 để tránh false positives như:
    - "PRESENCE CHECK" matching với "ST04-PARTS LOADED PRESENCE CHECK"
    """
    s1 = normalize_text(text1)
    s2 = normalize_text(text2)
    
    if not s1 or not s2:
        return False
    
    len1, len2 = len(s1), len(s2)
    min_len = min(len1, len2)
    max_len = max(len1, len2)
    
    # Kiểm tra một chuỗi chứa chuỗi kia - CHỈ cho phép nếu độ dài gần nhau
    # Tránh trường hợp "PRESENCE CHECK" match với "ST04-PARTS LOADED PRESENCE CHECK"
    if s1 in s2 or s2 in s1:
        # Chỉ chấp nhận nếu chuỗi ngắn hơn chiếm ít nhất 70% chuỗi dài hơn
        length_ratio = min_len / max_len
        if length_ratio >= 0.7:
            return True
        # Nếu không, tiếp tục kiểm tra bằng Levenshtein
    
    # Nếu độ dài khác nhau quá nhiều (>40%), không khớp
    if abs(len1 - len2) > max_len * 0.4:
        return False
    
    # Tính Levenshtein distance
    # Tạo ma trận
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

def find_matching_screen(ocr_data, roi_info, selected_area=None, selected_machine=None, debug=True):
    """
    Tìm screen và sub_page phù hợp nhất dựa trên Special_rois
    
    Hỗ trợ cấu trúc mới:
    machines > area (F1) > machine_code (IE-F1-CWA01) > screens > screen_name > sub_pages > page_num
    
    Args:
        ocr_data: Dữ liệu OCR đã trích xuất
        roi_info: Dữ liệu từ roi_info.json
        selected_area: Khu vực đã chọn (F1, F4, ...) - nếu None sẽ duyệt tất cả
        selected_machine: Mã máy đã chọn (IE-F1-CWA01, ...) - nếu None sẽ duyệt tất cả trong area
        debug: Hiển thị thông tin debug
    
    Trả về: (area, machine_code, screen_name, sub_page, sub_page_data, match_count, match_percentage)
    """
    if not roi_info or 'machines' not in roi_info:
        return None, None, None, None, None, 0, 0
    
    # Lấy tất cả text từ OCR (đã chuẩn hóa)
    ocr_texts = [normalize_text(item['text']) for item in ocr_data]
    
    if debug:
        print(f"\n   📝 OCR detected {len(ocr_texts)} text items")
        if selected_area and selected_machine:
            print(f"   🎯 Lọc theo: {selected_area}/{selected_machine}")
    
    best_match = None
    best_match_count = 0
    best_match_percentage = 0
    
    # Lưu tất cả kết quả matching để debug
    all_matches = []
    
    # Duyệt qua cấu trúc: machines > area > machine_code > screens
    for area, area_data in roi_info['machines'].items():
        # Nếu đã chọn area, chỉ duyệt area đó
        if selected_area and area != selected_area:
            continue
        
        # Kiểm tra xem area_data có phải là dict chứa machine_codes không
        if not isinstance(area_data, dict):
            continue
        
        for machine_code, machine_data in area_data.items():
            # Nếu đã chọn machine, chỉ duyệt machine đó
            if selected_machine and machine_code != selected_machine:
                continue
            
            # Bỏ qua nếu không phải dict hoặc không có screens
            if not isinstance(machine_data, dict) or 'screens' not in machine_data:
                continue
            
            for screen_name, screen_data in machine_data['screens'].items():
                # Kiểm tra cấu trúc với sub_pages
                if 'sub_pages' in screen_data:
                    # Duyệt qua từng sub_page
                    for sub_page, sub_page_data in screen_data['sub_pages'].items():
                        if 'Special_rois' not in sub_page_data:
                            continue
                        
                        special_rois = sub_page_data['Special_rois']
                        match_count = 0
                        matched_rois = []
                        
                        # Đếm số lượng Special_rois khớp với OCR results
                        for special_roi in special_rois:
                            special_roi_normalized = normalize_text(special_roi)
                            
                            for ocr_text in ocr_texts:
                                if fuzzy_match(special_roi_normalized, ocr_text):
                                    match_count += 1
                                    matched_rois.append(special_roi)
                                    break  # Mỗi Special_roi chỉ đếm 1 lần
                        
                        # Tính phần trăm khớp
                        if len(special_rois) > 0:
                            match_percentage = (match_count / len(special_rois)) * 100
                        else:
                            match_percentage = 0
                        
                        # Lưu kết quả để debug
                        all_matches.append({
                            'area': area,
                            'machine': machine_code,
                            'screen': screen_name,
                            'sub_page': sub_page,
                            'special_rois': special_rois,
                            'match_count': match_count,
                            'match_percentage': match_percentage,
                            'matched_rois': matched_rois
                        })
                        
                        # Cập nhật best match
                        if match_count > best_match_count or (match_count == best_match_count and match_percentage > best_match_percentage):
                            best_match_count = match_count
                            best_match_percentage = match_percentage
                            best_match = (area, machine_code, screen_name, sub_page, sub_page_data)
                else:
                    # Cấu trúc cũ (không có sub_pages) - tương thích ngược
                    if 'Special_rois' not in screen_data:
                        continue
                    
                    special_rois = screen_data['Special_rois']
                    match_count = 0
                    matched_rois = []
                    
                    # Đếm số lượng Special_rois khớp với OCR results
                    for special_roi in special_rois:
                        special_roi_normalized = normalize_text(special_roi)
                        
                        for ocr_text in ocr_texts:
                            if fuzzy_match(special_roi_normalized, ocr_text):
                                match_count += 1
                                matched_rois.append(special_roi)
                                break
                    
                    # Tính phần trăm khớp
                    if len(special_rois) > 0:
                        match_percentage = (match_count / len(special_rois)) * 100
                    else:
                        match_percentage = 0
                    
                    # Lưu kết quả để debug
                    all_matches.append({
                        'area': area,
                        'machine': machine_code,
                        'screen': screen_name,
                        'sub_page': '1',
                        'special_rois': special_rois,
                        'match_count': match_count,
                        'match_percentage': match_percentage,
                        'matched_rois': matched_rois
                    })
                    
                    # Cập nhật best match (sub_page = "1" cho cấu trúc cũ)
                    if match_count > best_match_count or (match_count == best_match_count and match_percentage > best_match_percentage):
                        best_match_count = match_count
                        best_match_percentage = match_percentage
                        best_match = (area, machine_code, screen_name, "1", screen_data)
    
    # In debug info về tất cả matches
    if debug and all_matches:
        print(f"\n   🔍 Screen matching results:")
        for m in all_matches:
            status = "✓" if m['match_count'] > 0 else "✗"
            print(f"      {status} {m['area']}/{m['machine']}/{m['screen']}/sub-page {m['sub_page']}: "
                  f"{m['match_count']}/{len(m['special_rois'])} matches ({m['match_percentage']:.0f}%)")
            if m['matched_rois']:
                print(f"         Matched: {m['matched_rois']}")
    
    if best_match:
        # Trả về: (area, machine_code, screen_name, sub_page, sub_page_data, match_count, match_percentage)
        return best_match[0], best_match[1], best_match[2], best_match[3], best_match[4], best_match_count, best_match_percentage
    
    return None, None, None, None, None, 0, 0

def polygon_to_normalized_bbox(polygon, img_width, img_height):
    """
    Chuyển đổi polygon từ PaddleOCR sang normalized bounding box [x1, y1, x2, y2]
    polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (4 góc của text box)
    """
    if not polygon or len(polygon) < 4:
        return None
    
    # Lấy tọa độ min/max
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    # Normalize
    norm_x1 = x_min / img_width
    norm_y1 = y_min / img_height
    norm_x2 = x_max / img_width
    norm_y2 = y_max / img_height
    
    return [norm_x1, norm_y1, norm_x2, norm_y2]

def calculate_iou(box1, box2):
    """
    Tính IoU (Intersection over Union) giữa 2 bounding boxes
    box format: [x1, y1, x2, y2] (normalized)
    """
    # Tính tọa độ intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Tính diện tích intersection
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    
    # Tính diện tích của mỗi box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Tính union
    union_area = box1_area + box2_area - intersection_area
    
    # Tính IoU
    if union_area <= 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou

def filter_ocr_by_roi(ocr_data, sub_page_data, img_width, img_height):
    """
    Lọc kết quả OCR dựa trên IoU với các ROIs của sub_page
    Hỗ trợ cấu trúc mới với sub_pages
    Trả về: list các kết quả OCR đã được lọc với thông tin ROI tương ứng
    """
    if not sub_page_data:
        return []
    
    # Lấy Rois từ sub_page_data (hỗ trợ cả cấu trúc cũ và mới)
    rois = sub_page_data.get('Rois', [])
    if not rois:
        return []
    filtered_results = []
    
    for ocr_item in ocr_data:
        polygon = ocr_item.get('bbox', [])
        if not polygon:
            continue
        
        # Chuyển đổi polygon sang normalized bbox
        ocr_bbox = polygon_to_normalized_bbox(polygon, img_width, img_height)
        if not ocr_bbox:
            continue
        
        # Tìm ROI có IoU cao nhất
        best_iou = 0
        best_roi_name = None
        best_roi_coords = None
        
        for roi in rois:
            roi_coords = roi.get('coordinates', [])
            if len(roi_coords) != 4:
                continue
            
            # roi_coords format: [x1, y1, x2, y2] (normalized)
            iou = calculate_iou(ocr_bbox, roi_coords)
            
            if iou > best_iou:
                best_iou = iou
                best_roi_name = roi.get('name', 'Unknown')
                best_roi_coords = roi_coords
        
        # Chỉ giữ lại nếu IoU > threshold
        if best_iou >= IOU_THRESHOLD:
            filtered_results.append({
                'text': ocr_item['text'],
                'confidence': ocr_item['confidence'],
                'bbox': polygon,
                'normalized_bbox': ocr_bbox,
                'matched_roi': best_roi_name,
                'roi_coords': best_roi_coords,
                'iou': best_iou
            })
    
    return filtered_results

# ============================================================
# HMI DETECTION FUNCTIONS (từ hmi_image_detector.py)
# ============================================================

def enhance_image_for_hmi(image):
    """Cải thiện chất lượng ảnh trước khi phát hiện cạnh"""
    # Chuyển từ OpenCV (BGR) sang PIL (RGB) để áp dụng ImageEnhance
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Tăng độ tương phản với PIL
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_pil = enhancer.enhance(2)  # Tăng độ tương phản lên 50%
    
    # Chuyển lại về định dạng OpenCV
    enhanced_image = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
    
    # Tiếp tục quy trình xử lý ảnh như trước
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    # Tăng clip limit để cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(11, 11))
    enhanced = clahe.apply(gray)
    
    # Tăng độ tương phản
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=0)
    
    # Làm mịn ảnh với kernel nhỏ hơn để giữ nguyên cạnh sắc nét hơn
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred, enhanced

def adaptive_edge_detection(image):
    """Phát hiện cạnh với nhiều phương pháp và kết hợp kết quả"""
    median_val = np.median(image)
    # Giảm ngưỡng để tăng độ nhạy cảm phát hiện cạnh
    lower = int(max(0, (1.0 - 0.33) * median_val))
    upper = int(min(255, (1.0 + 0.33) * median_val))
    canny_edges = cv2.Canny(image, lower, upper)
    
    # Sử dụng kernel lớn hơn cho bộ lọc Sobel
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Giảm ngưỡng sobel để bắt được nhiều cạnh hơn
    _, sobel_edges = cv2.threshold(sobel_edges, 80, 255, cv2.THRESH_BINARY)
    
    # Kết hợp cả hai phương pháp phát hiện cạnh
    combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
    
    # Tăng số lần giãn nở để kết nối các cạnh bị đứt đoạn
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=2)
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
    min_length = max(min_length, int(min_dimension * 0.02))
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if length < min_length:
            continue
        
        # Tính góc của đường thẳng
        if x2 != x1:
            angle = degrees(atan2(y2-y1, x2-x1))
        else:
            angle = 90  # Đường dọc
        
        # Mở rộng phạm vi phân loại đường ngang/dọc
        if abs(angle) < 40 or abs(angle) > 140:  # Đường ngang
            all_h_lines.append([x1, y1, x2, y2, angle, length])
        elif abs(angle - 90) < 40 or abs(angle + 90) < 40:  # Đường dọc
            all_v_lines.append([x1, y1, x2, y2, angle, length])
    
    # Sắp xếp theo độ dài
    all_h_lines.sort(key=lambda x: x[5], reverse=True)
    all_v_lines.sort(key=lambda x: x[5], reverse=True)
    
    # Đảm bảo có đủ số lượng đường ngang và dọc tối thiểu
    min_lines = min(4, len(all_h_lines))
    horizontal_lines = [line[:5] for line in all_h_lines[:max(min_lines, max_lines_per_direction)]]
    
    min_lines = min(4, len(all_v_lines))
    vertical_lines = [line[:5] for line in all_v_lines[:max(min_lines, max_lines_per_direction)]]
    
    return horizontal_lines, vertical_lines

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
        if abs(m) > 1e-10:
            x_top = (0 - b) / m
            if 0 <= x_top < width:
                intersections.append((int(x_top), 0))
            
        # Giao với cạnh dưới (y=height-1)
        if abs(m) > 1e-10:
            x_bottom = ((height - 1) - b) / m
            if 0 <= x_bottom < width:
                intersections.append((int(x_bottom), height - 1))
        
        # Nếu có đủ hai giao điểm, tạo đường kéo dài
        if len(intersections) >= 2:
            p1, p2 = intersections[:2]
            extended_lines.append([p1[0], p1[1], p2[0], p2[1], angle])
    
    return extended_lines

def find_intersections(horizontal_lines, vertical_lines, max_intersections=200):
    """Tìm giao điểm của các đường ngang và dọc, giới hạn số lượng giao điểm"""
    intersections = []
    
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            if len(intersections) >= max_intersections:
                break
                
            x1_h, y1_h, x2_h, y2_h, _ = h_line
            x1_v, y1_v, x2_v, y2_v, _ = v_line
            
            # Xử lý trường hợp đặc biệt của đường ngang và dọc
            if abs(y1_h - y2_h) < 5 and abs(x1_v - x2_v) < 5:
                intersections.append((int(x1_v), int(y1_h)))
                continue
            
            # Sử dụng phương pháp đơn giản hơn để tìm giao điểm
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
                
                # Trường hợp tổng quát
                denom = (y2_v - y1_v) * (x2_h - x1_h) - (x2_v - x1_v) * (y2_h - y1_h)
                
                if abs(denom) < 1e-10:
                    continue
                
                ua = ((x2_v - x1_v) * (y1_h - y1_v) - (y2_v - y1_v) * (x1_h - x1_v)) / denom
                
                x_intersect = x1_h + ua * (x2_h - x1_h)
                y_intersect = y1_h + ua * (y2_h - y1_h)
                
                if (min(x1_h, x2_h) - 10 <= x_intersect <= max(x1_h, x2_h) + 10 and
                    min(y1_v, y2_v) - 10 <= y_intersect <= max(y1_v, y2_v) + 10):
                    intersections.append((int(x_intersect), int(y_intersect)))
            
            except (ValueError, OverflowError, ZeroDivisionError):
                continue
        
        if len(intersections) >= max_intersections:
            break
    
    return intersections

def find_largest_rectangle(intersections, img_shape):
    """Tìm hình chữ nhật lớn nhất từ các giao điểm"""
    if len(intersections) < 4:
        return None
    
    # Tìm các điểm biên
    left_point = min(intersections, key=lambda p: p[0])
    right_point = max(intersections, key=lambda p: p[0])
    top_point = min(intersections, key=lambda p: p[1])
    bottom_point = max(intersections, key=lambda p: p[1])
    
    # Tính toán các góc của hình chữ nhật lớn nhất
    top_left = (left_point[0], top_point[1])
    top_right = (right_point[0], top_point[1])
    bottom_left = (left_point[0], bottom_point[1])
    bottom_right = (right_point[0], bottom_point[1])
    
    # Kiểm tra xem các góc có nằm gần các giao điểm không
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
    
    # Tính diện tích
    width = refined_top_right[0] - refined_top_left[0]
    height = refined_bottom_left[1] - refined_top_left[1]
    area = width * height
    
    # Kiểm tra kích thước hợp lý
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

def find_rectangle_from_classified_lines(horizontal_lines, vertical_lines, img_shape):
    """Tìm hình chữ nhật từ các đường đã phân loại ngang và dọc"""
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None
    
    # Tìm đường ngang trên cùng và dưới cùng
    top_line = min(horizontal_lines, key=lambda line: min(line[1], line[3]))
    bottom_line = max(horizontal_lines, key=lambda line: max(line[1], line[3]))
    
    # Tìm đường dọc trái cùng và phải cùng
    left_line = min(vertical_lines, key=lambda line: min(line[0], line[2]))
    right_line = max(vertical_lines, key=lambda line: max(line[0], line[2]))
    
    # Tính toán các tọa độ y cho đường ngang trên và dưới
    top_y = min(top_line[1], top_line[3])
    bottom_y = max(bottom_line[1], bottom_line[3])
    
    # Tính toán các tọa độ x cho đường dọc trái và phải
    left_x = min(left_line[0], left_line[2])
    right_x = max(right_line[0], right_line[2])
    
    # Kiểm tra ngang
    top_left_x = max(min(top_line[0], top_line[2]), left_x)
    top_right_x = min(max(top_line[0], top_line[2]), right_x)
    bottom_left_x = max(min(bottom_line[0], bottom_line[2]), left_x)
    bottom_right_x = min(max(bottom_line[0], bottom_line[2]), right_x)
    
    # Kiểm tra dọc
    left_top_y = max(min(left_line[1], left_line[3]), top_y)
    left_bottom_y = min(max(left_line[1], left_line[3]), bottom_y)
    right_top_y = max(min(right_line[1], right_line[3]), top_y)
    right_bottom_y = min(max(right_line[1], right_line[3]), bottom_y)
    
    if (top_right_x - top_left_x < 10 or bottom_right_x - bottom_left_x < 10 or
        left_bottom_y - left_top_y < 10 or right_bottom_y - right_top_y < 10):
        return None
    
    # Kiểm tra kích thước của hình chữ nhật
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

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect

def extract_content_region(img, save_folder, base_name):
    """Trích xuất vùng nội dung (không phải vùng đen xung quanh màn hình)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    enhanced_contrast = cv2.convertScaleAbs(gray, alpha=1.3, beta=5)
    
    enhanced_path = f"{save_folder}/8b_content_enhanced_{base_name}.jpg"
    cv2.imwrite(enhanced_path, enhanced_contrast)
    
    blurred = cv2.GaussianBlur(enhanced_contrast, (3, 3), 0)
    
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobel_x, sobel_y)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    _, gradient_thresh = cv2.threshold(gradient_mag, 20, 255, cv2.THRESH_BINARY)
    
    gradient_before_path = f"{save_folder}/8b_content_gradient_before_{base_name}.jpg"
    cv2.imwrite(gradient_before_path, gradient_thresh)
    
    vertical_kernel = np.ones((11, 3), np.uint8)
    gradient_dilated = cv2.dilate(gradient_thresh, vertical_kernel, iterations=3)
    
    horizontal_kernel = np.ones((3, 9), np.uint8)
    gradient_dilated = cv2.dilate(gradient_dilated, horizontal_kernel, iterations=2)
    
    gradient_path = f"{save_folder}/8b_content_gradient_{base_name}.jpg"
    cv2.imwrite(gradient_path, gradient_dilated)
    
    kernel = np.ones((5, 5), np.uint8)
    gradient_final = cv2.morphologyEx(gradient_dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    gradient_final_path = f"{save_folder}/8b_content_gradient_final_{base_name}.jpg"
    cv2.imwrite(gradient_final_path, gradient_final)
    
    contours, _ = cv2.findContours(gradient_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        enhanced_for_threshold = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        _, thresh = cv2.threshold(enhanced_for_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_path = f"{save_folder}/8b_content_otsu_thresh_{base_name}.jpg"
        cv2.imwrite(thresh_path, thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = img.shape[0] * img.shape[1] * 0.005
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    all_contours_img = img.copy()
    cv2.drawContours(all_contours_img, large_contours, -1, (0, 255, 0), 2)
    all_contours_path = f"{save_folder}/8b_all_large_contours_{base_name}.jpg"
    cv2.imwrite(all_contours_path, all_contours_img)
    
    mask = np.zeros_like(gray)
    if large_contours:
        largest_contour = max(large_contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    else:
        mask.fill(255)
    
    mask_path = f"{save_folder}/8b_content_final_mask_{base_name}.jpg"
    cv2.imwrite(mask_path, mask)
    
    contour_img = img.copy()
    if large_contours:
        cv2.drawContours(contour_img, [largest_contour], 0, (0, 255, 0), 2)
    contour_path = f"{save_folder}/8b_content_largest_contour_{base_name}.jpg"
    cv2.imwrite(contour_path, contour_img)
    
    return mask, large_contours[0] if large_contours else None

def fine_tune_hmi_screen(image, roi_coords, save_folder, base_name):
    """Tinh chỉnh vùng màn hình HMI đã phát hiện"""
    x_min, y_min, x_max, y_max = roi_coords
    roi = image[y_min:y_max, x_min:x_max]
    
    roi_original_path = f"{save_folder}/8b_roi_original_{base_name}.jpg"
    cv2.imwrite(roi_original_path, roi)
    
    content_mask, largest_contour = extract_content_region(roi, save_folder, base_name)
    
    if largest_contour is None:
        return roi, roi_coords
    
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < 0.1 * roi.shape[0] * roi.shape[1]:
        return roi, roi_coords
    
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    roi_approx = roi.copy()
    cv2.drawContours(roi_approx, [approx], 0, (0, 0, 255), 2)
    approx_path = f"{save_folder}/8d_roi_approx_{base_name}.jpg"
    cv2.imwrite(approx_path, roi_approx)
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = np.array(box, dtype=np.int32)
        
        roi_rect = roi.copy()
        cv2.drawContours(roi_rect, [approx], 0, (255, 0, 0), 2)
        rect_path = f"{save_folder}/8e_roi_adjusted_rect_{base_name}.jpg"
        cv2.imwrite(rect_path, roi_rect)
    
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
    
    roi_points = roi.copy()
    for i, point in enumerate(src_points):
        cv2.circle(roi_points, tuple(point.astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(roi_points, str(i), tuple(point.astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    points_path = f"{save_folder}/8f_roi_source_points_{base_name}.jpg"
    cv2.imwrite(points_path, roi_points)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(roi, M, (max_width, max_height))
    
    warped_path = f"{save_folder}/8g_roi_warped_{base_name}.jpg"
    cv2.imwrite(warped_path, warped)
    
    new_roi_coords = (x_min, y_min, x_min + warped.shape[1], y_min + warped.shape[0])
    
    return warped, new_roi_coords

def detect_hmi_screen(image_path, save_folder):
    """Phát hiện và trích xuất màn hình HMI từ ảnh"""
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Lấy tên cơ sở của file ảnh
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print("   Đang xử lý ảnh để phát hiện màn hình HMI...")
    
    # Lưu ảnh gốc
    original_path = f"{save_folder}/1_original_{base_name}.jpg"
    cv2.imwrite(original_path, image)
    
    # Tạo bản sao để vẽ kết quả
    result_image = image.copy()
    
    # Bước 1: Tăng cường chất lượng ảnh
    enhanced_img, enhanced_clahe = enhance_image_for_hmi(image)
    enhanced_path = f"{save_folder}/2_enhanced_{base_name}.jpg"
    cv2.imwrite(enhanced_path, enhanced_img)
    
    enhanced_clahe_path = f"{save_folder}/2b_enhanced_clahe_{base_name}.jpg"
    cv2.imwrite(enhanced_clahe_path, enhanced_clahe)
    
    # Bước 2: Phát hiện cạnh
    canny_edges, sobel_edges, edges = adaptive_edge_detection(enhanced_clahe)
    
    canny_path = f"{save_folder}/3a_canny_edges_{base_name}.jpg"
    cv2.imwrite(canny_path, canny_edges)
    
    sobel_path = f"{save_folder}/3b_sobel_edges_{base_name}.jpg"
    cv2.imwrite(sobel_path, sobel_edges)
    
    edges_path = f"{save_folder}/3c_combined_edges_{base_name}.jpg"
    cv2.imwrite(edges_path, edges)
    
    # Bước 3: Tìm và lọc contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_contours_image = image.copy()
    cv2.drawContours(all_contours_image, contours, -1, (0, 255, 0), 2)
    all_contours_path = f"{save_folder}/4a_all_contours_{base_name}.jpg"
    cv2.imwrite(all_contours_path, all_contours_image)
    
    min_contour_area = image.shape[0] * image.shape[1] * 0.001
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    large_contours_image = image.copy()
    cv2.drawContours(large_contours_image, large_contours, -1, (0, 255, 0), 2)
    large_contours_path = f"{save_folder}/4b_large_contours_{base_name}.jpg"
    cv2.imwrite(large_contours_path, large_contours_image)
    
    contour_mask = np.zeros_like(edges)
    cv2.drawContours(contour_mask, large_contours, -1, 255, 2)
    contour_mask_path = f"{save_folder}/4c_contour_mask_{base_name}.jpg"
    cv2.imwrite(contour_mask_path, contour_mask)
    
    # Bước 4: Phát hiện đường thẳng
    lines = cv2.HoughLinesP(contour_mask, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=30)
    
    if lines is None or len(lines) < 2:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=10, maxLineGap=40)
        
        if lines is None or len(lines) < 2:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=50)
    
    if lines is None:
        print("   ⚠ Không tìm thấy đường thẳng trong ảnh.")
        return None
    
    all_lines_image = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(all_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    all_lines_path = f"{save_folder}/5a_all_lines_{base_name}.jpg"
    cv2.imwrite(all_lines_path, all_lines_image)
    
    # Bước 5: Phân loại đường ngang/dọc
    height, width = image.shape[:2]
    horizontal_lines, vertical_lines = process_lines(lines, image.shape, min_length=20)
    
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        print("   ⚠ Không tìm thấy đủ đường ngang và dọc.")
        result_path = f"{save_folder}/9_result_{base_name}.jpg"
        cv2.imwrite(result_path, result_image)
        return None
    
    # Thử tìm hình chữ nhật từ các đường đã phân loại
    largest_rectangle = find_rectangle_from_classified_lines(horizontal_lines, vertical_lines, image.shape)
    
    if largest_rectangle is not None:
        direct_rectangle_image = image.copy()
        pts = np.array(largest_rectangle[:4])
        cv2.polylines(direct_rectangle_image, [pts], True, (255, 255, 0), 2)
        direct_rectangle_path = f"{save_folder}/5c_direct_rectangle_{base_name}.jpg"
        cv2.imwrite(direct_rectangle_path, direct_rectangle_image)
    else:
        # Nếu không tìm được, tiếp tục với quy trình thông thường
        extended_h_lines = extend_lines(horizontal_lines, width, height)
        extended_v_lines = extend_lines(vertical_lines, width, height)
        
        intersections = find_intersections(extended_h_lines, extended_v_lines)
        
        if len(intersections) < 4:
            print("   ⚠ Không tìm thấy đủ giao điểm để tạo hình chữ nhật.")
            result_path = f"{save_folder}/9_result_{base_name}.jpg"
            cv2.imwrite(result_path, result_image)
            return None
        
        largest_rectangle = find_largest_rectangle(intersections, image.shape)
        
        if largest_rectangle is None:
            print("   ⚠ Không tìm thấy hình chữ nhật phù hợp.")
            result_path = f"{save_folder}/9_result_{base_name}.jpg"
            cv2.imwrite(result_path, result_image)
            return None
    
    # Xác định vùng HMI từ hình chữ nhật lớn nhất
    top_left, top_right, bottom_right, bottom_left, _ = largest_rectangle
    
    x_min = min(top_left[0], bottom_left[0])
    y_min = min(top_left[1], top_right[1])
    x_max = max(top_right[0], bottom_right[0])
    y_max = max(bottom_left[1], bottom_right[1])
    
    # Kiểm tra biên
    if x_min < 0: x_min = 0
    if y_min < 0: y_min = 0
    if x_max >= image.shape[1]: x_max = image.shape[1] - 1
    if y_max >= image.shape[0]: y_max = image.shape[0] - 1
    
    if x_max > x_min and y_max > y_min:
        roi_coords = (x_min, y_min, x_max, y_max)
        
        # Vẽ hình chữ nhật lên ảnh kết quả
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Cắt và lưu vùng HMI
        roi = image[y_min:y_max, x_min:x_max]
        roi_path = f"{save_folder}/8b_roi_{base_name}.jpg"
        cv2.imwrite(roi_path, roi)
        
        # Tinh chỉnh và trải phẳng vùng HMI
        warped_roi, refined_coords = fine_tune_hmi_screen(image, roi_coords, save_folder, base_name)
        
        # Lưu ảnh kết quả
        result_path = f"{save_folder}/9_result_{base_name}.jpg"
        cv2.imwrite(result_path, result_image)
        
        # Lưu ảnh HMI đã trích xuất
        hmi_path = f"{save_folder}/hmi_{base_name}.jpg"
        cv2.imwrite(hmi_path, warped_roi)
        print(f"   ✓ Đã phát hiện và trích xuất màn hình HMI")
        
        return warped_roi
    
    return None

# ============================================================
# PADDLEOCR FUNCTIONS
# ============================================================

def get_ocr_instance():
    """Lấy hoặc tạo OCR instance (singleton pattern để tăng tốc)"""
    global _ocr_instance
    if _ocr_instance is None:
        print("Đang khởi tạo PaddleOCR reader...")
        
        # Khởi tạo với suppress output để ẩn thông báo không cần thiết
        with suppress_output():
            _ocr_instance = PaddleOCR(
                lang='en',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_thresh=0.15,
                text_det_box_thresh=0.25,
                text_det_unclip_ratio=2.2,
                text_rec_score_thresh=0.0,
                text_det_limit_side_len=512,
                text_det_limit_type='max',
            )
        print("✓ Khởi tạo thành công")
    return _ocr_instance

def select_image():
    """Mở hộp thoại chọn file ảnh"""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để đọc",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def read_image_with_paddleocr(image_input):
    """Đọc văn bản từ ảnh bằng PaddleOCR
    
    Args:
        image_input: có thể là đường dẫn file hoặc numpy array (ảnh OpenCV)
    
    Returns:
        tuple: (results, img_width, img_height)
    """
    ocr = get_ocr_instance()
    
    start_time = time.time()
    
    # Lấy kích thước ảnh
    if isinstance(image_input, np.ndarray):
        img_height, img_width = image_input.shape[:2]
        temp_path = "_temp_ocr_image.jpg"
        cv2.imwrite(temp_path, image_input)
        results = ocr.predict(temp_path)
        # Xóa file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        img = cv2.imread(image_input)
        if img is not None:
            img_height, img_width = img.shape[:2]
        else:
            img_height, img_width = 1, 1
        print(f"   Đang đọc OCR từ ảnh...")
        results = ocr.predict(image_input)
    
    elapsed = time.time() - start_time
    print(f"   ✓ OCR hoàn thành trong {elapsed:.2f} giây")
    
    return results, img_width, img_height

def extract_ocr_data(results):
    """Trích xuất dữ liệu từ OCRResult objects"""
    all_data = []
    
    if not results:
        return all_data
    
    for result in results:
        if hasattr(result, 'json') and result.json:
            json_data = result.json
            res = json_data.get('res', json_data)
            
            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            polys = res.get('rec_polys', res.get('dt_polys', []))
            
            for i in range(len(texts)):
                data = {
                    'text': texts[i] if i < len(texts) else '',
                    'confidence': scores[i] if i < len(scores) else 0.0,
                    'bbox': polys[i] if i < len(polys) else []
                }
                all_data.append(data)
    
    return all_data

def write_filtered_results_to_file(filtered_results, area, machine_code, screen_name, sub_page, output_file='paddleocr_output.txt'):
    """Ghi kết quả đã lọc (theo IoU) vào file txt"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"AREA: {area}\n")
        f.write(f"MACHINE: {machine_code}\n")
        f.write(f"SCREEN: {screen_name}\n")
        f.write(f"SUB-PAGE: {sub_page}\n")
        f.write(f"{'='*60}\n\n")
        
        if not filtered_results:
            f.write("Không tìm thấy kết quả OCR phù hợp với các ROI.\n")
            return 0
        
        for i, item in enumerate(filtered_results, 1):
            f.write(f"=== Kết quả {i} ===\n")
            f.write(f"ROI Name: {item['matched_roi']}\n")
            f.write(f"Văn bản: {item['text']}\n")
            f.write(f"Độ tin cậy: {item['confidence']:.4f}\n")
            f.write(f"IoU: {item['iou']:.2%}\n")
            f.write(f"Tọa độ OCR (normalized): {item['normalized_bbox']}\n")
            f.write(f"Tọa độ ROI (normalized): {item['roi_coords']}\n")
            f.write("\n")
    
    return len(filtered_results)

def print_results_summary(ocr_data, filtered_results=None, area=None, machine_code=None, screen_name=None, sub_page=None):
    """In tóm tắt kết quả ra console"""
    print("\n" + "="*60)
    print("TÓM TẮT KẾT QUẢ OCR")
    print("="*60)
    
    if area and machine_code and screen_name:
        sub_page_info = f" (Sub-page {sub_page})" if sub_page else ""
        print(f"🎯 Screen detected: {area}/{machine_code}/{screen_name}{sub_page_info}")
        print("-"*60)
    
    if filtered_results:
        print(f"\n📋 Kết quả đã lọc theo ROI (IoU >= {IOU_THRESHOLD:.0%}):")
        print("-"*60)
        for i, item in enumerate(filtered_results, 1):
            conf_pct = item['confidence'] * 100
            iou_pct = item['iou'] * 100
            print(f"  {i:2}. [{item['matched_roi']}] \"{item['text']}\"")
            print(f"      Confidence: {conf_pct:.1f}% | IoU: {iou_pct:.1f}%")
        
        print("\n" + "="*60)
        print(f"Tổng cộng: {len(filtered_results)} kết quả phù hợp")
    else:
        if not ocr_data:
            print("   Không tìm thấy văn bản nào.")
        else:
            print(f"\n📋 Tất cả kết quả OCR ({len(ocr_data)} items):")
            for i, data in enumerate(ocr_data, 1):
                conf_pct = data['confidence'] * 100
                print(f"  {i:2}. \"{data['text']}\" (confidence: {conf_pct:.1f}%)")
            print(f"\nTổng cộng: {len(ocr_data)} kết quả")
    
    print("="*60)

def process_single_image(image_path, image_count, save_folder, selected_area=None, selected_machine=None):
    """
    Xử lý một ảnh: phát hiện HMI -> OCR -> Match screen -> Filter by IoU
    
    Args:
        image_path: Đường dẫn ảnh
        image_count: Số thứ tự ảnh
        save_folder: Thư mục lưu kết quả
        selected_area: Khu vực đã chọn (F1, F4, ...)
        selected_machine: Mã máy đã chọn (IE-F1-CWA01, ...)
    """
    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_save_folder = f"{save_folder}/{base_name}_steps"
        if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)
        
        # Bước 1: Phát hiện và trích xuất màn hình HMI
        print(f"\n📷 Đang xử lý ảnh: {os.path.basename(image_path)}")
        if selected_area and selected_machine:
            print(f"   🎯 Đã chọn: {selected_area}/{selected_machine}")
        hmi_start = time.time()
        
        hmi_image = detect_hmi_screen(image_path, image_save_folder)
        
        hmi_time = time.time() - hmi_start
        print(f"   ⏱ Thời gian phát hiện HMI: {hmi_time:.2f} giây")
        
        # Bước 2: Thực hiện OCR trên ảnh HMI
        if hmi_image is not None:
            print(f"\n🔍 Đang thực hiện OCR trên màn hình HMI đã trích xuất...")
            results, img_width, img_height = read_image_with_paddleocr(hmi_image)
        else:
            print(f"\n⚠ Không tìm thấy màn hình HMI, thực hiện OCR trên ảnh gốc...")
            results, img_width, img_height = read_image_with_paddleocr(image_path)
        
        # Trích xuất dữ liệu OCR
        ocr_data = extract_ocr_data(results)
        
        # Bước 3: Load ROI info và tìm screen phù hợp
        roi_info = load_roi_info()
        
        area = None
        machine_code = None
        screen_name = None
        sub_page = None
        sub_page_data = None
        filtered_results = []
        
        if roi_info:
            print(f"\n🔎 Đang so khớp với Special_rois...")
            # Truyền selected_area và selected_machine để lọc
            area, machine_code, screen_name, sub_page, sub_page_data, match_count, match_percentage = find_matching_screen(
                ocr_data, roi_info, 
                selected_area=selected_area, 
                selected_machine=selected_machine
            )
            
            if screen_name:
                special_rois = sub_page_data.get('Special_rois', [])
                print(f"   ✓ Tìm thấy screen phù hợp: {area}/{machine_code}/{screen_name} (Sub-page {sub_page})")
                print(f"   ✓ Khớp {match_count}/{len(special_rois)} Special_rois ({match_percentage:.1f}%)")
                
                # Bước 4: Lọc kết quả OCR theo IoU với ROIs
                print(f"\n📐 Đang tính IoU và lọc kết quả (threshold >= {IOU_THRESHOLD:.0%})...")
                filtered_results = filter_ocr_by_roi(ocr_data, sub_page_data, img_width, img_height)
                print(f"   ✓ Tìm thấy {len(filtered_results)} kết quả phù hợp với ROIs")
            else:
                print(f"   ⚠ Không tìm thấy screen phù hợp trong roi_info.json")
        else:
            print(f"   ⚠ Không thể load roi_info.json")
        
        # In tóm tắt kết quả
        print_results_summary(ocr_data, filtered_results, area, machine_code, screen_name, sub_page)
        
        # Ghi kết quả vào file
        if image_count == 1:
            output_file = 'paddleocr_output.txt'
        else:
            output_file = f'paddleocr_output_{base_name}.txt'
        
        if filtered_results:
            # Ghi kết quả đã lọc
            count = write_filtered_results_to_file(filtered_results, area, machine_code, screen_name, sub_page, output_file)
            if count > 0:
                print(f"\n✓ Đã ghi {count} kết quả (đã lọc theo IoU) vào file: {output_file}")
            else:
                print("\n⚠ Không có kết quả nào phù hợp với ROIs.")
        else:
            # Nếu không có filtered results, ghi tất cả kết quả OCR
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Không tìm thấy screen phù hợp hoặc không có kết quả khớp với ROI.\n")
                f.write("\n=== TẤT CẢ KẾT QUẢ OCR ===\n\n")
                for i, data in enumerate(ocr_data, 1):
                    f.write(f"=== Kết quả {i} ===\n")
                    f.write(f"Văn bản: {data['text']}\n")
                    f.write(f"Độ tin cậy: {data['confidence']:.4f}\n")
                    f.write(f"Tọa độ: {data['bbox']}\n")
                    f.write("\n")
            print(f"\n✓ Đã ghi {len(ocr_data)} kết quả (chưa lọc) vào file: {output_file}")
        
        return True, hmi_time
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    """Hàm chính của chương trình"""
    print("=" * 60)
    print("   CHƯƠNG TRÌNH PHÁT HIỆN HMI VÀ ĐỌC OCR")
    print("   (Tích hợp HMI Detection + PaddleOCR + ROI Matching)")
    print("=" * 60)
    print()
    
    # Tạo thư mục lưu ảnh
    save_folder = "detected_images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # ============================================================
    # LOAD MACHINE SCREENS CONFIG
    # ============================================================
    machine_screens = load_machine_screens()
    if not machine_screens:
        print("⚠ Không thể load machine_screens.json. Chương trình sẽ duyệt tất cả ROI.")
    
    # ============================================================
    # KHỞI TẠO PADDLEOCR 1 LẦN DUY NHẤT
    # ============================================================
    init_start = time.time()
    
    try:
        get_ocr_instance()
    except ImportError:
        print("Lỗi: PaddleOCR chưa được cài đặt.")
        print("Vui lòng chạy: pip install paddleocr paddlepaddle")
        return
    
    init_time = time.time() - init_start
    print(f"⏱ Thời gian khởi tạo PaddleOCR: {init_time:.2f} giây")
    print()
    
    # ============================================================
    # VÒNG LẶP XỬ LÝ NHIỀU ẢNH
    # ============================================================
    image_count = 0
    total_processing_time = 0
    
    # Lưu lựa chọn area và machine để tái sử dụng
    last_selected_area = None
    last_selected_machine = None
    
    while True:
        print("-" * 60)
        print(f"📁 Vui lòng chọn ảnh (hoặc Cancel để thoát)...")
        image_path = select_image()
        
        if not image_path:
            print("\n🛑 Không có ảnh được chọn. Kết thúc chương trình.")
            break
        
        if not os.path.exists(image_path):
            print(f"⚠ Lỗi: Không tìm thấy file ảnh tại: {image_path}")
            continue
        
        # ============================================================
        # BƯỚC CHỌN KHU VỰC VÀ MÃ MÁY
        # ============================================================
        selected_area = None
        selected_machine = None
        
        if machine_screens:
            # Hỏi có muốn dùng lại lựa chọn trước không
            if last_selected_area and last_selected_machine:
                print(f"\n📌 Lựa chọn trước: {last_selected_area}/{last_selected_machine}")
                reuse = input("Dùng lại lựa chọn này? (Y/n): ").strip().lower()
                if reuse == '' or reuse == 'y':
                    selected_area = last_selected_area
                    selected_machine = last_selected_machine
                    print(f"   ✓ Sử dụng lại: {selected_area}/{selected_machine}")
            
            # Nếu không dùng lại, chọn mới
            if not selected_area or not selected_machine:
                # Bước 1: Chọn khu vực
                selected_area = select_area(machine_screens)
                
                if selected_area:
                    # Bước 2: Chọn mã máy
                    selected_machine = select_machine(machine_screens, selected_area)
                    
                    if not selected_machine:
                        # Quay lại chọn khu vực
                        print("   ↩ Quay lại chọn khu vực...")
                        continue
                else:
                    # Không chọn khu vực - duyệt tất cả
                    print("   ⚠ Không chọn khu vực. Sẽ duyệt tất cả ROI.")
            
            # Lưu lại lựa chọn
            if selected_area and selected_machine:
                last_selected_area = selected_area
                last_selected_machine = selected_machine
        
        image_count += 1
        process_start = time.time()
        
        # Truyền area và machine đã chọn vào process_single_image
        success, hmi_time = process_single_image(
            image_path, image_count, save_folder,
            selected_area=selected_area,
            selected_machine=selected_machine
        )
        
        if success:
            process_time = time.time() - process_start
            total_processing_time += process_time
            print(f"\n⏱ Tổng thời gian xử lý ảnh này: {process_time:.2f} giây")
        
        print()
    
    # ============================================================
    # THỐNG KÊ CUỐI CÙNG
    # ============================================================
    if image_count > 0:
        print()
        print("=" * 60)
        print("📊 THỐNG KÊ TỔNG HỢP")
        print("=" * 60)
        print(f"   • Số ảnh đã xử lý: {image_count}")
        print(f"   • Tổng thời gian xử lý: {total_processing_time:.2f} giây")
        print(f"   • Trung bình/ảnh: {total_processing_time/image_count:.2f} giây")
        print(f"   • Thời gian khởi tạo (1 lần): {init_time:.2f} giây")
        print(f"   • Thư mục kết quả: {save_folder}/")
        print("=" * 60)
    
    print("\n👋 Cảm ơn bạn đã sử dụng chương trình!")

def ocr_reader(image_path):
    """Wrapper function để đọc OCR"""
    results, _, _ = read_image_with_paddleocr(image_path)
    return results

if __name__ == "__main__":
    main()
