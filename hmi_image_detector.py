import cv2
import numpy as np
from math import sqrt, atan2, degrees
import time
import os
from tkinter import Tk, filedialog
from PIL import Image, ImageEnhance

def enhance_image(image):
    """Cải thiện chất lượng anh trước khi phat hien canh"""
    # Chuyển tu OpenCV (BGR) sang PIL (RGB) để áp dụng ImageEnhance
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Tăng độ tương phản voi PIL
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_pil = enhancer.enhance(2)  # Tăng độ tương phản lên 50%
    
    # Chuyển lại về định dạng OpenCV
    enhanced_image = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
    
    # Tiếp tục quy trinh xử lý anh như trước
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    # Tăng clip limit để cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(11, 11))  # Tăng tu 3.0 lên 4.0
    enhanced = clahe.apply(gray)
    
    # Tăng độ tương phản
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=0)  # Thêm bước tăng contrast
    
    # Làm mịn anh voi kernel nhỏ hơn để giữ nguyên canh sắc nét hơn
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)  # Giảm tu (7, 7) xuống (5, 5)
    return blurred, enhanced

def adaptive_edge_detection(image):
    """Phát hiện canh voi nhiều phương pháp và ket hop ket qua"""
    median_val = np.median(image)
    # Giảm ngưỡng để tăng độ nhạy cảm phat hien canh
    lower = int(max(0, (1.0 - 0.33) * median_val))  # Giảm tu 0.25 xuống 0.33
    upper = int(min(255, (1.0 + 0.33) * median_val))  # Tăng tu 0.25 lên 0.33
    canny_edges = cv2.Canny(image, lower, upper)
    
    # Sử dụng kernel lon hơn cho bộ lọc Sobel
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Giảm ngưỡng sobel để bắt duoc nhiều canh hơn
    _, sobel_edges = cv2.threshold(sobel_edges, 80, 255, cv2.THRESH_BINARY)  # Giảm tu 50 xuống 40
    
    # Kết hợp cả hai phương pháp phat hien canh
    combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
    
    # Tăng số lần giãn nở để kết nối cac canh bị đứt đoạn
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=2)  # Tăng tu 1 lên 2
    final_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    
    return canny_edges, sobel_edges, final_edges

def process_lines(lines, img_shape, min_length=20, max_lines_per_direction=30):
    """Xử lý và nhóm cac duong thang theo hướng ngang/doc, giới hạn so luong duong"""
    if lines is None:
        return [], [], []
    
    horizontal_lines = []
    vertical_lines = []
    
    all_h_lines = []
    all_v_lines = []
    
    height, width = img_shape[:2]
    min_dimension = min(height, width)
    
    # Giảm độ dài tối thiểu để phat hien nhiều duong hơn
    min_length = max(min_length, int(min_dimension * 0.02))  # Giảm tu 0.03 xuống 0.02
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if length < min_length:
            continue
        
        # Tính góc của duong thang
        if x2 != x1:
            angle = degrees(atan2(y2-y1, x2-x1))
        else:
            angle = 90  # Đường doc
        
        # Mở rộng phạm vi phan loai duong ngang/doc
        if abs(angle) < 40 or abs(angle) > 140:  # Đường ngang (mở rộng phạm vi tu 35 lên 40)
            all_h_lines.append([x1, y1, x2, y2, angle, length])
        elif abs(angle - 90) < 40 or abs(angle + 90) < 40:  # Đường doc (mở rộng phạm vi tu 35 lên 40)
            all_v_lines.append([x1, y1, x2, y2, angle, length])
    
    # Sắp xếp theo độ dài
    all_h_lines.sort(key=lambda x: x[5], reverse=True)
    all_v_lines.sort(key=lambda x: x[5], reverse=True)
    
    # Đảm bảo có du so luong duong ngang va doc tối thiểu
    min_lines = min(4, len(all_h_lines))  # Tăng so luong dòng tối thiểu tu 3 lên 4
    horizontal_lines = [line[:5] for line in all_h_lines[:max(min_lines, max_lines_per_direction)]]
    
    min_lines = min(4, len(all_v_lines))  # Tăng so luong dòng tối thiểu tu 3 lên 4
    vertical_lines = [line[:5] for line in all_v_lines[:max(min_lines, max_lines_per_direction)]]
    
    # Thêm debug
    print(f"So luong duong ngang: {len(horizontal_lines)}")
    print(f"So luong duong doc: {len(vertical_lines)}")
    
    return horizontal_lines, vertical_lines

def extend_lines(lines, width, height):
    """Kéo dài cac duong thang đến biên của anh"""
    extended_lines = []
    
    for x1, y1, x2, y2, angle in lines:
        # Xử lý duong doc (x không đổi)
        if abs(x2 - x1) < 5:  # Đường doc hoặc gần doc
            extended_lines.append([x1, 0, x1, height - 1, angle])
            continue
            
        # Xử lý duong ngang (y không đổi)
        if abs(y2 - y1) < 5:  # Đường ngang hoặc gần ngang
            extended_lines.append([0, y1, width - 1, y1, angle])
            continue
        
        # Xử lý cac duong xiên
        m = (y2 - y1) / (x2 - x1)  # Hệ số góc
        b = y1 - m * x1  # Hệ số tự do
        
        # Tính toán giao diem voi cac canh của anh
        intersections = []
        
        # Giao voi canh trái (x=0)
        y_left = m * 0 + b
        if 0 <= y_left < height:
            intersections.append((0, int(y_left)))
            
        # Giao voi canh phải (x=width-1)
        y_right = m * (width - 1) + b
        if 0 <= y_right < height:
            intersections.append((width - 1, int(y_right)))
            
        # Giao voi canh trên (y=0)
        if abs(m) > 1e-10:  # Tránh chia cho số quá nhỏ
            x_top = (0 - b) / m
            if 0 <= x_top < width:
                intersections.append((int(x_top), 0))
            
        # Giao voi canh dưới (y=height-1)
        if abs(m) > 1e-10:  # Tránh chia cho số quá nhỏ
            x_bottom = ((height - 1) - b) / m
            if 0 <= x_bottom < width:
                intersections.append((int(x_bottom), height - 1))
        
        # Nếu có du hai giao diem, tạo duong keo dai
        if len(intersections) >= 2:
            # Lấy hai giao diem đầu tiên
            p1, p2 = intersections[:2]
            extended_lines.append([p1[0], p1[1], p2[0], p2[1], angle])
    
    return extended_lines

def find_intersections(horizontal_lines, vertical_lines, max_intersections=200):
    """Tìm giao diem của cac duong ngang va doc, giới hạn so luong giao diem"""
    intersections = []
    
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            if len(intersections) >= max_intersections:
                break
                
            x1_h, y1_h, x2_h, y2_h, _ = h_line
            x1_v, y1_v, x2_v, y2_v, _ = v_line
            
            # Xử lý trường hợp đặc biệt của duong ngang va doc
            if abs(y1_h - y2_h) < 5 and abs(x1_v - x2_v) < 5:
                # Giao điểm của duong ngang thuần túy và duong doc thuần túy
                intersections.append((int(x1_v), int(y1_h)))
                continue
            
            # Sử dụng phương pháp đơn giản hơn để tìm giao diem
            try:
                # Chuyển sang float để tránh tràn số
                x1_h, y1_h, x2_h, y2_h = float(x1_h), float(y1_h), float(x2_h), float(y2_h)
                x1_v, y1_v, x2_v, y2_v = float(x1_v), float(y1_v), float(x2_v), float(y2_v)
                
                # Kiểm tra nếu duong ngang gần như ngang
                if abs(y2_h - y1_h) < 1e-10:
                    # Đường ngang hoàn toàn ngang
                    # Tính giá trị x tại y = y1_h trên duong doc
                    if abs(x2_v - x1_v) < 1e-10:
                        x_intersect = x1_v
                    else:
                        t = (y1_h - y1_v) / (y2_v - y1_v)
                        x_intersect = x1_v + t * (x2_v - x1_v)
                    
                    intersections.append((int(x_intersect), int(y1_h)))
                    continue
                
                # Kiểm tra nếu duong doc gần như doc
                if abs(x2_v - x1_v) < 1e-10:
                    # Đường doc hoàn toàn doc
                    # Tính giá trị y tại x = x1_v trên duong ngang
                    if abs(x2_h - x1_h) < 1e-10:
                        y_intersect = y1_h
                    else:
                        t = (x1_v - x1_h) / (x2_h - x1_h)
                        y_intersect = y1_h + t * (y2_h - y1_h)
                    
                    intersections.append((int(x1_v), int(y_intersect)))
                    continue
                
                # Trường hợp tổng quát - tìm giao diem bằng tham số t
                # Đường 1: P1 + t(P2-P1)
                # Đường 2: P3 + s(P4-P3)
                
                # Giải hệ phương trình:
                # P1.x + t(P2.x - P1.x) = P3.x + s(P4.x - P3.x)
                # P1.y + t(P2.y - P1.y) = P3.y + s(P4.y - P3.y)
                
                denom = (y2_v - y1_v) * (x2_h - x1_h) - (x2_v - x1_v) * (y2_h - y1_h)
                
                if abs(denom) < 1e-10:
                    # Các duong song song hoặc trùng nhau
                    continue
                
                # Tính tham số t cho duong 1
                ua = ((x2_v - x1_v) * (y1_h - y1_v) - (y2_v - y1_v) * (x1_h - x1_v)) / denom
                
                # Tính tọa độ giao diem
                x_intersect = x1_h + ua * (x2_h - x1_h)
                y_intersect = y1_h + ua * (y2_h - y1_h)
                
                # Kiểm tra giao diem có nằm trong đoạn duong không
                if (min(x1_h, x2_h) - 10 <= x_intersect <= max(x1_h, x2_h) + 10 and
                    min(y1_v, y2_v) - 10 <= y_intersect <= max(y1_v, y2_v) + 10):
                    intersections.append((int(x_intersect), int(y_intersect)))
            
            except (ValueError, OverflowError, ZeroDivisionError) as e:
                # Bỏ qua cac lỗi tính toán
                continue
        
        if len(intersections) >= max_intersections:
            break
    
    return intersections

def find_largest_rectangle(intersections, img_shape):
    """Tìm hinh chu nhat lon nhất tu cac giao diem"""
    if len(intersections) < 4:
        return None
    
    # Tìm cac điểm biên
    left_point = min(intersections, key=lambda p: p[0])
    right_point = max(intersections, key=lambda p: p[0])
    top_point = min(intersections, key=lambda p: p[1])
    bottom_point = max(intersections, key=lambda p: p[1])
    
    # Tính toán cac góc của hinh chu nhat lon nhất
    top_left = (left_point[0], top_point[1])
    top_right = (right_point[0], top_point[1])
    bottom_left = (left_point[0], bottom_point[1])
    bottom_right = (right_point[0], bottom_point[1])
    
    # Kiểm tra xem cac góc có nằm gần cac giao diem không
    threshold = 30  # Khoảng cach dung sai
    
    # Tìm giao diem gần nhất cho mỗi góc
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
    
    # Kiểm tra diện tích (ít nhất 1% và không quá 90% diện tích anh)
    if area < 0.01 * total_area or area > 0.9 * total_area:
        return None
    
    # Đảm bảo không có số âm
    if width <= 0 or height <= 0:
        return None
    
    # Kiểm tra tỷ lệ canh phu hop (tránh hinh chu nhat quá dài hoặc quá rộng)
    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
    if aspect_ratio > 5:  # Giới hạn tỷ lệ canh
        return None
    
    # Trả về hinh chu nhat và diện tích
    return (refined_top_left, refined_top_right, refined_bottom_right, refined_bottom_left, area)

def analyze_content(frame, rectangle):
    """Phân tích nội dung bên trong hinh chu nhat để xác định có phải man hinh HMI không"""
    pt1, pt2, pt3, pt4, _ = rectangle
    
    pts = [pt1, pt2, pt3, pt4]
    pts.sort(key=lambda p: p[0] + p[1])
    
    top_left = pts[0]
    bottom_right = pts[-1]
    
    remaining = [p for p in pts if p != top_left and p != bottom_right]
    top_right = min(remaining, key=lambda p: p[1])
    bottom_left = max(remaining, key=lambda p: p[1])
    
    x_min = min(top_left[0], bottom_left[0])
    y_min = min(top_left[1], top_right[1])
    x_max = max(top_right[0], bottom_right[0])
    y_max = max(bottom_left[1], bottom_right[1])
    
    # Kiểm tra biên
    if x_min < 0 or y_min < 0 or x_max >= frame.shape[1] or y_max >= frame.shape[0]:
        return False, None
    
    roi = frame[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return False, None
    
    # Phân tích đặc tính màu sắc
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, std_dev = cv2.meanStdDev(gray)
    color_uniformity = std_dev[0][0] < 60
    
    # Phân tích canh và duong thang
    enhanced, _ = enhance_image(roi)
    canny_edges, sobel_edges, edges = adaptive_edge_detection(enhanced)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=10)
    has_internal_lines = lines is not None and len(lines) > 3
    
    # Phân tích độ tương phản
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    contrast = max_val - min_val > 80
    
    # Kết hợp cac điều kiện
    is_hmi = (color_uniformity and has_internal_lines) or (contrast and has_internal_lines)
    
    # Trả về ket qua và vùng ROI
    roi_coords = (x_min, y_min, x_max, y_max)
    return is_hmi, roi_coords

def draw_lines_on_image(image, horizontal_lines, vertical_lines):
    """Vẽ duong ngang va doc lên anh"""
    img_with_lines = image.copy()
    
    # Vẽ duong ngang
    for line in horizontal_lines:
        x1, y1, x2, y2, _ = line
        cv2.line(img_with_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Vẽ duong doc
    for line in vertical_lines:
        x1, y1, x2, y2, _ = line
        cv2.line(img_with_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    return img_with_lines

def draw_intersections_on_image(image, intersections):
    """Vẽ cac giao diem lên anh"""
    img_with_intersections = image.copy()
    
    for point in intersections:
        cv2.circle(img_with_intersections, point, 5, (255, 0, 0), -1)
    
    return img_with_intersections

def draw_rectangles_on_image(image, rectangles):
    """Vẽ cac hinh chu nhat lên anh"""
    img_with_rectangles = image.copy()
    
    for rect in rectangles:
        pts = rect[:4]
        pts = np.array(pts)
        cv2.polylines(img_with_rectangles, [pts], True, (255, 255, 0), 2)
    
    return img_with_rectangles

def find_hmi_in_image(image_path, save_folder):
    """Tìm kiếm man hinh HMI trong anh và luu tat ca cac bước xử lý"""
    # Đọc anh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc anh: {image_path}")
        return [], []
    
    # Lấy tên cơ sở của file anh
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Xử lý anh
    print("Dang xu ly anh...")
    
    # Lưu anh goc
    original_path = f"{save_folder}/1_original_{base_name}.jpg"
    cv2.imwrite(original_path, image)
    print(f"Da luu anh goc vao: {original_path}")
    
    # Tạo bản sao để vẽ ket qua
    result_image = image.copy()
    
    # Bước 1: Tăng cường chất lượng anh
    enhanced_img, enhanced_clahe = enhance_image(image)
    enhanced_path = f"{save_folder}/2_enhanced_{base_name}.jpg"
    cv2.imwrite(enhanced_path, enhanced_img)
    print(f"Da luu anh tang cuong vao: {enhanced_path}")
    
    enhanced_clahe_path = f"{save_folder}/2b_enhanced_clahe_{base_name}.jpg"
    cv2.imwrite(enhanced_clahe_path, enhanced_clahe)
    print(f"Da luu anh tang cuong CLAHE vao: {enhanced_clahe_path}")
    
    # Bước 2: Phát hiện canh
    canny_edges, sobel_edges, edges = adaptive_edge_detection(enhanced_clahe)
    
    canny_path = f"{save_folder}/3a_canny_edges_{base_name}.jpg"
    cv2.imwrite(canny_path, canny_edges)
    print(f"Da luu anh canh Canny vao: {canny_path}")
    
    sobel_path = f"{save_folder}/3b_sobel_edges_{base_name}.jpg"
    cv2.imwrite(sobel_path, sobel_edges)
    print(f"Da luu anh canh Sobel vao: {sobel_path}")
    
    edges_path = f"{save_folder}/3c_combined_edges_{base_name}.jpg"
    cv2.imwrite(edges_path, edges)
    print(f"Da luu anh canh ket hop vao: {edges_path}")
    
    # Bước 3: Tìm và lọc contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Vẽ tat ca contour
    all_contours_image = image.copy()
    cv2.drawContours(all_contours_image, contours, -1, (0, 255, 0), 2)
    all_contours_path = f"{save_folder}/4a_all_contours_{base_name}.jpg"
    cv2.imwrite(all_contours_path, all_contours_image)
    print(f"Da luu anh tat ca contour vao: {all_contours_path}")
    
    # Lọc contour theo diện tích
    min_contour_area = image.shape[0] * image.shape[1] * 0.001  # 0.1% diện tích anh
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Vẽ contour lon
    large_contours_image = image.copy()
    cv2.drawContours(large_contours_image, large_contours, -1, (0, 255, 0), 2)
    large_contours_path = f"{save_folder}/4b_large_contours_{base_name}.jpg"
    cv2.imwrite(large_contours_path, large_contours_image)
    print(f"Da luu anh contour lon vao: {large_contours_path}")
    
    # Tạo contour mask
    contour_mask = np.zeros_like(edges)
    cv2.drawContours(contour_mask, large_contours, -1, 255, 2)
    contour_mask_path = f"{save_folder}/4c_contour_mask_{base_name}.jpg"
    cv2.imwrite(contour_mask_path, contour_mask)
    print(f"Da luu anh contour mask vao: {contour_mask_path}")
    
    # Bước 4: Phát hiện duong thang - Điều chỉnh cac tham số
    lines = cv2.HoughLinesP(contour_mask, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=30)  # Giảm threshold, minLineLength và tăng maxLineGap

    # Nếu không tìm duoc duong thang, thử điều chỉnh tham số
    if lines is None or len(lines) < 2:
        # Thử voi cac tham số dễ dàng hơn
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=10, maxLineGap=40)
        
        if lines is None or len(lines) < 2:
            # Thử lần cuối voi cac tham số rất dễ dàng
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=50)
    
    if lines is None:
        print("Khong tim thay duong thang trong anh.")
        return [], []
    
    # Vẽ tat ca duong thang
    all_lines_image = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(all_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    all_lines_path = f"{save_folder}/5a_all_lines_{base_name}.jpg"
    cv2.imwrite(all_lines_path, all_lines_image)
    print(f"Da luu anh tat ca duong thang vao: {all_lines_path}")
    
    # Bước 5: Phân loại duong ngang/doc
    height, width = image.shape[:2]
    horizontal_lines, vertical_lines = process_lines(lines, image.shape, min_length=20)
    
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        print("Khong tim thay du duong ngang va doc.")
        # Lưu ket qua dù không tìm thấy du duong
        result_path = f"{save_folder}/9_result_{base_name}.jpg"
        cv2.imwrite(result_path, result_image)
        print(f"Da luu anh ket qua vao: {result_path}")
        return [], []
    
    # Vẽ duong ngang/doc
    classified_lines_image = draw_lines_on_image(image, horizontal_lines, vertical_lines)
    classified_lines_path = f"{save_folder}/5b_classified_lines_{base_name}.jpg"
    cv2.imwrite(classified_lines_path, classified_lines_image)
    print(f"Da luu anh phan loai duong vao: {classified_lines_path}")
    
    # PHẦN MỚI: Thử tìm hinh chu nhat tu cac duong da phan loai
    largest_rectangle = find_rectangle_from_classified_lines(horizontal_lines, vertical_lines, image.shape)
    
    # Nếu tìm thấy hinh chu nhat hợp lệ tu cac duong da phan loai
    if largest_rectangle is not None:
        print("Da tim thay hinh chu nhat tu cac duong da phan loai!")
        # Bỏ qua cac bước 6, 7 và chuyển thẳng đến bước 8
        # Vẽ hinh chu nhat tìm duoc
        direct_rectangle_image = image.copy()
        pts = np.array(largest_rectangle[:4])
        cv2.polylines(direct_rectangle_image, [pts], True, (255, 255, 0), 2)
        direct_rectangle_path = f"{save_folder}/5c_direct_rectangle_{base_name}.jpg"
        cv2.imwrite(direct_rectangle_path, direct_rectangle_image)
        print(f"Da luu anh hinh chu nhat truc tiep vao: {direct_rectangle_path}")
    else:
        # Nếu không tìm duoc hinh chu nhat tu cac duong da phan loai, tiep tuc voi quy trinh thong thuong
        print("Khong tim thay hinh chu nhat tu cac duong da phan loai, tiep tuc voi quy trinh thong thuong...")
        
        # Bước 6: Kéo dài duong
        extended_h_lines = extend_lines(horizontal_lines, width, height)
        extended_v_lines = extend_lines(vertical_lines, width, height)
        
        # Vẽ duong keo dai
        extended_lines_image = draw_lines_on_image(image, extended_h_lines, extended_v_lines)
        extended_lines_path = f"{save_folder}/6_extended_lines_{base_name}.jpg"
        cv2.imwrite(extended_lines_path, extended_lines_image)
        print(f"Da luu anh duong keo dai vao: {extended_lines_path}")
        
        # Bước 7: Tìm giao diem
        intersections = find_intersections(extended_h_lines, extended_v_lines)
        
        if len(intersections) < 4:
            print("Khong tim thay du giao diem để tạo hinh chu nhat.")
            # Lưu ket qua dù không tìm thấy du giao diem
            result_path = f"{save_folder}/9_result_{base_name}.jpg"
            cv2.imwrite(result_path, result_image)
            print(f"Da luu anh ket qua vao: {result_path}")
            return [], []
        
        # Vẽ giao diem
        intersections_image = draw_intersections_on_image(extended_lines_image, intersections)
        intersections_path = f"{save_folder}/7_intersections_{base_name}.jpg"
        cv2.imwrite(intersections_path, intersections_image)
        print(f"Da luu anh giao diem vao: {intersections_path}")
        
        # Bước 8: Tìm hinh chu nhat lon nhất tu cac giao diem xa nhất
        largest_rectangle = find_largest_rectangle(intersections, image.shape)
        
        if largest_rectangle is None:
            print("Khong tim thay hinh chu nhat phu hop.")
            # Lưu ket qua dù không tìm thấy hinh chu nhat
            result_path = f"{save_folder}/9_result_{base_name}.jpg"
            cv2.imwrite(result_path, result_image)
            print(f"Da luu anh ket qua vao: {result_path}")
            return [], []
        
        # Vẽ hinh chu nhat lon nhất
        largest_rectangle_image = image.copy()
        pts = np.array(largest_rectangle[:4])
        cv2.polylines(largest_rectangle_image, [pts], True, (255, 255, 0), 2)
        largest_rectangle_path = f"{save_folder}/8_largest_rectangle_{base_name}.jpg"
        cv2.imwrite(largest_rectangle_path, largest_rectangle_image)
        print(f"Da luu anh hinh chu nhat lon nhất vao: {largest_rectangle_path}")
    
    # Bước 9: Xác định vùng HMI tu hinh chu nhat lon nhất
    detected_hmis = []
    refined_hmis = []
    
    # Lấy cac góc của hinh chu nhat
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
    
    if x_max > x_min and y_max > y_min:
        roi_coords = (x_min, y_min, x_max, y_max)
        detected_hmis.append(roi_coords)
        
        # Vẽ hinh chu nhat lên anh ket qua
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Cắt và luu vùng HMI
        roi = image[y_min:y_max, x_min:x_max]
        roi_path = f"{save_folder}/8b_roi_{base_name}.jpg"
        cv2.imwrite(roi_path, roi)
        print(f"Da luu anh vung HMI ban dau vao: {roi_path}")
        
        # Tinh chỉnh và trai phang vùng HMI
        warped_roi, refined_coords = fine_tune_hmi_screen(image, roi_coords, save_folder, base_name)
        refined_hmis.append((warped_roi, refined_coords))
    
    # Lưu anh ket qua (da đánh dấu cac man hinh HMI)
    result_path = f"{save_folder}/9_result_{base_name}.jpg"
    cv2.imwrite(result_path, result_image)
    print(f"Da luu anh ket qua vao: {result_path}")
    
    return detected_hmis, refined_hmis

def select_image():
    """Hiển thị hộp thoại để chọn anh"""
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ goc
    file_path = filedialog.askopenfilename(
        title="Chọn anh",
        filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return file_path

def main():
    # Tạo thư mục để luu hình anh nếu chưa tồn tại
    save_folder = "detected_images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Chọn anh để xử lý
    print("Vui lòng chọn anh để phat hien man hinh HMI...")
    image_path = select_image()
    
    if not image_path:
        print("Không có anh nào duoc chọn. Thoát chương trình.")
        return
    
    # Tạo thư mục riêng cho anh này nếu cần
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_save_folder = f"{save_folder}/{base_name}_steps"
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)
    
    # Tìm kiếm man hinh HMI trong anh và luu tat ca cac bước xử lý
    hmi_regions, refined_hmis = find_hmi_in_image(image_path, image_save_folder)
    
    # Đọc anh goc
    image = cv2.imread(image_path)
    
    # Cắt và luu cac vùng man hinh HMI
    if refined_hmis:
        for i, (warped_roi, _) in enumerate(refined_hmis):
            # Lưu vùng HMI da trai phang
            save_path = f"{save_folder}/hmi_{base_name}_{i+1}.jpg"
            cv2.imwrite(save_path, warped_roi)
            print(f"Da luu man hinh HMI {i+1} da trai phang vao: {save_path}")
        
        print(f"Hoan tat! Đã phat hien và luu {len(refined_hmis)} man hinh HMI.")
    else:
        print("Khong tim thay man hinh HMI nào trong anh.")
        print("Tất cả cac anh qua cac bước xử lý da duoc luu vao thư mục.")

def extract_content_region(img, save_folder, base_name):
    """
    Trích xuất vùng nội dung (không phải vùng đen xung quanh man hinh sử dụng gradient và kernel theo chiều doc
    """
    # Chuyển sang anh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tăng độ tương phản để làm nổi bật duong viền man hinh
    enhanced_contrast = cv2.convertScaleAbs(gray, alpha=1.3, beta=5)
    
    # Lưu anh tang cuong độ tương phản để debug
    enhanced_path = f"{save_folder}/8b_content_enhanced_{base_name}.jpg"
    cv2.imwrite(enhanced_path, enhanced_contrast)
    
    # Làm mịn anh để giảm nhiễu nhưng vẫn giữ duoc canh
    blurred = cv2.GaussianBlur(enhanced_contrast, (3, 3), 0)  # Kernel nhỏ hơn để giữ duoc canh
    
    # Phân tích gradient để tìm cac vùng có độ tương phản cao
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobel_x, sobel_y)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Ngưỡng gradient - sử dụng ngưỡng thấp hơn để bắt duoc nhiều canh hơn
    _, gradient_thresh = cv2.threshold(gradient_mag, 20, 255, cv2.THRESH_BINARY)  # Giảm ngưỡng tu 30 xuống 20
    
    # Lưu anh gradient trước khi dilation
    gradient_before_path = f"{save_folder}/8b_content_gradient_before_{base_name}.jpg"
    cv2.imwrite(gradient_before_path, gradient_thresh)
    
    # Tạo kernel theo chiều doc cao hơn để bắt duoc toàn bộ cac canh doc
    vertical_kernel = np.ones((11, 3), np.uint8)  # Tăng tu (9, 3) lên (11, 3)
    
    # Mở rộng cac canh theo chiều doc
    gradient_dilated = cv2.dilate(gradient_thresh, vertical_kernel, iterations=3)  # Tăng iterations tu 2 lên 3
    
    # Đảm bảo kết nối tốt theo chiều ngang
    horizontal_kernel = np.ones((3, 9), np.uint8)  # Tăng tu (3, 7) lên (3, 9)
    gradient_dilated = cv2.dilate(gradient_dilated, horizontal_kernel, iterations=2)  # Tăng iterations tu 1 lên 2
    
    # Lưu anh gradient sau khi dilation
    gradient_path = f"{save_folder}/8b_content_gradient_{base_name}.jpg"
    cv2.imwrite(gradient_path, gradient_dilated)
    
    # Làm mịn và loại bỏ nhiễu
    kernel = np.ones((5, 5), np.uint8)
    gradient_final = cv2.morphologyEx(gradient_dilated, cv2.MORPH_CLOSE, kernel, iterations=3)  # Tăng tu 2 lên 3
    
    # Lưu anh gradient sau khi hoàn thiện
    gradient_final_path = f"{save_folder}/8b_content_gradient_final_{base_name}.jpg"
    cv2.imwrite(gradient_final_path, gradient_final)
    
    # Tìm contour truc tiep tu anh gradient
    contours, _ = cv2.findContours(gradient_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu không tìm thấy contour, thử voi phương pháp ngưỡng
    if not contours:
        print("Khong tim thay contour tu gradient, chuyển sang phương pháp ngưỡng")
        # Áp dụng phương pháp ngưỡng tự động bằng Otsu
        # Trước tiên, tăng độ tương phản
        enhanced_for_threshold = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        _, thresh = cv2.threshold(enhanced_for_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_path = f"{save_folder}/8b_content_otsu_thresh_{base_name}.jpg"
        cv2.imwrite(thresh_path, thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc ra cac contour lon (diện tích > 0.5% của anh)
    min_area = img.shape[0] * img.shape[1] * 0.005  # Giảm tu 1% xuống 0.5%
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Vẽ tat ca contour lon để kiểm tra
    all_contours_img = img.copy()
    cv2.drawContours(all_contours_img, large_contours, -1, (0, 255, 0), 2)
    all_contours_path = f"{save_folder}/8b_all_large_contours_{base_name}.jpg"
    cv2.imwrite(all_contours_path, all_contours_img)
    
    # Tạo mask tu contour lon nhất
    mask = np.zeros_like(gray)
    if large_contours:
        largest_contour = max(large_contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)  # Vẽ đầy contour
    else:
        print("Khong tim thay contour lon, trả về mask đầy")
        mask.fill(255)  # Trả về mask đầy nếu không tìm thấy contour
    
    # Lưu mask cuối cùng và contour lon nhất
    mask_path = f"{save_folder}/8b_content_final_mask_{base_name}.jpg"
    cv2.imwrite(mask_path, mask)
    
    contour_img = img.copy()
    if large_contours:
        cv2.drawContours(contour_img, [largest_contour], 0, (0, 255, 0), 2)
    contour_path = f"{save_folder}/8b_content_largest_contour_{base_name}.jpg"
    cv2.imwrite(contour_path, contour_img)
    
    return mask, large_contours[0] if large_contours else None

def fine_tune_hmi_screen(image, roi_coords, save_folder, base_name):
    """
    Tinh chỉnh vùng man hinh HMI da phat hien:
    1. Loại bỏ vùng đen xung quanh man hinh sử dụng gradient và kernel theo chiều doc
    2. Áp dụng Warp Perspective truc tiep trên contour lon nhất
    """
    x_min, y_min, x_max, y_max = roi_coords
    roi = image[y_min:y_max, x_min:x_max]
    
    # Lưu anh ROI goc để debug
    roi_original_path = f"{save_folder}/8b_roi_original_{base_name}.jpg"
    cv2.imwrite(roi_original_path, roi)
    
    # THAY ĐỔI: Tìm vùng nội dung và lấy contour lon nhất truc tiep
    content_mask, largest_contour = extract_content_region(roi, save_folder, base_name)
    
    # Kiểm tra nếu không tìm duoc contour
    if largest_contour is None:
        print("Khong tim thay contour lon trong ROI")
        return roi, roi_coords
    
    # Kiểm tra diện tích contour
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < 0.1 * roi.shape[0] * roi.shape[1]:
        print("Vùng nội dung quá nhỏ, có thể không phải là man hinh HMI")
        return roi, roi_coords
    
    # Xấp xỉ contour thanh đa giác
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Vẽ đa giác xấp xỉ để debug
    roi_approx = roi.copy()
    cv2.drawContours(roi_approx, [approx], 0, (0, 0, 255), 2)
    approx_path = f"{save_folder}/8d_roi_approx_{base_name}.jpg"
    cv2.imwrite(approx_path, roi_approx)
    
    # Nếu không có đúng 4 điểm, điều chỉnh để có 4 điểm
    if len(approx) != 4:
        # Sử dụng hinh chu nhat bao quanh tối thiểu
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = np.array(box, dtype=np.int32)
        
        # Vẽ hinh chu nhat da điều chỉnh
        roi_rect = roi.copy()
        cv2.drawContours(roi_rect, [approx], 0, (255, 0, 0), 2)
        rect_path = f"{save_folder}/8e_roi_adjusted_rect_{base_name}.jpg"
        cv2.imwrite(rect_path, roi_rect)
    
    # Chuyển đổi sang mảng điểm
    points = approx.reshape(-1, 2)
    
    # Sắp xếp cac điểm để chuẩn bị cho biến đổi phối canh
    points = order_points(points)
    
    # Tính toán chiều rộng và chiều cao của man hinh đích
    # Sử dụng khoảng cach Euclidean
    width_a = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    width_b = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    height_b = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Đảm bảo kích thước hợp lý
    if max_width < 10 or max_height < 10:
        print("Kích thước man hinh HMI quá nhỏ")
        return roi, roi_coords
    
    # Tạo điểm đích cho biến đổi phối canh
    dst_points = np.array([
        [0, 0],                     # top-left
        [max_width - 1, 0],         # top-right
        [max_width - 1, max_height - 1],  # bottom-right
        [0, max_height - 1]         # bottom-left
    ], dtype=np.float32)
    
    # Chuyển đổi points sang float32
    src_points = points.astype(np.float32)
    
    # Vẽ điểm nguồn lên anh để debug
    roi_points = roi.copy()
    for i, point in enumerate(src_points):
        cv2.circle(roi_points, tuple(point.astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(roi_points, str(i), tuple(point.astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    points_path = f"{save_folder}/8f_roi_source_points_{base_name}.jpg"
    cv2.imwrite(points_path, roi_points)
    
    # Thực hiện biến đổi phối canh
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(roi, M, (max_width, max_height))
    
    # Lưu anh da biến đổi
    warped_path = f"{save_folder}/8g_roi_warped_{base_name}.jpg"
    cv2.imwrite(warped_path, warped)
    
    # Kiểm tra xem có cần phải thực hiện hậu xử lý không
    # (Loại bỏ vì giờ da trích xuất truc tiep tu contour gradient)
    
    # Tính toán tọa độ mới
    new_roi_coords = (x_min, y_min, x_min + warped.shape[1], y_min + warped.shape[0])
    
    return warped, new_roi_coords

def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
    """
    # Khởi tạo danh sách để luu cac điểm da sắp xếp
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Tính tổng của cac tọa độ x+y
    # Điểm có tổng nhỏ nhất là top-left
    # Điểm có tổng lon nhất là bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # Tính hiệu của cac tọa độ (y-x)
    # Điểm có hiệu lon nhất là bottom-left
    # Điểm có hiệu nhỏ nhất là top-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    # Trả về danh sách cac điểm da sắp xếp
    return rect

def find_rectangle_from_classified_lines(horizontal_lines, vertical_lines, img_shape):
    """Tìm hinh chu nhat tu cac duong da phan loai ngang va doc"""
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None
    
    # Tìm duong ngang trên cùng và dưới cùng
    top_line = min(horizontal_lines, key=lambda line: min(line[1], line[3]))
    bottom_line = max(horizontal_lines, key=lambda line: max(line[1], line[3]))
    
    # Tìm duong doc trái cùng và phải cùng
    left_line = min(vertical_lines, key=lambda line: min(line[0], line[2]))
    right_line = max(vertical_lines, key=lambda line: max(line[0], line[2]))
    
    # Tính toán cac tọa độ y cho duong ngang trên và dưới
    top_y = min(top_line[1], top_line[3])
    bottom_y = max(bottom_line[1], bottom_line[3])
    
    # Tính toán cac tọa độ x cho duong doc trái và phải
    left_x = min(left_line[0], left_line[2])
    right_x = max(right_line[0], right_line[2])
    
    # Kiểm tra xem cac duong có tạo thanh hinh chu nhat hợp lý không (du gần nhau)
    # Kiểm tra ngang
    top_left_x = max(min(top_line[0], top_line[2]), left_x)
    top_right_x = min(max(top_line[0], top_line[2]), right_x)
    bottom_left_x = max(min(bottom_line[0], bottom_line[2]), left_x)
    bottom_right_x = min(max(bottom_line[0], bottom_line[2]), right_x)
    
    # Kiểm tra doc
    left_top_y = max(min(left_line[1], left_line[3]), top_y)
    left_bottom_y = min(max(left_line[1], left_line[3]), bottom_y)
    right_top_y = max(min(right_line[1], right_line[3]), top_y)
    right_bottom_y = min(max(right_line[1], right_line[3]), bottom_y)
    
    # Kiểm tra xem cac duong có du dài để nối voi nhau không
    if (top_right_x - top_left_x < 10 or bottom_right_x - bottom_left_x < 10 or
        left_bottom_y - left_top_y < 10 or right_bottom_y - right_top_y < 10):
        return None
    
    # Kiểm tra kích thước của hinh chu nhat
    height, width = img_shape[:2]
    
    # Điều chỉnh tọa độ để nằm trong anh
    if left_x < 0: left_x = 0
    if top_y < 0: top_y = 0
    if right_x >= width: right_x = width - 1
    if bottom_y >= height: bottom_y = height - 1
    
    # Tính kích thước hinh chu nhat
    rect_width = right_x - left_x
    rect_height = bottom_y - top_y
    
    # Kiểm tra kích thước tối thiểu
    if rect_width < 20 or rect_height < 20:
        return None
    
    # Kiểm tra tỷ lệ canh
    aspect_ratio = max(rect_width, rect_height) / (min(rect_width, rect_height) + 1e-6)
    if aspect_ratio > 5:  # Giới hạn tỷ lệ canh
        return None
    
    # Tạo cac góc của hinh chu nhat
    top_left = (int(left_x), int(top_y))
    top_right = (int(right_x), int(top_y))
    bottom_right = (int(right_x), int(bottom_y))
    bottom_left = (int(left_x), int(bottom_y))
    
    # Tính diện tích
    area = rect_width * rect_height
    
    # Kiểm tra diện tích tối thiểu và tối đa
    total_area = height * width
    if area < 0.01 * total_area or area > 0.9 * total_area:
        return None
    
    return (top_left, top_right, bottom_right, bottom_left, area)

if __name__ == "__main__":
    main() 