"""
Image Processor Module  
Handles all image processing operations including alignment, preprocessing,
quality check, and HMI screen detection
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance


# Global CV2 detectors - will be initialized by app
sift_detector = None
flann_matcher = None


def init_cv2_detectors(sift_det, flann_match):
    """Initialize global CV2 detectors from app"""
    global sift_detector, flann_matcher
    sift_detector = sift_det
    flann_matcher = flann_match


class ImageAligner:
    """Class để căn chỉnh ảnh HMI với template using SIFT features"""
    
    def __init__(self, template_img, source_img):
        """Khởi tạo với ảnh mẫu và ảnh nguồn"""
        self.template_img = template_img.copy()
        self.source_img = source_img.copy()
        self.warped_img = None
        self.detector = sift_detector
        
    def align_images(self):
        """Căn chỉnh ảnh nguồn để khớp với ảnh mẫu"""
        if self.detector is None:
            print("Warning: SIFT detector not available")
            return self.source_img.copy()
            
        # Convert to grayscale
        template_gray = cv2.cvtColor(self.template_img, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(self.source_img, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        template_keypoints, template_descriptors = self.detector.detectAndCompute(template_gray, None)
        source_keypoints, source_descriptors = self.detector.detectAndCompute(source_gray, None)
        
        if flann_matcher is None:
            print("Warning: FLANN matcher not available")
            return self.source_img.copy()
            
        # Match features
        matches = flann_matcher.knnMatch(source_descriptors, template_descriptors, k=2)
        
        # Filter good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches")
        
        if len(good_matches) < 10:
            print("Warning: Not enough matches for homography")
            return self.source_img.copy()
        
        # Extract location of good matches
        source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches])
        template_points = np.float32([template_keypoints[m.trainIdx].pt for m in good_matches])
        
        # Find homography matrix
        H, mask = cv2.findHomography(source_points, template_points, cv2.RANSAC, 5.0)
        
        # Warp source image
        h, w = self.template_img.shape[:2]
        self.warped_img = cv2.warpPerspective(self.source_img, H, (w, h))
        self.homography_matrix = H
        
        return self.warped_img
    
    def get_homography_matrix(self):
        """Trả về ma trận homography"""
        if hasattr(self, 'homography_matrix'):
            return self.homography_matrix
        return None
    
    def transform_roi_coordinates(self, roi_coordinates):
        """Biến đổi tọa độ ROI dựa trên ma trận homography"""
        try:
            H = self.get_homography_matrix()
            if H is None:
                return roi_coordinates
        
            transformed_coordinates = []
            for coord_set in roi_coordinates:
                if len(coord_set) != 4:
                    continue
                    
                x1, y1, x2, y2 = coord_set
                tx1, ty1 = self.transform_point((x1, y1), H)
                tx2, ty2 = self.transform_point((x2, y2), H)
                transformed_coordinates.append((tx1, ty1, tx2, ty2))
        
            return transformed_coordinates
        except Exception as e:
            print(f"Error transforming ROI coordinates: {str(e)}")
            return roi_coordinates
    
    def transform_point(self, point, H):
        """Áp dụng ma trận homography cho một điểm"""
        x, y = point
        p = np.array([x, y, 1])
        p_transformed = np.dot(H, p)
        x_transformed = p_transformed[0] / p_transformed[2]
        y_transformed = p_transformed[1] / p_transformed[2]
        return int(x_transformed), int(y_transformed)


def preprocess_hmi_image_with_alignment(image, template_path, roi_coordinates, original_filename):
    """Tiền xử lý ảnh với căn chỉnh perspective"""
    # Đọc ảnh template
    template_img = cv2.imread(template_path)
    if template_img is None:
        print(f"Warning: Could not read template at {template_path}")
        return preprocess_hmi_image(image, roi_coordinates, original_filename)
    
    print(f"Image shape: {image.shape}, Template shape: {template_img.shape}")
    
    # Căn chỉnh ảnh
    aligner = ImageAligner(template_img, image)
    aligned_image = aligner.align_images()
    
    # Xử lý ROI trên ảnh đã căn chỉnh
    results = preprocess_hmi_image(aligned_image, roi_coordinates, original_filename)
    
    return results


def preprocess_hmi_image(image, roi_coordinates, original_filename):
    """
    Tiền xử lý ảnh HMI và trích xuất các ROI
    
    Returns:
        list: Danh sách các ROI đã được preprocess
    """
    results = []
    
    try:
        img_height, img_width = image.shape[:2]
        
        for i, coords in enumerate(roi_coordinates):
            if len(coords) != 4:
                continue
            
            # Chuyển đổi tọa độ
            is_normalized = any(isinstance(value, float) and 0 <= value <= 1 for value in coords)
            
            if is_normalized:
                x1, y1, x2, y2 = coords
                x1, x2 = int(x1 * img_width), int(x2 * img_width)
                y1, y2 = int(y1 * img_height), int(y2 * img_height)
            else:
                x1, y1, x2, y2 = [int(float(c)) for c in coords]
            
            # Đảm bảo thứ tự tọa độ
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Kiểm tra tọa độ hợp lệ
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
                continue
            
            # Cắt ROI
            roi = image[y1:y2, x1:x2]
            
            if roi.size > 0:
                results.append({
                    "roi_index": i,
                    "roi_image": roi,
                    "coordinates": (x1, y1, x2, y2)
                })
    
    except Exception as e:
        print(f"Error in preprocess_hmi_image: {str(e)}")
    
    return results


def preprocess_roi_for_ocr(roi, roi_index, original_filename, roi_name=None, 
                           image_aligned=None, x1=None, y1=None, x2=None, y2=None):
    """
    Tiền xử lý ROI để tăng chất lượng OCR
    """
    try:
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Resize nếu ROI quá nhỏ
        h, w = gray.shape
        if h < 30 or w < 30:
            scale = max(30/h, 30/w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return morph
        
    except Exception as e:
        print(f"Error in preprocess_roi_for_ocr: {str(e)}")
        return roi


def check_image_quality(image):
    """
    Kiểm tra chất lượng ảnh
    
    Returns:
        dict: Thông tin về chất lượng ảnh
    """
    result = {
        'is_good_quality': True,
        'issues': [],
        'blurriness': 0,
        'brightness': 0,
        'contrast': 0,
        'has_glare': False,
        'has_moire': False
    }
    
    if image is None or image.size == 0:
        result['is_good_quality'] = False
        result['issues'].append('empty_image')
        return result
    
    # Convert to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Check brightness
    brightness = np.mean(gray)
    result['brightness'] = brightness
    
    # Check contrast
    contrast = np.std(gray)
    result['contrast'] = contrast
    
    # Check blurriness using Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result['blurriness'] = laplacian_var
    
    blur_threshold = 7.0
    if laplacian_var < blur_threshold:
        result['is_good_quality'] = False
        result['issues'].append('blurry')
        return result
    
    # Check brightness issues
    if brightness > 220:
        result['is_good_quality'] = False
        result['issues'].append('too_bright')
        return result
    
    # Check contrast
    if contrast < 16:
        result['is_good_quality'] = False
        result['issues'].append('low_contrast')
        return result
    
    # Check glare
    bright_threshold = 250
    bright_pixels = np.sum(gray > bright_threshold)
    bright_ratio = bright_pixels / gray.size
    
    if bright_ratio > 0.2:
        result['has_glare'] = True
        result['is_good_quality'] = False
        result['issues'].append('glare')
        return result
    
    # Check moire pattern using FFT
    try:
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        magnitude_spectrum = 20 * np.log(magnitude_spectrum + 1e-10)
        magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
        
        threshold = np.percentile(magnitude_spectrum, 99)
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        r = min(rows, cols) // 4
        
        mask = np.zeros((rows, cols), dtype=bool)
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
                if r / 2 < dist < r:
                    mask[i, j] = True
        
        filtered_magnitude = magnitude_spectrum * mask
        peaks = np.sum(filtered_magnitude > threshold)
        peak_ratio = peaks / np.sum(mask) if np.sum(mask) > 0 else 0
        
        if (brightness > 200) and (contrast > 20) and (0.077 > peak_ratio > 0.005):
            result['has_moire'] = True
            result['is_good_quality'] = False
            result['issues'].append('moire_pattern')
    except:
        pass
    
    return result


def enhance_image_quality(image, quality_info):
    """Cải thiện chất lượng ảnh dựa trên kết quả kiểm tra"""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    enhanced = gray.copy()
    
    # Handle blurry images
    if 'blurry' in quality_info['issues']:
        kernel = np.array([[-1, -1, -1],
                           [-1, 10, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Handle too bright images
    if 'too_bright' in quality_info['issues']:
        alpha = 0.9
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=0)
    
    # Handle glare
    if quality_info['has_glare']:
        enhanced = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=11, C=2
        )
    
    return enhanced


def enhance_image(image):
    """Tăng cường chất lượng ảnh tổng quát"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def adaptive_edge_detection(image):
    """Phát hiện cạnh thích ứng"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Auto threshold Canny
    median = np.median(blurred)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    
    edges = cv2.Canny(blurred, lower, upper)
    
    return edges


def detect_hmi_screen(image):
    """
    Phát hiện và trích xuất vùng màn hình HMI từ ảnh
    Sử dụng thuật toán GIỐNG Y CHANG như cut_img.py (option 1)
    
    Returns:
        tuple: (extracted_screen, processing_time)
    """
    import time
    import tempfile
    import os
    start_time = time.time()
    
    try:
        if image is None or len(image.shape) != 3:
            return None, time.time() - start_time
        
        # Import function find_hmi_in_image từ hmi_image_detector.py - GIỐNG Y CHANG CUT_IMG.PY
        from hmi_image_detector import find_hmi_in_image
        
        # Tạo file tạm để lưu ảnh với chất lượng cao (giống như cut_img.py)
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "temp_image.png")
        
        # Lưu ảnh với định dạng PNG để tránh mất mát dữ liệu
        cv2.imwrite(temp_image_path, image)
        
        # Gọi function find_hmi_in_image GIỐNG Y CHANG như cut_img.py
        detected_hmis, refined_hmis = find_hmi_in_image(temp_image_path, temp_dir)
        
        # Dọn dẹp file tạm
        try:
            os.remove(temp_image_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        if refined_hmis and len(refined_hmis) > 0:
            # Lấy màn hình HMI đầu tiên (đã được tinh chỉnh) - GIỐNG Y CHANG CUT_IMG.PY
            hmi_screen = refined_hmis[0][0]  # warped_roi
            processing_time = time.time() - start_time
            return hmi_screen, processing_time
        else:
            return None, time.time() - start_time
            
    except Exception as e:
        print(f"Error in detect_hmi_screen: {e}")
        return None, time.time() - start_time


def extract_content_region(img):
    """Trích xuất vùng nội dung chính từ ảnh"""
    # Simplified version
    return img


def fine_tune_hmi_screen(image, roi_coords):
    """Fine-tune HMI screen detection"""
    # Simplified version
    return image

