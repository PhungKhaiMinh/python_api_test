import cv2
import numpy as np
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim

def get_machine_type_from_config_smart(area, machine_code):
    """
    Lấy machine type từ machine_screens.json dựa trên area và machine_code
    """
    try:
        # Sử dụng đường dẫn tuyệt đối dựa trên vị trí file hiện tại
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'roi_data', 'machine_screens.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if area in config['areas'] and machine_code in config['areas'][area]['machines']:
            machine_type = config['areas'][area]['machines'][machine_code]['type']
            machine_name = config['areas'][area]['machines'][machine_code]['name']
            return machine_type, machine_name
        
        return None, None
    except Exception as e:
        print(f"Error reading machine config: {e}")
        return None, None

def get_reference_templates_for_type_smart(machine_type):
    """
    Lấy danh sách tất cả reference templates cho một machine type cụ thể
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reference_dir = os.path.join(current_dir, 'roi_data', 'reference_images')
    templates = []
    
    if not os.path.exists(reference_dir):
        return templates
    
    # Tìm tất cả file template cho machine type này
    for filename in os.listdir(reference_dir):
        if filename.startswith(f"template_{machine_type}_") and filename.endswith(('.png', '.jpg')):
            template_path = os.path.join(reference_dir, filename)
            # Extract screen_id from filename: template_F41_Clamp.jpg -> Clamp
            screen_id = filename.replace(f"template_{machine_type}_", "").replace(".jpg", "").replace(".png", "")
            templates.append({
                'path': template_path,
                'filename': filename,
                'machine_type': machine_type,
                'screen_id': screen_id
            })
    
    return templates

def enhanced_template_matching_smart(image, template, scales=[0.9, 1.0, 1.1]):
    """
    Template matching với multiple scales và edge enhancement
    """
    best_score = 0
    best_location = None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
        
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    
    # Edge enhancement
    image_edges = cv2.Canny(image_gray, 50, 150)
    template_edges = cv2.Canny(template_gray, 50, 150)
    
    for scale in scales:
        # Resize template
        if scale != 1.0:
            h, w = template_edges.shape
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h > 0 and new_w > 0:
                scaled_template = cv2.resize(template_edges, (new_w, new_h))
            else:
                continue
        else:
            scaled_template = template_edges
        
        # Template matching
        if scaled_template.shape[0] <= image_edges.shape[0] and scaled_template.shape[1] <= image_edges.shape[1]:
            result = cv2.matchTemplate(image_edges, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_location = max_loc
    
    return best_score, best_location

def calculate_advanced_similarity_smart(image, template):
    """
    Tính toán similarity score nâng cao với multiple metrics
    """
    # Resize images to same size for comparison
    h, w = template.shape[:2]
    image_resized = cv2.resize(image, (w, h))
    
    # Convert to grayscale if needed
    if len(image_resized.shape) == 3:
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_resized
        
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    
    # 1. Enhanced template matching
    template_score, _ = enhanced_template_matching_smart(image, template)
    
    # 2. SSIM with edge enhancement
    ssim_score = ssim(image_gray, template_gray)
    
    # 3. Histogram comparison
    hist1 = cv2.calcHist([image_gray], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([template_gray], [0], None, [64], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 4. Feature matching (simplified)
    try:
        orb = cv2.ORB_create(nfeatures=100)
        kp1, des1 = orb.detectAndCompute(image_gray, None)
        kp2, des2 = orb.detectAndCompute(template_gray, None)
        
        feature_score = 0
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) > 0:
                feature_score = min(len(matches) / max(len(kp1), len(kp2)), 1.0)
    except:
        feature_score = 0
    
    # Weighted combination
    weights = {
        'template': 0.4,
        'ssim': 0.3,
        'histogram': 0.2,
        'features': 0.1
    }
    
    final_score = (
        weights['template'] * template_score +
        weights['ssim'] * max(0, ssim_score) +
        weights['histogram'] * max(0, hist_score) +
        weights['features'] * feature_score
    )
    
    return final_score

def auto_detect_machine_and_screen_smart(image, area=None, machine_code=None):
    """
    Thuật toán auto detection thông minh với khả năng sử dụng thông tin area và machine_code
    
    Args:
        image: Ảnh input
        area: Khu vực máy (optional) - nếu có sẽ giới hạn phạm vi tìm kiếm
        machine_code: Mã máy (optional) - nếu có sẽ xác định chính xác machine type
    
    Returns:
        Dict chứa thông tin detection hoặc None nếu không tìm thấy
    """
    start_time = time.time()
    print(f"🚀 Starting SMART detection with area={area}, machine_code={machine_code}")
    
    # Nếu có đầy đủ area và machine_code, xác định machine type trước
    target_machine_type = None
    target_machine_name = None
    
    if area and machine_code:
        target_machine_type, target_machine_name = get_machine_type_from_config_smart(area, machine_code)
        if target_machine_type:
            print(f"✅ Determined machine type: {target_machine_type} from config")
        else:
            print(f"⚠️ Could not find machine type for area={area}, machine_code={machine_code}")
    
    # Xác định danh sách templates cần kiểm tra
    candidates = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reference_dir = os.path.join(current_dir, 'roi_data', 'reference_images')
    
    if not os.path.exists(reference_dir):
        print(f"❌ Reference directory not found: {reference_dir}")
        return None
    
    if target_machine_type:
        # Chỉ kiểm tra templates của machine type đã xác định
        print(f"🎯 Searching only in {target_machine_type} templates")
        templates = get_reference_templates_for_type_smart(target_machine_type)
        candidates = templates
    else:
        # Kiểm tra tất cả templates nhưng ưu tiên theo area nếu có
        print("🔍 Searching in all templates")
        for filename in os.listdir(reference_dir):
            if filename.startswith('template_') and filename.endswith(('.png', '.jpg')):
                # Parse filename to get machine info: template_F41_Clamp.jpg
                parts = filename.replace('template_', '').replace('.jpg', '').replace('.png', '').split('_')
                if len(parts) >= 2:
                    file_machine_type = parts[0]
                    file_screen_id = '_'.join(parts[1:])
                    
                    template_path = os.path.join(reference_dir, filename)
                    candidates.append({
                        'path': template_path,
                        'filename': filename,
                        'machine_type': file_machine_type,
                        'screen_id': file_screen_id
                    })
    
    if not candidates:
        print("❌ No template candidates found")
        return None
    
    print(f"📋 Found {len(candidates)} template candidates")
    
    # Sắp xếp candidates theo độ ưu tiên
    if area:
        # Ưu tiên templates của area được chỉ định
        area_priority_map = {'F1': 1, 'F4': 2}
        area_priority = area_priority_map.get(area, 999)
        
        def get_priority(candidate):
            machine_type = candidate['machine_type']
            if machine_type.startswith('F1'):
                return 1 if area == 'F1' else 3
            elif machine_type.startswith('F4'):
                return 2 if area == 'F4' else 3
            else:
                return 4
        
        candidates.sort(key=get_priority)
    
    # Parallel processing với early termination
    best_result = None
    best_score = 0
    high_confidence_threshold = 0.8
    
    def process_template(candidate):
        try:
            template_path = candidate['path']
            template = cv2.imread(template_path)
            
            if template is None:
                return None
            
            # Tính similarity score
            similarity_score = calculate_advanced_similarity_smart(image, template)
            
            return {
                'candidate': candidate,
                'similarity_score': similarity_score,
                'template_path': template_path
            }
        except Exception as e:
            print(f"Error processing template {candidate['filename']}: {e}")
            return None
    
    # Process templates in batches với early termination
    batch_size = 4
    max_workers = 3
    
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_template, candidate) for candidate in batch]
            
            for future in as_completed(futures):
                result = future.result()
                if result and result['similarity_score'] > best_score:
                    best_score = result['similarity_score']
                    best_result = result
                    
                    print(f"🎯 New best: {result['candidate']['filename']} - Score: {best_score:.4f}")
                    
                    # Early termination nếu đạt high confidence
                    if best_score >= high_confidence_threshold:
                        print(f"✅ High confidence achieved: {best_score:.4f}")
                        break
        
        # Break khỏi batch loop nếu đã đạt high confidence
        if best_score >= high_confidence_threshold:
            break
        
        # Time limit check
        if time.time() - start_time > 8:  # 8 second limit
            print("⏰ Time limit reached, using best result so far")
            break
    
    if not best_result:
        print("❌ No matching template found")
        return None
    
    # Xây dựng kết quả
    candidate = best_result['candidate']
    machine_type = candidate['machine_type']
    screen_id = candidate['screen_id']
    
    # Xác định area và machine_code
    if area and machine_code:
        # Sử dụng thông tin đã có
        result_area = area
        result_machine_code = machine_code
        result_machine_name = target_machine_name or f"Máy {machine_code}"
    else:
        # Tìm từ config dựa trên machine_type
        result_area = None
        result_machine_code = None
        result_machine_name = None
        
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'roi_data', 'machine_screens.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Tìm machine đầu tiên có type phù hợp
            for area_key, area_data in config['areas'].items():
                for machine_key, machine_data in area_data['machines'].items():
                    if machine_data.get('type') == machine_type:
                        result_area = area_key
                        result_machine_code = machine_key
                        result_machine_name = machine_data.get('name', f"Máy {machine_key}")
                        break
                if result_area:
                    break
        except Exception as e:
            print(f"Error reading config for area/machine_code: {e}")
    
    # Tìm screen_numeric_id
    screen_numeric_id = None
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'roi_data', 'machine_screens.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if machine_type in config['machine_types']:
            for screen in config['machine_types'][machine_type]['screens']:
                if screen['screen_id'] == screen_id:
                    screen_numeric_id = screen['id']
                    break
    except Exception as e:
        print(f"Error finding screen_numeric_id: {e}")
    
    processing_time = time.time() - start_time
    
    result = {
        'machine_code': result_machine_code or 'UNKNOWN',
        'machine_type': machine_type,
        'area': result_area or 'UNKNOWN',
        'machine_name': result_machine_name or f"Máy {machine_type}",
        'screen_id': screen_id,
        'screen_numeric_id': screen_numeric_id,
        'template_path': best_result['template_path'],
        'similarity_score': best_score,
        'processing_time': processing_time,
        'detection_method': 'smart_detection_v1.0'
    }
    
    print(f"✅ SMART Detection completed in {processing_time:.2f}s")
    print(f"   Result: {machine_type} - {screen_id} (Score: {best_score:.4f})")
    
    return result 