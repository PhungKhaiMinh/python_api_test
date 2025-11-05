"""
Config Manager Module
Handles configuration and ROI data management
"""

import os
import json
import traceback
from .cache_manager import get_roi_info_cached


def get_roi_data_folder():
    """Get ROI data folder path"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'roi_data')


def get_roi_coordinates(machine_code, screen_id=None, machine_type=None):
    """
    Lấy tọa độ ROI và tên ROI cho một máy và màn hình cụ thể
    
    Returns:
        tuple: (roi_coordinates, roi_names)
    """
    try:
        roi_data = get_roi_info_cached()
        
        # Nếu machine_type không được cung cấp, lấy từ machine_screens.json
        if not machine_type:
            machine_type = get_machine_type(machine_code)
            if not machine_type:
                print(f"Could not determine machine_type for machine_code: {machine_code}")
                return [], []
            print(f"Determined machine_type: {machine_type} for machine_code: {machine_code}")
        
        screen_name = None
        
        # Kiểm tra xem screen_id có phải là tên màn hình không
        if isinstance(screen_id, str) and screen_id in ["Production_Data", "Faults", "Feeders_and_Conveyors", 
                                                  "Main_Machine_Parameters", "Selectors_and_Maintenance",
                                                  "Setting", "Temp", "Plasticizer", "Overview", "Tracking", "Production", 
                                                  "Clamp", "Ejector", "Injection", "Reject_Summary"]:
            screen_name = screen_id
            print(f"Using screen_id as screen_name: {screen_name}")
        elif screen_id is not None:
            # Lấy tên màn hình từ screen_id (nếu là numeric id)
            machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
            
            with open(machine_screens_path, 'r', encoding='utf-8') as f:
                machine_screens = json.load(f)
            
            # Tìm trong areas
            for area_code, area_info in machine_screens.get("areas", {}).items():
                machines = area_info.get("machines", {})
                if machine_code in machines:
                    for screen in machines[machine_code].get("screens", []):
                        if str(screen.get("id")) == str(screen_id):
                            screen_name = screen.get("screen_id")
                            print(f"Found screen_name: {screen_name} for screen_id: {screen_id}")
                            break
        
        print(f"Looking for ROIs in machine_type: {machine_type}, screen: {screen_name}")
        
        # Tìm trong machines
        if machine_type in roi_data.get("machines", {}):
            screens_data = roi_data["machines"][machine_type].get("screens", {})
            
            if screen_name and screen_name in screens_data:
                roi_list = screens_data[screen_name]
                
                # Xử lý định dạng ROI
                roi_coordinates = []
                roi_names = []
                
                for roi_item in roi_list:
                    if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                        roi_coordinates.append(roi_item["coordinates"])
                        roi_names.append(roi_item["name"])
                    else:
                        roi_coordinates.append(roi_item)
                        roi_names.append(f"ROI_{len(roi_names)}")
                
                print(f"Found {len(roi_coordinates)} ROIs for {screen_name}")
                return roi_coordinates, roi_names
            else:
                print(f"Screen '{screen_name}' not found in roi_info.json")
        else:
            print(f"Machine type '{machine_type}' not found in roi_info.json")
        
        return [], []
    except Exception as e:
        print(f"Error reading ROI coordinates: {str(e)}")
        traceback.print_exc()
        return [], []


def get_roi_coordinates_with_subpage(machine_type, screen_name, sub_page=None, machine_code=None):
    """
    Lấy danh sách tọa độ ROI từ roi_info.json với hỗ trợ sub-pages
    
    Args:
        machine_type: Loại máy (F1, F41, F42)
        screen_name: Tên màn hình
        sub_page: Sub-page (1, 2) cho màn hình có nhiều trang
        machine_code: Mã máy cụ thể (cần cho cấu trúc mới với Reject Summary)
        
    Returns:
        tuple: (roi_coordinates, roi_names)
    """
    try:
        roi_data = get_roi_info_cached()
        
        print(f"Looking for ROIs in machine_type: {machine_type}, screen: {screen_name}, sub_page: {sub_page}, machine_code: {machine_code}")
        
        # Tìm trong machines
        if machine_type in roi_data.get("machines", {}):
            screens_data = roi_data["machines"][machine_type].get("screens", {})
            
            if screen_name and screen_name in screens_data:
                screen_data = screens_data[screen_name]
                
                # Kiểm tra cấu trúc mới (với sub_pages)
                if isinstance(screen_data, dict) and "sub_pages" in screen_data:
                    # Cấu trúc mới: sub_pages > machine_code > sub_page > rois
                    if not machine_code:
                        print("[ERROR] machine_code is required for new sub_pages structure")
                        return [], []
                    
                    if machine_code not in screen_data["sub_pages"]:
                        print(f"Machine code '{machine_code}' not found in sub_pages for screen '{screen_name}'")
                        return [], []
                    
                    if sub_page not in screen_data["sub_pages"][machine_code]:
                        print(f"Sub-page '{sub_page}' not found for machine '{machine_code}', screen '{screen_name}'")
                        return [], []
                    
                    sub_page_data = screen_data["sub_pages"][machine_code][sub_page]
                    roi_list = sub_page_data.get("rois", [])
                    
                    roi_coordinates = []
                    roi_names = []
                    
                    for roi_item in roi_list:
                        if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                            roi_coordinates.append(roi_item["coordinates"])
                            roi_names.append(roi_item["name"])
                        elif isinstance(roi_item, (list, tuple)) and len(roi_item) == 4:
                            roi_coordinates.append(roi_item)
                            roi_names.append(f"ROI_{len(roi_names)}")
                    
                    print(f"Found {len(roi_coordinates)} ROIs for {screen_name} sub-page {sub_page} (new structure)")
                    return roi_coordinates, roi_names
                else:
                    # Cấu trúc cũ (không có sub_pages)
                    roi_list = screen_data
                    
                    # Lọc theo sub_page nếu có
                    if sub_page:
                        roi_list = [roi for roi in roi_list 
                                   if isinstance(roi, dict) and roi.get("sub_page") == str(sub_page)]
                    
                    # Xử lý định dạng ROI
                    roi_coordinates = []
                    roi_names = []
                    
                    for roi_item in roi_list:
                        if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                            roi_coordinates.append(roi_item["coordinates"])
                            roi_names.append(roi_item["name"])
                        else:
                            roi_coordinates.append(roi_item)
                            roi_names.append(f"ROI_{len(roi_names)}")
                    
                    print(f"Found {len(roi_coordinates)} ROIs for {screen_name} (sub_page: {sub_page}) (old structure)")
                    return roi_coordinates, roi_names
            else:
                print(f"Screen '{screen_name}' not found in roi_info.json")
        else:
            print(f"Machine type '{machine_type}' not found in roi_info.json")
        
        return [], []
    except Exception as e:
        print(f"Error reading ROI coordinates: {str(e)}")
        traceback.print_exc()
        return [], []


def get_special_region_coordinates(machine_type, machine_code, screen_name, sub_page=None):
    """
    Lấy tọa độ vùng đặc biệt để phân biệt sub-page
    
    Args:
        machine_type: Loại máy (F1, F41, F42)
        machine_code: Mã máy cụ thể (IE-F1-CWA01)
        screen_name: Tên màn hình
        sub_page: Sub-page (nếu có)
        
    Returns:
        list: Danh sách tọa độ vùng đặc biệt
    """
    try:
        roi_data = get_roi_info_cached()
        
        if machine_type not in roi_data.get("machines", {}):
            print(f"Machine type '{machine_type}' not found")
            return []
        
        screens_data = roi_data["machines"][machine_type].get("screens", {})
        
        if screen_name not in screens_data:
            print(f"Screen '{screen_name}' not found")
            return []
        
        screen_data = screens_data[screen_name]
        
        # Kiểm tra cấu trúc mới (với sub_pages)
        if isinstance(screen_data, dict) and "sub_pages" in screen_data:
            # Cấu trúc mới: sub_pages > machine_code > sub_page > special_region
            if machine_code not in screen_data["sub_pages"]:
                print(f"Machine code '{machine_code}' not found in sub_pages")
                return []
            
            if sub_page:
                # Lấy special_region của sub-page cụ thể
                if sub_page in screen_data["sub_pages"][machine_code]:
                    regions = screen_data["sub_pages"][machine_code][sub_page].get("special_region", [])
                    print(f"Found {len(regions)} special regions for {machine_code}/{screen_name}/sub-page {sub_page} (new structure)")
                    return regions
                else:
                    print(f"Sub-page '{sub_page}' not found for {machine_code}/{screen_name}")
                    return []
            else:
                # Lấy special_region của tất cả sub-pages (để OCR xác định sub-page)
                # Lấy sub-page 1 (mặc định để OCR)
                for sp in ["1", "2"]:
                    if sp in screen_data["sub_pages"][machine_code]:
                        regions = screen_data["sub_pages"][machine_code][sp].get("special_region", [])
                        if regions:
                            print(f"Found {len(regions)} special regions for {machine_code}/{screen_name}/sub-page {sp} (for detection)")
                            return regions
                print(f"No special regions found for any sub-page of {machine_code}/{screen_name}")
                return []
        else:
            # Cấu trúc cũ (special_regions riêng)
            if ("special_regions" in roi_data["machines"][machine_type] and
                machine_code in roi_data["machines"][machine_type]["special_regions"] and
                screen_name in roi_data["machines"][machine_type]["special_regions"][machine_code]):
                
                regions = roi_data["machines"][machine_type]["special_regions"][machine_code][screen_name]
                print(f"Found {len(regions)} special regions for {machine_code}/{screen_name} (old structure)")
                return regions
            
            print(f"No special regions found for {machine_code}/{screen_name}")
            return []
    except Exception as e:
        print(f"Error reading special region coordinates: {str(e)}")
        traceback.print_exc()
        return []


def get_machine_type(machine_code):
    """
    Lấy loại máy (machine_type) từ mã máy (machine_code)
    
    Returns:
        str: Loại máy (F1, F41, F42, ...) hoặc None
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm kiếm trong cấu trúc areas
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            if machine_code in machines:
                return machines[machine_code].get('type')
        
        return None
    except Exception as e:
        print(f"Error getting machine type: {str(e)}")
        return None


def get_area_for_machine(machine_code):
    """
    Lấy khu vực (area) chứa mã máy (machine_code)
    
    Returns:
        str: Mã khu vực hoặc None
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            if machine_code in machines:
                return area_code
        
        return None
    except Exception as e:
        print(f"Error getting area for machine: {str(e)}")
        return None


def get_machine_name_from_code(machine_code):
    """
    Lấy tên máy từ mã máy
    
    Returns:
        str: Tên máy hoặc None
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            if machine_code in machines:
                machine_info = machines[machine_code]
                return machine_info.get('name')
        
        return None
    except Exception as e:
        print(f"Error getting machine name from code: {str(e)}")
        return None


def get_all_machine_types():
    """
    Lấy tất cả machine_type có sẵn
    
    Returns:
        list: Danh sách các machine_type duy nhất
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return []
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        machine_types = set()
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            for machine_code, machine_info in machines.items():
                machine_type = machine_info.get('type')
                if machine_type:
                    machine_types.add(machine_type)
        
        return list(machine_types)
    except Exception as e:
        print(f"Error getting all machine types: {str(e)}")
        return []


def find_machine_code_from_template(template_filename):
    """
    Tìm machine_code từ template filename
    
    Returns:
        tuple: (machine_code, area) hoặc (None, None)
    """
    try:
        # Format: template_{machine_type}_{screen_name}.ext
        parts = template_filename.split('_')
        if len(parts) < 3:
            return None, None
        
        machine_type = parts[1]
        
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None, None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm machine_code đầu tiên có machine_type tương ứng
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            for machine_code, machine_info in machines.items():
                if machine_info.get('type') == machine_type:
                    return machine_code, area_code
        
        return None, None
    except Exception as e:
        print(f"Error finding machine code from template: {str(e)}")
        return None, None


def get_screen_numeric_id(machine_type, screen_name):
    """
    Lấy numeric ID của màn hình từ machine_type và screen_name
    
    Returns:
        int: Screen numeric ID hoặc None
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tìm trong machine_types
        if 'machine_types' in data and machine_type in data['machine_types']:
            screens = data['machine_types'][machine_type].get('screens', [])
            for screen in screens:
                if screen.get('screen_id') == screen_name:
                    return screen.get('id')
        
        return None
    except Exception as e:
        print(f"Error getting screen numeric id: {str(e)}")
        return None


def get_reference_template_path(machine_type, screen_id):
    """
    Tìm đường dẫn template ảnh tham chiếu cho machine_type và screen_id
    
    Returns:
        str: Đường dẫn template hoặc None
    """
    try:
        reference_folder = os.path.join(get_roi_data_folder(), 'reference_images')
        if not os.path.exists(reference_folder):
            return None
        
        # Tìm file template có format: template_{machine_type}_{screen_id}.jpg
        template_filename = f"template_{machine_type}_{screen_id}.jpg"
        template_path = os.path.join(reference_folder, template_filename)
        
        if os.path.exists(template_path):
            return template_path
        
        # Thử với extension .png
        template_filename = f"template_{machine_type}_{screen_id}.png"
        template_path = os.path.join(reference_folder, template_filename)
        
        if os.path.exists(template_path):
            return template_path
        
        return None
    except Exception as e:
        print(f"Error getting reference template path: {str(e)}")
        return None


def get_reference_template_path_with_subpage(machine_type, screen_id, sub_page=None):
    """
    Tìm đường dẫn template ảnh tham chiếu cho machine_type, screen_id và sub_page
    
    Args:
        machine_type: Loại máy (F1, F41, F42)
        screen_id: Tên màn hình
        sub_page: Sub-page (1, 2) cho màn hình có nhiều trang
        
    Returns:
        str: Đường dẫn template hoặc None
    """
    try:
        reference_folder = os.path.join(get_roi_data_folder(), 'reference_images')
        if not os.path.exists(reference_folder):
            return None
        
        # Nếu có sub_page, tìm template riêng cho từng trang
        if sub_page:
            template_filename = f"template_{machine_type}_{screen_id}_page{sub_page}.jpg"
            template_path = os.path.join(reference_folder, template_filename)
            
            if os.path.exists(template_path):
                return template_path
            
            # Thử với extension .png
            template_filename = f"template_{machine_type}_{screen_id}_page{sub_page}.png"
            template_path = os.path.join(reference_folder, template_filename)
            
            if os.path.exists(template_path):
                return template_path
        
        # Fallback: tìm template chung (không có sub_page)
        return get_reference_template_path(machine_type, screen_id)
        
    except Exception as e:
        print(f"Error getting reference template path with subpage: {str(e)}")
        return None


def get_decimal_places_config():
    """Đọc cấu hình số chữ số thập phân từ file"""
    decimal_config_path = os.path.join(get_roi_data_folder(), 'decimal_places.json')
    if os.path.exists(decimal_config_path):
        try:
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading decimal places config: {str(e)}")
    return {}


def is_named_roi_format(roi_list):
    """
    Kiểm tra xem ROI list có dùng format named hay không
    """
    if roi_list and len(roi_list) > 0:
        first_item = roi_list[0]
        return isinstance(first_item, dict) and "name" in first_item and "coordinates" in first_item
    return False


def get_current_machine_info():
    """
    Lấy thông tin máy hiện tại từ file current_machine_screen.json
    
    Returns:
        dict: Machine info hoặc default values
    """
    try:
        current_machine_file = os.path.join(os.path.dirname(get_roi_data_folder()), 'current_machine_screen.json')
        if os.path.exists(current_machine_file):
            with open(current_machine_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"machine_code": "F41", "screen_id": "Main"}
    except Exception as e:
        print(f"Error getting current machine info: {str(e)}")
        return {"machine_code": "F41", "screen_id": "Main"}


def save_current_machine_info(machine_code, screen_id):
    """
    Lưu thông tin máy hiện tại vào file
    """
    try:
        current_machine_file = os.path.join(os.path.dirname(get_roi_data_folder()), 'current_machine_screen.json')
        data = {
            "machine_code": machine_code,
            "screen_id": screen_id
        }
        with open(current_machine_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Clear cache để force reload
        from .cache_manager import clear_cache
        clear_cache('machine')
        
        return True
    except Exception as e:
        print(f"Error saving current machine info: {str(e)}")
        return False

