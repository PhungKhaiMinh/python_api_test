"""
Config Manager Module
Handles configuration and ROI data management for PaddleOCR workflow
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
    Get ROI coordinates and names for a machine and screen
    
    Structure: machines > machine_type > machine_code > screens > screen_name > sub_pages > "1" > Rois
    
    Returns:
        tuple: (roi_coordinates, roi_names)
    """
    try:
        roi_data = get_roi_info_cached()
        
        if not machine_type:
            machine_type = get_machine_type(machine_code)
            if not machine_type:
                print(f"[WARN] Could not determine machine_type for machine_code: {machine_code}")
                return [], []
        
        screen_name = None
        
        if isinstance(screen_id, str) and screen_id in ["Production_Data", "Faults", "Feeders_and_Conveyors", 
                                                  "Main_Machine_Parameters", "Selectors_and_Maintenance",
                                                  "Setting", "Temp", "Plasticizer", "Overview", "Tracking", "Production", 
                                                  "Clamp", "Ejector", "Injection", "Reject_Summary"]:
            screen_name = screen_id
        elif screen_id is not None:
            machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
            
            with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
                machine_screens = json.load(f)
            
            for area_code, area_info in machine_screens.get("areas", {}).items():
                machines = area_info.get("machines", {})
                if machine_code in machines:
                    for screen in machines[machine_code].get("screens", []):
                        if str(screen.get("id")) == str(screen_id):
                            screen_name = screen.get("screen_id")
                            break
        
        print(f"[*] Looking for ROIs: machine_type={machine_type}, machine_code={machine_code}, screen={screen_name}")
        
        if machine_type in roi_data.get("machines", {}):
            machine_type_data = roi_data["machines"][machine_type]
            
            if machine_code and machine_code in machine_type_data:
                machine_data = machine_type_data[machine_code]
                screens_data = machine_data.get("screens", {})
            
                if screen_name and screen_name in screens_data:
                    screen_data = screens_data[screen_name]
                
                    if isinstance(screen_data, dict) and "sub_pages" in screen_data:
                        if "1" in screen_data["sub_pages"]:
                            sub_page_data = screen_data["sub_pages"]["1"]
                            roi_list = sub_page_data.get("Rois", sub_page_data.get("rois", []))
                            
                            roi_coordinates = []
                            roi_names = []
                            
                            for roi_item in roi_list:
                                if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                                    roi_coordinates.append(roi_item["coordinates"])
                                    roi_names.append(roi_item["name"])
                            
                            print(f"[OK] Found {len(roi_coordinates)} ROIs for {machine_code}/{screen_name}")
                            return roi_coordinates, roi_names
            
            if "screens" in machine_type_data:
                screens_data = machine_type_data.get("screens", {})
                
                if screen_name and screen_name in screens_data:
                    screen_data = screens_data[screen_name]
                    
                    if isinstance(screen_data, list):
                        roi_coordinates = []
                        roi_names = []
                        
                        for roi_item in screen_data:
                            if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                                roi_coordinates.append(roi_item["coordinates"])
                                roi_names.append(roi_item["name"])
                        
                        print(f"[OK] Found {len(roi_coordinates)} ROIs (legacy structure)")
                        return roi_coordinates, roi_names
                    elif isinstance(screen_data, dict) and "sub_pages" in screen_data:
                        if "1" in screen_data["sub_pages"]:
                            sub_page_data = screen_data["sub_pages"]["1"]
                            roi_list = sub_page_data.get("Rois", sub_page_data.get("rois", []))
                            
                            roi_coordinates = []
                            roi_names = []
                            
                            for roi_item in roi_list:
                                if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                                    roi_coordinates.append(roi_item["coordinates"])
                                    roi_names.append(roi_item["name"])
                
                            print(f"[OK] Found {len(roi_coordinates)} ROIs (old sub_pages structure)")
                return roi_coordinates, roi_names
        
        print(f"[WARN] No ROIs found for {machine_type}/{machine_code}/{screen_name}")
        return [], []
        
    except Exception as e:
        print(f"[ERROR] get_roi_coordinates: {str(e)}")
        traceback.print_exc()
        return [], []


def get_roi_coordinates_with_subpage(machine_type, screen_name, sub_page=None, machine_code=None):
    """
    Get ROI coordinates from roi_info.json with sub-page support
        
    Returns:
        tuple: (roi_coordinates, roi_names)
    """
    try:
        roi_data = get_roi_info_cached()
        
        print(f"[*] Looking for ROIs: machine_type={machine_type}, screen={screen_name}, sub_page={sub_page}, machine_code={machine_code}")
        
        if machine_type in roi_data.get("machines", {}):
            machine_type_data = roi_data["machines"][machine_type]
            
            if machine_code and machine_code in machine_type_data:
                machine_data = machine_type_data[machine_code]
                screens_data = machine_data.get("screens", {})
            
                if screen_name in screens_data:
                    screen_data = screens_data[screen_name]
                
                    if isinstance(screen_data, dict) and "sub_pages" in screen_data:
                        sub_page_key = str(sub_page) if sub_page else "1"
                    
                        if sub_page_key in screen_data["sub_pages"]:
                            sub_page_data = screen_data["sub_pages"][sub_page_key]
                            roi_list = sub_page_data.get("Rois", sub_page_data.get("rois", []))
                    
                            roi_coordinates = []
                            roi_names = []
                    
                            for roi_item in roi_list:
                                if isinstance(roi_item, dict) and "name" in roi_item and "coordinates" in roi_item:
                                    roi_coordinates.append(roi_item["coordinates"])
                                    roi_names.append(roi_item["name"])
                                elif isinstance(roi_item, (list, tuple)) and len(roi_item) == 4:
                                    roi_coordinates.append(roi_item)
                                    roi_names.append(f"ROI_{len(roi_names)}")
                    
                            print(f"[OK] Found {len(roi_coordinates)} ROIs for {machine_code}/{screen_name} sub-page {sub_page_key}")
                            return roi_coordinates, roi_names
        
        print(f"[WARN] No ROIs found for {machine_type}/{machine_code}/{screen_name}")
        return [], []
        
    except Exception as e:
        print(f"[ERROR] get_roi_coordinates_with_subpage: {str(e)}")
        traceback.print_exc()
        return [], []


def get_machine_type(machine_code):
    """
    Get machine_type from machine_code
    
    Returns:
        str: Machine type (F1, F41, F42, ...) or None
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
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
    Get area for machine_code
    
    Returns:
        str: Area code or None
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
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
    Get machine name from machine_code
    
    Returns:
        str: Machine name or None
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return None
        
        with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
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
    Get all available machine_types
    
    Returns:
        list: List of unique machine_types
    """
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return []
        
        with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
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


def get_decimal_places_config():
    """Read decimal places configuration from file"""
    decimal_config_path = os.path.join(get_roi_data_folder(), 'decimal_places.json')
    if os.path.exists(decimal_config_path):
        try:
            with open(decimal_config_path, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading decimal places config: {str(e)}")
    return {}


def is_named_roi_format(roi_list):
    """Check if ROI list uses named format"""
    if roi_list and len(roi_list) > 0:
        first_item = roi_list[0]
        return isinstance(first_item, dict) and "name" in first_item and "coordinates" in first_item
    return False


def get_current_machine_info():
    """
    Get current machine info from file
    
    Returns:
        dict: Machine info or default values
    """
    try:
        current_machine_file = os.path.join(os.path.dirname(get_roi_data_folder()), 'current_machine_screen.json')
        if os.path.exists(current_machine_file):
            with open(current_machine_file, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        return {"machine_code": "F41", "screen_id": "Main"}
    except Exception as e:
        print(f"Error getting current machine info: {str(e)}")
        return {"machine_code": "F41", "screen_id": "Main"}


def save_current_machine_info(machine_code, screen_id):
    """
    Save current machine info to file
    """
    try:
        current_machine_file = os.path.join(os.path.dirname(get_roi_data_folder()), 'current_machine_screen.json')
        data = {
            "machine_code": machine_code,
            "screen_id": screen_id
        }
        with open(current_machine_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        from .cache_manager import clear_cache
        clear_cache('machine')
        
        return True
    except Exception as e:
        print(f"Error saving current machine info: {str(e)}")
        return False
