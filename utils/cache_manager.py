"""
Cache Manager Module
Handles all caching logic for ROI info, decimal places, machine info, and template images
"""

import os
import json
import cv2
import threading

# Global cache variables
_roi_info_cache = None
_roi_info_cache_lock = threading.Lock()
_decimal_places_cache = None
_decimal_places_cache_lock = threading.Lock()
_machine_info_cache = None
_machine_info_cache_lock = threading.Lock()
_template_image_cache = {}
_template_cache_lock = threading.Lock()


def initialize_all_caches():
    """Khởi tạo tất cả cache ngay khi chương trình bắt đầu để tránh delay lần đầu gọi API"""
    print("\n[*] Initializing all caches at startup...")
    
    try:
        # Cache ROI info
        roi_info = get_roi_info_cached()
        print(f"[OK] ROI info cached: {len(roi_info)} items")
    except Exception as e:
        print(f"[ERROR] Error caching ROI info: {e}")
    
    try:
        # Cache decimal places config
        decimal_config = get_decimal_places_config_cached()
        print(f"[OK] Decimal places config cached: {len(decimal_config)} items")
    except Exception as e:
        print(f"[ERROR] Error caching decimal places config: {e}")
    
    try:
        # Cache machine info
        machine_info = get_machine_info_cached()
        print(f"[OK] Machine info cached: {machine_info}")
    except Exception as e:
        print(f"[ERROR] Error caching machine info: {e}")
    
    try:
        # Pre-cache common template images từ reference_images folder
        reference_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'roi_data', 'reference_images')
        if os.path.exists(reference_folder):
            template_files = [f for f in os.listdir(reference_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            cached_count = 0
            for template_file in template_files[:]:  # Cache tất cả template
                template_path = os.path.join(reference_folder, template_file)
                if get_template_image_cached(template_path) is not None:
                    cached_count += 1
            print(f"[OK] Template images pre-cached: {cached_count}/{len(template_files)} files")
        else:
            print("ℹ️  Reference images folder not found - skipping template pre-caching")
    except Exception as e:
        print(f"[ERROR] Error pre-caching template images: {e}")
    
    print("🎯 Cache initialization completed!\n")


def get_template_image_cached(template_path):
    """Cache template images để tránh đọc lại từ disk"""
    if not template_path or not os.path.exists(template_path):
        return None
        
    with _template_cache_lock:
        if template_path not in _template_image_cache:
            try:
                template_img = cv2.imread(template_path)
                if template_img is not None:
                    _template_image_cache[template_path] = template_img
                    print(f"[OK] Template image cached: {os.path.basename(template_path)}")
                return template_img
            except Exception as e:
                print(f"[ERROR] Error caching template image: {e}")
                return None
        
        return _template_image_cache[template_path]


def get_roi_info_cached():
    """Cache ROI info để tránh đọc file JSON nhiều lần"""
    global _roi_info_cache
    
    with _roi_info_cache_lock:
        if _roi_info_cache is None:
            try:
                roi_json_path = 'roi_data/roi_info.json'
                if not os.path.exists(roi_json_path):
                    roi_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'roi_data/roi_info.json')
                
                with open(roi_json_path, 'r', encoding='utf-8') as f:
                    _roi_info_cache = json.load(f)
                print("[OK] ROI info cached successfully")
            except Exception as e:
                print(f"[ERROR] Error caching ROI info: {e}")
                _roi_info_cache = {}
        
        return _roi_info_cache


def get_decimal_places_config_cached():
    """Cache decimal places config để tránh đọc file nhiều lần"""
    global _decimal_places_cache
    
    with _decimal_places_cache_lock:
        if _decimal_places_cache is None:
            try:
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'roi_data', 'decimal_places.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        _decimal_places_cache = json.load(f)
                else:
                    _decimal_places_cache = {}
                print("[OK] Decimal places config cached successfully")
            except Exception as e:
                print(f"[ERROR] Error caching decimal places config: {e}")
                _decimal_places_cache = {}
        
        return _decimal_places_cache


def get_machine_info_cached():
    """Cache machine info để tránh gọi hàm nặng nhiều lần"""
    global _machine_info_cache
    
    with _machine_info_cache_lock:
        if _machine_info_cache is None:
            try:
                # Đọc từ current machine screen file
                current_machine_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'current_machine_screen.json')
                if os.path.exists(current_machine_file):
                    with open(current_machine_file, 'r', encoding='utf-8') as f:
                        _machine_info_cache = json.load(f)
                else:
                    _machine_info_cache = {"machine_code": "F41", "screen_id": "Main"}
                print("[OK] Machine info cached successfully")
            except Exception as e:
                print(f"[ERROR] Error caching machine info: {e}")
                _machine_info_cache = {"machine_code": "F41", "screen_id": "Main"}
        
        return _machine_info_cache


def clear_cache(cache_type='all'):
    """
    Xóa cache để force reload data mới
    
    Args:
        cache_type: 'all', 'roi', 'decimal', 'machine', 'template'
    """
    global _roi_info_cache, _decimal_places_cache, _machine_info_cache, _template_image_cache
    
    if cache_type in ['all', 'roi']:
        with _roi_info_cache_lock:
            _roi_info_cache = None
            print("[OK] ROI info cache cleared")
    
    if cache_type in ['all', 'decimal']:
        with _decimal_places_cache_lock:
            _decimal_places_cache = None
            print("[OK] Decimal places cache cleared")
    
    if cache_type in ['all', 'machine']:
        with _machine_info_cache_lock:
            _machine_info_cache = None
            print("[OK] Machine info cache cleared")
    
    if cache_type in ['all', 'template']:
        with _template_cache_lock:
            _template_image_cache.clear()
            print("[OK] Template image cache cleared")

