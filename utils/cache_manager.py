"""
Cache Manager Module
Handles all caching logic for ROI info, decimal places, and machine info
PaddleOCR Edition - Template image caching removed
"""

import os
import json
import threading

# Global cache variables
_roi_info_cache = None
_roi_info_cache_lock = threading.Lock()
_decimal_places_cache = None
_decimal_places_cache_lock = threading.Lock()
_machine_info_cache = None
_machine_info_cache_lock = threading.Lock()


def initialize_all_caches():
    """Initialize all caches at startup to avoid delay on first API call"""
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
    
    print("[OK] Cache initialization completed!\n")


def get_roi_info_cached():
    """Cache ROI info to avoid reading JSON file multiple times"""
    global _roi_info_cache
    
    with _roi_info_cache_lock:
        if _roi_info_cache is None:
            try:
                roi_json_path = 'roi_data/roi_info.json'
                if not os.path.exists(roi_json_path):
                    roi_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'roi_data/roi_info.json')
                
                with open(roi_json_path, 'r', encoding='utf-8-sig') as f:
                    _roi_info_cache = json.load(f)
                print("[OK] ROI info cached successfully")
            except Exception as e:
                print(f"[ERROR] Error caching ROI info: {e}")
                _roi_info_cache = {}
        
        return _roi_info_cache


def get_decimal_places_config_cached():
    """Cache decimal places config to avoid reading file multiple times"""
    global _decimal_places_cache
    
    with _decimal_places_cache_lock:
        if _decimal_places_cache is None:
            try:
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'roi_data', 'decimal_places.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8-sig') as f:
                        _decimal_places_cache = json.load(f)
                else:
                    _decimal_places_cache = {}
                print("[OK] Decimal places config cached successfully")
            except Exception as e:
                print(f"[ERROR] Error caching decimal places config: {e}")
                _decimal_places_cache = {}
        
        return _decimal_places_cache


# Alias for compatibility
get_decimal_places_cached = get_decimal_places_config_cached


def get_machine_info_cached():
    """Cache machine info to avoid calling heavy functions multiple times"""
    global _machine_info_cache
    
    with _machine_info_cache_lock:
        if _machine_info_cache is None:
            try:
                # Read from current machine screen file
                current_machine_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'current_machine_screen.json')
                if os.path.exists(current_machine_file):
                    with open(current_machine_file, 'r', encoding='utf-8-sig') as f:
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
    Clear cache to force reload data
    
    Args:
        cache_type: 'all', 'roi', 'decimal', 'machine'
    """
    global _roi_info_cache, _decimal_places_cache, _machine_info_cache
    
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
