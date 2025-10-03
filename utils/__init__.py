"""
Utils package for HMI OCR API
Contains utility modules for caching, config, image processing, and OCR
"""

from .cache_manager import *
from .config_manager import *
from .image_processor import *
from .ocr_processor import *

__all__ = [
    # Cache Manager
    'initialize_all_caches',
    'get_template_image_cached',
    'get_roi_info_cached',
    'get_decimal_places_config_cached',
    'get_machine_info_cached',
    
    # Config Manager
    'get_roi_coordinates',
    'get_machine_type',
    'get_area_for_machine',
    'get_machine_name_from_code',
    'get_all_machine_types',
    'find_machine_code_from_template',
    'get_screen_numeric_id',
    'get_reference_template_path',
    'get_decimal_places_config',
    'is_named_roi_format',
    'get_current_machine_info',
    
    # Image Processor
    'ImageAligner',
    'preprocess_hmi_image_with_alignment',
    'preprocess_hmi_image',
    'preprocess_roi_for_ocr',
    'check_image_quality',
    'enhance_image_quality',
    'enhance_image',
    'adaptive_edge_detection',
    'detect_hmi_screen',
    'extract_content_region',
    'fine_tune_hmi_screen',
    
    # OCR Processor
    'process_single_roi_optimized',
    'process_roi_with_retry_logic_optimized',
    'perform_ocr_on_roi_optimized',
    'perform_ocr_on_roi',
    'process_roi_with_retry_logic',
]

