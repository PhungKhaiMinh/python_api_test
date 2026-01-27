"""
Utils package - PaddleOCR Edition
"""

from .config_manager import (
    get_roi_coordinates,
    get_roi_coordinates_with_subpage,
    get_machine_type,
    get_area_for_machine,
    get_machine_name_from_code,
    get_all_machine_types,
    get_decimal_places_config,
    get_current_machine_info,
    save_current_machine_info,
    is_named_roi_format,
    get_roi_data_folder
)

from .cache_manager import (
    get_roi_info_cached,
    get_decimal_places_cached,
    get_machine_info_cached,
    clear_cache,
    initialize_all_caches
)

from .image_processor import (
    detect_hmi_screen,
    preprocess_hmi_image,
    preprocess_roi_for_ocr,
    check_image_quality,
    enhance_image_quality,
    enhance_image,
    ImageAligner
)

from .ocr_processor import (
    perform_ocr_on_roi,
    perform_ocr_on_roi_optimized,
    apply_decimal_places_format,
    init_ocr_globals,
    perform_full_image_ocr
)

from .paddleocr_engine import (
    get_paddleocr_instance,
    read_image_with_paddleocr,
    read_image_with_paddleocr_batch,
    extract_ocr_data,
    find_matching_screen,
    filter_ocr_by_roi,
    filter_ocr_by_roi_parallel,
    post_process_ocr_text,
    load_roi_info,
    init_paddleocr_globals,
    detect_hmi_screen_paddle,
    get_ocr_performance_stats,
    HAS_PADDLEOCR,
    USE_GPU
)

__all__ = [
    # Config Manager
    'get_roi_coordinates',
    'get_roi_coordinates_with_subpage',
    'get_machine_type',
    'get_area_for_machine',
    'get_machine_name_from_code',
    'get_all_machine_types',
    'get_decimal_places_config',
    'get_current_machine_info',
    'save_current_machine_info',
    'is_named_roi_format',
    'get_roi_data_folder',
    
    # Cache Manager
    'get_roi_info_cached',
    'get_decimal_places_cached',
    'get_machine_info_cached',
    'clear_cache',
    'initialize_all_caches',
    
    # Image Processor
    'detect_hmi_screen',
    'preprocess_hmi_image',
    'preprocess_roi_for_ocr',
    'check_image_quality',
    'enhance_image_quality',
    'enhance_image',
    'ImageAligner',
    
    # OCR Processor
    'perform_ocr_on_roi',
    'perform_ocr_on_roi_optimized',
    'apply_decimal_places_format',
    'init_ocr_globals',
    'perform_full_image_ocr',
    
    # PaddleOCR Engine
    'get_paddleocr_instance',
    'read_image_with_paddleocr',
    'read_image_with_paddleocr_batch',
    'extract_ocr_data',
    'find_matching_screen',
    'filter_ocr_by_roi',
    'filter_ocr_by_roi_parallel',
    'post_process_ocr_text',
    'load_roi_info',
    'init_paddleocr_globals',
    'detect_hmi_screen_paddle',
    'get_ocr_performance_stats',
    'HAS_PADDLEOCR',
    'USE_GPU'
]
