"""
Swagger/OpenAPI Specifications for all API endpoints
This file contains all Swagger documentation strings for easy maintenance
"""


# ==================== SYSTEM ENDPOINTS ====================

def get_home_spec():
    """Swagger spec for GET /"""
    return """
    Get server status and available endpoints
    ---
    tags:
      - System
    responses:
      200:
        description: Server information
        schema:
          type: object
          properties:
            status:
              type: string
            version:
              type: string
            endpoints:
              type: array
              items:
                type: string
    """


def get_debug_spec():
    """Swagger spec for GET /debug"""
    return """
    Get debug information about server and routes
    ---
    tags:
      - System
    responses:
      200:
        description: Debug information
        schema:
          type: object
          properties:
            server_info:
              type: object
            routes:
              type: array
    """


def get_performance_spec():
    """Swagger spec for GET /api/performance"""
    return """
    Get performance statistics and system information
    ---
    tags:
      - System
    responses:
      200:
        description: Performance statistics
        schema:
          type: object
          properties:
            timestamp:
              type: string
            gpu_available:
              type: boolean
            optimization_enabled:
              type: boolean
            gpu_info:
              type: object
            system:
              type: object
            ocr:
              type: object
      500:
        description: Server error
    """


def get_history_spec():
    """Swagger spec for GET /api/history"""
    return """
    Get OCR history with filtering support
    ---
    tags:
      - History
    parameters:
      - name: start_time
        in: query
        type: string
        required: true
        description: Start time in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
        example: "2025-10-03"
      - name: end_time
        in: query
        type: string
        required: true
        description: End time in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
        example: "2025-10-05"
      - name: machine_code
        in: query
        type: string
        required: false
        description: Filter by machine code (e.g., "IE-F1-CWA01")
        example: "IE-F1-CWA01"
      - name: area
        in: query
        type: string
        required: false
        description: Filter by area (e.g., "F1", "F4")
        example: "F1"
      - name: screen_id
        in: query
        type: string
        required: false
        description: Filter by screen ID (e.g., "Production_Data", "Overview")
        example: "Production_Data"
      - name: limit
        in: query
        type: integer
        required: false
        default: 100
        description: Maximum number of results
        example: 50
    responses:
      200:
        description: Successfully retrieved OCR history
        schema:
          type: object
          properties:
            history:
              type: array
              items:
                type: object
            count:
              type: integer
            limit:
              type: integer
            filters_applied:
              type: object
      400:
        description: Bad request - missing required parameters
    """


# ==================== IMAGE PROCESSING ENDPOINTS ====================

def get_upload_image_spec():
    """Swagger spec for POST /api/images"""
    return """
    Upload image và thực hiện OCR
    ---
    tags:
      - Image Processing
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Image file to process (jpg, png, bmp)
      - name: area
        in: formData
        type: string
        required: true
        description: Area code (e.g., "F1", "F4")
        example: "F1"
      - name: machine_code
        in: formData
        type: string
        required: true
        description: Machine code (e.g., "IE-F1-CWA01")
        example: "IE-F1-CWA01"
    responses:
      200:
        description: Successfully processed image
        schema:
          type: object
          properties:
            success:
              type: boolean
            filename:
              type: string
            machine_code:
              type: string
            machine_type:
              type: string
            screen_id:
              type: string
            area:
              type: string
            ocr_results:
              type: array
              items:
                type: object
            roi_count:
              type: integer
      400:
        description: Bad request
      500:
        description: Server error
    """


def get_images_list_spec():
    """Swagger spec for GET /api/images"""
    return """
    Get list of uploaded images
    ---
    tags:
      - Image Processing
    responses:
      200:
        description: List of uploaded images
        schema:
          type: object
          properties:
            images:
              type: array
              items:
                type: string
      500:
        description: Server error
    """


def get_image_spec():
    """Swagger spec for GET /api/images/<filename>"""
    return """
    Get specific image file
    ---
    tags:
      - Image Processing
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Image filename
        example: "1234567890_test.jpg"
    responses:
      200:
        description: Image file
      404:
        description: File not found
    """


def get_delete_image_spec():
    """Swagger spec for DELETE /api/images/<filename>"""
    return """
    Delete image file
    ---
    tags:
      - Image Processing
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Image filename to delete
        example: "1234567890_test.jpg"
    responses:
      200:
        description: Successfully deleted
        schema:
          type: object
          properties:
            message:
              type: string
      404:
        description: File not found
      500:
        description: Server error
    """


def get_hmi_detection_spec():
    """Swagger spec for GET /api/images/hmi_detection/<filename>"""
    return """
    Get HMI detection visualization image
    ---
    tags:
      - Image Processing
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Image filename
    responses:
      200:
        description: HMI detection visualization
      404:
        description: File not found
    """


# ==================== DECIMAL PLACES ENDPOINTS ====================

def get_decimal_places_all_spec():
    """Swagger spec for GET /api/decimal_places"""
    return """
    Get all decimal places configuration
    ---
    tags:
      - Decimal Places
    responses:
      200:
        description: Successfully retrieved decimal places configuration
        schema:
          type: object
      500:
        description: Server error
    """


def get_decimal_places_post_spec():
    """Swagger spec for POST /api/decimal_places"""
    return """
    Update decimal places configuration
    ---
    tags:
      - Decimal Places
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - machine_code
            - screen_id
            - roi_config
          properties:
            machine_code:
              type: string
              example: "IE-F1-CWA01"
            screen_id:
              type: string
              example: "Production_Data"
            roi_config:
              type: object
              example: {"ROI_name": 2}
    responses:
      200:
        description: Successfully updated
      400:
        description: Bad request
      500:
        description: Server error
    """


# ====== OLD SPECS REMOVED - REPLACED BY UNIFIED API ======
# The following specs were removed as they are replaced by:
# - get_decimal_places_unified_spec()
# - get_decimal_places_unified_post_spec()


def get_set_decimal_value_spec():
    """Swagger spec for POST /api/set_decimal_value"""
    return """
    Set decimal value for current machine/screen/ROI
    ---
    tags:
      - Decimal Places
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - machine_code
            - screen_id
            - roi_index
            - decimal_places
          properties:
            machine_code:
              type: string
            screen_id:
              type: string
            roi_index:
              type: string
            decimal_places:
              type: integer
    responses:
      200:
        description: Successfully set decimal value
      400:
        description: Bad request
      500:
        description: Server error
    """


def get_set_all_decimal_values_spec():
    """Swagger spec for POST /api/set_all_decimal_values"""
    return """
    Set all decimal values for a screen
    ---
    tags:
      - Decimal Places
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - machine_code
            - screen_id
            - decimal_config
          properties:
            machine_code:
              type: string
            screen_id:
              type: string
            decimal_config:
              type: object
    responses:
      200:
        description: Successfully set all decimal values
      400:
        description: Bad request
      500:
        description: Server error
    """


# ====== REJECT_SUMMARY OLD SPECS REMOVED ======
# These specs were also replaced by the unified API


def get_decimal_places_unified_spec():
    """Swagger spec for GET /api/decimal_places/<machine_type>/<screen_name> (UNIFIED)"""
    return """
    [UNIFIED API] Get decimal places configuration
    ---
    tags:
      - Decimal Places
    parameters:
      - name: machine_type
        in: path
        type: string
        required: true
        description: Machine type (F1, F41, F42)
        example: "F1"
      - name: screen_name
        in: path
        type: string
        required: true
        description: Screen name (Production_Data, Reject_Summary, Injection, etc.)
        example: "Production_Data"
      - name: machine_code
        in: query
        type: string
        required: false
        description: Machine code (e.g., IE-F1-CWA01) - required for Reject_Summary
        example: "IE-F1-CWA01"
      - name: sub_page
        in: query
        type: string
        required: false
        description: Sub-page number (1, 2, etc.) - optional for Reject_Summary
        example: "1"
    responses:
      200:
        description: Successfully retrieved decimal places
        schema:
          type: object
          properties:
            machine_type:
              type: string
              example: "F1"
            screen_name:
              type: string
              example: "Production_Data"
            machine_code:
              type: string
              example: "IE-F1-CWA01"
            sub_page:
              type: string
              example: "1"
            decimal_config:
              type: object
              example: {"Total Parts": 0, "Good Parts": 0}
      404:
        description: Machine type or screen not found
      500:
        description: Server error
    """


def get_decimal_places_unified_post_spec():
    """Swagger spec for POST /api/decimal_places/<machine_type>/<screen_name> (UNIFIED)"""
    return """
    [UNIFIED API] Update decimal places configuration
    ---
    tags:
      - Decimal Places
    parameters:
      - name: machine_type
        in: path
        type: string
        required: true
        description: Machine type (F1, F41, F42)
        example: "F1"
      - name: screen_name
        in: path
        type: string
        required: true
        description: Screen name (Production_Data, Reject_Summary, Injection, etc.)
        example: "Production_Data"
      - name: machine_code
        in: query
        type: string
        required: false
        description: Machine code (e.g., IE-F1-CWA01) - required for Reject_Summary
        example: "IE-F1-CWA01"
      - name: sub_page
        in: query
        type: string
        required: false
        description: Sub-page number (1, 2, etc.) - optional for Reject_Summary
        example: "1"
      - name: body
        in: body
        required: true
        description: |
          ROI configuration object. Content depends on the context:
          
          1. Standard screen: {"ROI_name": decimal_places}
             Example: {"Injection speed": 1, "Charge speed": 1}
          
          2. Reject_Summary with machine_code only: {"sub_page": {...}}
             Example: {"1": {"ST02_TESTED": 0}, "2": {"ST14_1_TESTED": 0}}
          
          3. Reject_Summary with machine_code and sub_page: {"ROI_name": decimal_places}
             Example: {"ST02_TESTED": 0, "ST02_REJECTS": 0}
        schema:
          type: object
          example: {"Total Parts": 0, "Good Parts": 0}
    responses:
      200:
        description: Successfully updated decimal places
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Decimal places updated successfully"
            machine_type:
              type: string
              example: "F1"
            screen_name:
              type: string
              example: "Production_Data"
            config:
              type: object
      400:
        description: Bad request (invalid JSON or missing required fields)
      500:
        description: Server error
    """


# ====== POST SPEC FOR REJECT_SUMMARY SUB-PAGE REMOVED ======
# Replaced by get_decimal_places_unified_post_spec()


# ==================== MACHINE MANAGEMENT ENDPOINTS ====================

def get_machines_spec():
    """Swagger spec for GET /api/machines"""
    return """
    Get all machines information
    ---
    tags:
      - Machine Management
    responses:
      200:
        description: Successfully retrieved machines information
        schema:
          type: object
          properties:
            areas:
              type: object
      500:
        description: Server error
    """


def get_machines_by_area_spec():
    """Swagger spec for GET /api/machines/<area_code>"""
    return """
    Get machines for specific area
    ---
    tags:
      - Machine Management
    parameters:
      - name: area_code
        in: path
        type: string
        required: true
        description: Area code (e.g., "F1", "F4")
        example: "F1"
    responses:
      200:
        description: Successfully retrieved machines for area
        schema:
          type: object
          properties:
            area_code:
              type: string
            area_name:
              type: string
            machines:
              type: object
      404:
        description: Area not found
      500:
        description: Server error
    """


def get_machine_screens_spec():
    """Swagger spec for GET /api/machine_screens/<machine_code>"""
    return """
    Get screens for specific machine
    ---
    tags:
      - Machine Management
    parameters:
      - name: machine_code
        in: path
        type: string
        required: true
        description: Machine code (e.g., "IE-F1-CWA01")
        example: "IE-F1-CWA01"
    responses:
      200:
        description: Successfully retrieved machine screens
        schema:
          type: object
          properties:
            machine_code:
              type: string
            machine_name:
              type: string
            screens:
              type: array
              items:
                type: object
      404:
        description: Machine not found
      500:
        description: Server error
    """


def get_set_machine_screen_spec():
    """Swagger spec for POST /api/set_machine_screen"""
    return """
    Set current machine and screen
    ---
    tags:
      - Machine Management
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - machine_code
            - screen_id
          properties:
            machine_code:
              type: string
            screen_id:
              type: string
    responses:
      200:
        description: Successfully set machine and screen
      400:
        description: Bad request
      500:
        description: Server error
    """


def get_current_machine_screen_spec():
    """Swagger spec for GET /api/current_machine_screen"""
    return """
    Get current machine and screen
    ---
    tags:
      - Machine Management
    responses:
      200:
        description: Current machine and screen information
        schema:
          type: object
          properties:
            machine_code:
              type: string
            screen_id:
              type: string
      500:
        description: Server error
    """


def get_machine_screen_status_spec():
    """Swagger spec for GET /api/machine_screen_status"""
    return """
    Get machine and screen status
    ---
    tags:
      - Machine Management
    responses:
      200:
        description: Machine and screen status
        schema:
          type: object
          properties:
            machine_code:
              type: string
            screen_id:
              type: string
            status:
              type: string
      500:
        description: Server error
    """


def get_update_machine_screen_spec():
    """Swagger spec for POST /api/update_machine_screen"""
    return """
    Update machine and screen information
    ---
    tags:
      - Machine Management
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - machine_code
            - screen_id
          properties:
            machine_code:
              type: string
            screen_id:
              type: string
    responses:
      200:
        description: Successfully updated
      400:
        description: Bad request
      500:
        description: Server error
    """


# ==================== REFERENCE IMAGES ENDPOINTS ====================

def get_reference_images_post_spec():
    """Swagger spec for POST /api/reference_images"""
    return """
    Upload reference template image
    ---
    tags:
      - Reference Images
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Reference template image file
      - name: machine_type
        in: formData
        type: string
        required: true
        description: Machine type (e.g., "F1", "F41", "F42")
        example: "F1"
      - name: screen_id
        in: formData
        type: string
        required: true
        description: Screen ID (e.g., "Production_Data", "Overview")
        example: "Production_Data"
    responses:
      200:
        description: Successfully uploaded reference image
        schema:
          type: object
          properties:
            message:
              type: string
            filename:
              type: string
      400:
        description: Bad request
      500:
        description: Server error
    """


def get_reference_images_list_spec():
    """Swagger spec for GET /api/reference_images"""
    return """
    Get list of reference template images
    ---
    tags:
      - Reference Images
    responses:
      200:
        description: List of reference images
        schema:
          type: object
          properties:
            images:
              type: array
              items:
                type: object
      500:
        description: Server error
    """


def get_reference_image_spec():
    """Swagger spec for GET /api/reference_images/<filename>"""
    return """
    Get specific reference image file
    ---
    tags:
      - Reference Images
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Reference image filename
        example: "template_F1_Production_Data.jpg"
    responses:
      200:
        description: Reference image file
      404:
        description: File not found
    """


def get_delete_reference_image_spec():
    """Swagger spec for DELETE /api/reference_images/<filename>"""
    return """
    Delete reference image file
    ---
    tags:
      - Reference Images
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Reference image filename to delete
        example: "template_F1_Production_Data.jpg"
    responses:
      200:
        description: Successfully deleted
        schema:
          type: object
          properties:
            message:
              type: string
      404:
        description: File not found
      500:
        description: Server error
    """

