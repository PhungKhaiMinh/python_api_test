"""
Image Routes Module
Handles image upload, retrieval, and deletion endpoints
"""

from flask import Blueprint, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import cv2
import time
from smart_detection_functions import auto_detect_machine_and_screen_smart
from utils import (
    get_roi_coordinates, get_template_image_cached, get_reference_template_path,
    ImageAligner, perform_ocr_on_roi_optimized, detect_hmi_screen
)

image_bp = Blueprint('image', __name__)

# Will be set by app
UPLOAD_FOLDER = None
HMI_REFINED_FOLDER = None
ALLOWED_EXTENSIONS = None


def init_image_routes(upload_folder, hmi_folder, allowed_ext):
    """Initialize route config from app"""
    global UPLOAD_FOLDER, HMI_REFINED_FOLDER, ALLOWED_EXTENSIONS
    UPLOAD_FOLDER = upload_folder
    HMI_REFINED_FOLDER = hmi_folder
    ALLOWED_EXTENSIONS = allowed_ext


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@image_bp.route('/api/images', methods=['POST'])
def upload_image():
    """Upload image và thực hiện OCR"""
    try:
        # Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get params
        area = request.form.get('area')
        machine_code = request.form.get('machine_code')
        
        if not area or not machine_code:
            return jsonify({"error": "Missing area or machine_code"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename_with_timestamp = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename_with_timestamp)
        file.save(filepath)
        
        # Read image
        uploaded_image = cv2.imread(filepath)
        if uploaded_image is None:
            return jsonify({"error": "Failed to read image"}), 400
        
        # Auto detect machine and screen
        print(f"[*] Auto-detecting machine and screen for {area}/{machine_code}...")
        detection_result = auto_detect_machine_and_screen_smart(
            uploaded_image, area=area, machine_code=machine_code
        )
        
        machine_type = detection_result.get('machine_type', 'F41')
        screen_id = detection_result.get('screen_id', 'Main')
        template_path = detection_result.get('template_path')
        
        print(f"[OK] Detected: {machine_type} - {screen_id}")
        
        # Get ROI coordinates
        roi_coordinates, roi_names = get_roi_coordinates(machine_code, screen_id, machine_type)
        
        if not roi_coordinates:
            return jsonify({"error": f"No ROI coordinates for {machine_code}/{screen_id}"}), 404
        
        # Align image if template available
        image = uploaded_image
        if template_path:
            template_img = get_template_image_cached(template_path)
            if template_img is not None:
                aligner = ImageAligner(template_img, image)
                aligned_image = aligner.align_images()
                image = aligned_image
        
        # Perform OCR
        ocr_results = perform_ocr_on_roi_optimized(
            image, roi_coordinates, filename_with_timestamp,
            template_path=template_path,
            roi_names=roi_names,
            machine_code=machine_code,
            screen_id=screen_id
        )
        
        return jsonify({
            "success": True,
            "filename": filename_with_timestamp,
            "machine_code": machine_code,
            "machine_type": machine_type,
            "screen_id": screen_id,
            "area": area,
            "detection_method": detection_result.get('detection_method'),
            "similarity_score": detection_result.get('similarity_score', 0),
            "ocr_results": ocr_results,
            "roi_count": len(ocr_results)
        }), 200
        
    except Exception as e:
        print(f"Error in upload_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@image_bp.route('/api/images', methods=['GET'])
def get_images():
    """Get list of uploaded images"""
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            return jsonify({"images": []}), 200
        
        files = [f for f in os.listdir(UPLOAD_FOLDER) 
                if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)) and allowed_file(f)]
        
        return jsonify({"images": files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@image_bp.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    """Get specific image file"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except:
        abort(404)


@image_bp.route('/api/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    """Delete image file"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"message": f"Deleted {filename}"}), 200
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@image_bp.route('/api/images/processed_roi/<filename>', methods=['GET'])
def get_processed_roi(filename):
    """Get processed ROI image"""
    processed_folder = os.path.join(UPLOAD_FOLDER, 'processed_roi')
    try:
        return send_from_directory(processed_folder, filename)
    except:
        abort(404)


@image_bp.route('/api/images/hmi_refined/<filename>', methods=['GET'])
def get_hmi_refined_image(filename):
    """Get HMI refined image"""
    try:
        return send_from_directory(HMI_REFINED_FOLDER, filename)
    except:
        abort(404)


@image_bp.route('/api/images/aligned/<filename>', methods=['GET'])
def get_aligned_image(filename):
    """Get aligned image"""
    aligned_folder = os.path.join(UPLOAD_FOLDER, 'aligned')
    try:
        return send_from_directory(aligned_folder, filename)
    except:
        abort(404)


@image_bp.route('/api/images/hmi_detection/<filename>', methods=['GET'])
def get_hmi_detection_image(filename):
    """Get HMI detection image"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except:
        abort(404)

