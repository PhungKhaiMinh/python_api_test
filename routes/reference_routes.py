"""
Reference Routes Module
Handles reference images management endpoints
"""

from flask import Blueprint, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import time

reference_bp = Blueprint('reference', __name__)

# Will be set by app
REFERENCE_IMAGES_FOLDER = None
ALLOWED_EXTENSIONS = None


def init_reference_routes(ref_folder, allowed_ext):
    """Initialize route config from app"""
    global REFERENCE_IMAGES_FOLDER, ALLOWED_EXTENSIONS
    REFERENCE_IMAGES_FOLDER = ref_folder
    ALLOWED_EXTENSIONS = allowed_ext


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@reference_bp.route('/api/reference_images', methods=['POST'])
def upload_reference_image():
    """Upload reference image"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get params
        machine_type = request.form.get('machine_type')
        screen_id = request.form.get('screen_id')
        
        if not machine_type or not screen_id:
            return jsonify({"error": "Missing machine_type or screen_id"}), 400
        
        # Create filename: template_{machine_type}_{screen_id}.jpg
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"template_{machine_type}_{screen_id}.{ext}"
        filepath = os.path.join(REFERENCE_IMAGES_FOLDER, filename)
        
        # Save file
        file.save(filepath)
        
        # Clear template cache
        from utils.cache_manager import clear_cache
        clear_cache('template')
        
        return jsonify({
            "message": "Reference image uploaded successfully",
            "filename": filename,
            "machine_type": machine_type,
            "screen_id": screen_id
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reference_bp.route('/api/reference_images', methods=['GET'])
def get_reference_images():
    """Get list of reference images"""
    try:
        machine_type = request.args.get('machine_type')
        screen_id = request.args.get('screen_id')
        
        if not os.path.exists(REFERENCE_IMAGES_FOLDER):
            return jsonify({"images": []}), 200
        
        files = [f for f in os.listdir(REFERENCE_IMAGES_FOLDER)
                if os.path.isfile(os.path.join(REFERENCE_IMAGES_FOLDER, f)) and allowed_file(f)]
        
        # Filter by machine_type and screen_id if provided
        if machine_type:
            files = [f for f in files if f.startswith(f"template_{machine_type}_")]
        
        if screen_id:
            if machine_type:
                files = [f for f in files if f.startswith(f"template_{machine_type}_{screen_id}.")]
            else:
                files = [f for f in files if f"_{screen_id}." in f]
        
        # Parse filenames to get metadata
        image_list = []
        for filename in files:
            # Format: template_{machine_type}_{screen_id}.ext
            parts = filename.replace('template_', '').rsplit('.', 1)
            if len(parts) == 2:
                name_parts = parts[0].split('_', 1)
                if len(name_parts) == 2:
                    image_list.append({
                        "filename": filename,
                        "machine_type": name_parts[0],
                        "screen_id": name_parts[1],
                        "path": f"/api/reference_images/{filename}"
                    })
        
        return jsonify({"images": image_list}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reference_bp.route('/api/reference_images/<filename>', methods=['GET'])
def get_reference_image(filename):
    """Get specific reference image"""
    try:
        return send_from_directory(REFERENCE_IMAGES_FOLDER, filename)
    except:
        abort(404)


@reference_bp.route('/api/reference_images/<filename>', methods=['DELETE'])
def delete_reference_image(filename):
    """Delete reference image"""
    try:
        filepath = os.path.join(REFERENCE_IMAGES_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            
            # Clear template cache
            from utils.cache_manager import clear_cache
            clear_cache('template')
            
            return jsonify({"message": f"Deleted {filename}"}), 200
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

