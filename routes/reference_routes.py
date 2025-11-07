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
        from utils.swagger_specs import get_reference_images_post_spec
        upload_reference_image.__doc__ = get_reference_images_post_spec().strip()
    except:
        pass
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get params
        area = request.form.get('area')  # F1, F2, etc.
        machine_code = request.form.get('machine_code')  # IE-F1-CWA01, etc.
        machine_type = request.form.get('machine_type')  # F1, F41, F42
        screen_id = request.form.get('screen_id')
        sub_page = request.form.get('sub_page')  # Optional: 1, 2 (cho Reject_Summary)
        
        if not machine_type or not screen_id:
            return jsonify({"error": "Missing machine_type or screen_id"}), 400
        
        # Create filename based on area
        ext = file.filename.rsplit('.', 1)[1].lower()
        
        # ====== AREA F1: Mỗi machine_code có template riêng ======
        if area == "F1" and machine_code:
            if sub_page:
                # Format: template_F1_{machine_code}_{screen_id}_page{N}.jpg
                filename = f"template_F1_{machine_code}_{screen_id}_page{sub_page}.{ext}"
            else:
                # Format: template_F1_{machine_code}_{screen_id}.jpg
                filename = f"template_F1_{machine_code}_{screen_id}.{ext}"
        else:
            # ====== AREA KHÁC: Dùng machine_type chung ======
            # Format: template_{machine_type}_{screen_id}.jpg
            filename = f"template_{machine_type}_{screen_id}.{ext}"
        
        filepath = os.path.join(REFERENCE_IMAGES_FOLDER, filename)
        
        # Save file
        file.save(filepath)
        
        # Clear template cache
        from utils.cache_manager import clear_cache
        clear_cache('template')
        
        response_data = {
            "message": "Reference image uploaded successfully",
            "filename": filename,
            "machine_type": machine_type,
            "screen_id": screen_id
        }
        if area == "F1" and machine_code:
            response_data["area"] = area
            response_data["machine_code"] = machine_code
        if sub_page:
            response_data["sub_page"] = sub_page
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reference_bp.route('/api/reference_images', methods=['GET'])
def get_reference_images():
    """Get list of reference images"""
    try:
        from utils.swagger_specs import get_reference_images_list_spec
        get_reference_images.__doc__ = get_reference_images_list_spec().strip()
    except:
        pass
    try:
        area = request.args.get('area')
        machine_code = request.args.get('machine_code')
        machine_type = request.args.get('machine_type')
        screen_id = request.args.get('screen_id')
        
        if not os.path.exists(REFERENCE_IMAGES_FOLDER):
            return jsonify({"images": []}), 200
        
        files = [f for f in os.listdir(REFERENCE_IMAGES_FOLDER)
                if os.path.isfile(os.path.join(REFERENCE_IMAGES_FOLDER, f)) and allowed_file(f)]
        
        # Filter by area and machine_code for F1
        if area == "F1" and machine_code:
            files = [f for f in files if f.startswith(f"template_F1_{machine_code}_")]
        elif machine_type:
            files = [f for f in files if f.startswith(f"template_{machine_type}_")]
        
        if screen_id:
            files = [f for f in files if f"_{screen_id}." in f or f"_{screen_id}_page" in f]
        
        # Parse filenames to get metadata
        image_list = []
        for filename in files:
            metadata = {"filename": filename, "path": f"/api/reference_images/{filename}"}
            
            # Parse format: template_F1_{machine_code}_{screen_id}_page{N}.ext
            if filename.startswith("template_F1_"):
                parts = filename.replace('template_F1_', '').rsplit('.', 1)[0]
                # Split by _ to get machine_code and rest
                sections = parts.split('_', 1)
                if len(sections) == 2:
                    metadata["area"] = "F1"
                    metadata["machine_code"] = sections[0]
                    metadata["machine_type"] = "F1"
                    
                    # Check for sub_page
                    remaining = sections[1]
                    if "_page" in remaining:
                        screen_parts = remaining.split("_page")
                        metadata["screen_id"] = screen_parts[0]
                        if len(screen_parts) > 1:
                            metadata["sub_page"] = screen_parts[1]
                    else:
                        metadata["screen_id"] = remaining
            
            # Parse format: template_{machine_type}_{screen_id}.ext
            elif filename.startswith("template_"):
                parts = filename.replace('template_', '').rsplit('.', 1)[0]
                name_parts = parts.split('_', 1)
                if len(name_parts) == 2:
                    metadata["machine_type"] = name_parts[0]
                    metadata["screen_id"] = name_parts[1]
            
            if "screen_id" in metadata:  # Only add if successfully parsed
                image_list.append(metadata)
        
        return jsonify({"images": image_list}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reference_bp.route('/api/reference_images/<filename>', methods=['GET'])
def get_reference_image(filename):
    """Get specific reference image"""
    try:
        from utils.swagger_specs import get_reference_image_spec
        get_reference_image.__doc__ = get_reference_image_spec().strip()
    except:
        pass
    try:
        return send_from_directory(REFERENCE_IMAGES_FOLDER, filename)
    except:
        abort(404)


@reference_bp.route('/api/reference_images/<filename>', methods=['DELETE'])
def delete_reference_image(filename):
    """Delete reference image"""
    try:
        from utils.swagger_specs import get_delete_reference_image_spec
        delete_reference_image.__doc__ = get_delete_reference_image_spec().strip()
    except:
        pass
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

