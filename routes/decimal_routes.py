"""
Decimal Routes Module
Handles decimal places configuration endpoints
"""

from flask import Blueprint, request, jsonify
import os
import json
from utils import get_roi_data_folder, get_decimal_places_config
from utils.config_manager import get_machine_type
from utils.cache_manager import clear_cache

decimal_bp = Blueprint('decimal', __name__)


@decimal_bp.route('/api/decimal_places', methods=['GET'])
def get_decimal_places():
    """Get decimal places configuration"""
    try:
        from utils.swagger_specs import get_decimal_places_all_spec
        get_decimal_places.__doc__ = get_decimal_places_all_spec().strip()
    except:
        pass
    try:
        config = get_decimal_places_config()
        return jsonify(config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places', methods=['POST'])
def update_decimal_places():
    """Update decimal places configuration"""
    try:
        from utils.swagger_specs import get_decimal_places_post_spec
        update_decimal_places.__doc__ = get_decimal_places_post_spec().strip()
    except:
        pass
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        if 'machine_code' not in data or 'screen_id' not in data or 'roi_config' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        machine_code = data['machine_code']
        screen_id = data['screen_id']
        roi_config = data['roi_config']
        
        # Read current config
        decimal_config_path = os.path.join(get_roi_data_folder(), 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Check if it's already a machine_type (F1, F41, F42)
        machine_type = machine_code if machine_code in config else None
        
        # Try to convert machine_code to machine_type if needed
        if not machine_type:
            machine_type = get_machine_type(machine_code)
            if not machine_type:
                return jsonify({"error": f"Could not determine machine_type from machine_code: {machine_code}"}), 400
        
        # Update config
        if machine_type not in config:
            config[machine_type] = {}
        if screen_id not in config[machine_type]:
            config[machine_type][screen_id] = {}
        
        for roi_name, decimal_places in roi_config.items():
            config[machine_type][screen_id][roi_name] = int(decimal_places)
        
        # Save config
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        clear_cache('decimal')
        
        return jsonify({
            "message": "Decimal places updated successfully",
            "config": config
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places/<machine_type>/<screen_name>', methods=['GET'])
def get_decimal_places_unified(machine_type, screen_name):
    """
    [UNIFIED API] Get decimal places configuration
    
    Required path parameters:
    - machine_type: Machine type (F1, F41, F42)
    - screen_name: Screen name (Production_Data, Reject_Summary, etc.)
    
    Optional query parameters:
    - machine_code: Machine code (e.g., IE-F1-CWA01) - only for Reject_Summary
    - sub_page: Sub-page number (1, 2, etc.) - only for Reject_Summary with machine_code
    
    Examples:
    1. Get all config for a screen:
       GET /api/decimal_places/F41/Injection
       
    2. Get all Reject_Summary config:
       GET /api/decimal_places/F1/Reject_Summary
       
    3. Get Reject_Summary for specific machine:
       GET /api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01
       
    4. Get Reject_Summary for specific machine and sub-page:
       GET /api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01&sub_page=1
    """
    try:
        from utils.swagger_specs import get_decimal_places_unified_spec
        get_decimal_places_unified.__doc__ = get_decimal_places_unified_spec().strip()
    except:
        pass
        
    try:
        # Get optional query parameters
        machine_code = request.args.get('machine_code')
        sub_page = request.args.get('sub_page')
        
        # Load config
        config = get_decimal_places_config()
        
        # Validate machine_type exists
        if machine_type not in config:
            return jsonify({"error": f"Machine type '{machine_type}' not found"}), 404
        
        # Validate screen_name exists
        if screen_name not in config[machine_type]:
            return jsonify({"error": f"Screen '{screen_name}' not found for machine type '{machine_type}'"}), 404
        
        screen_config = config[machine_type][screen_name]
        
        # CASE 1: Reject_Summary with machine_code and sub_page
        if screen_name == "Reject_Summary" and machine_code and sub_page:
            if machine_code not in screen_config:
                return jsonify({"error": f"Machine code '{machine_code}' not found in Reject_Summary"}), 404
            
            if sub_page not in screen_config[machine_code]:
                return jsonify({"error": f"Sub-page '{sub_page}' not found for machine '{machine_code}'"}), 404
            
            return jsonify({
                "machine_type": machine_type,
                "screen_name": screen_name,
                "machine_code": machine_code,
                "sub_page": sub_page,
                "decimal_config": screen_config[machine_code][sub_page]
            }), 200
        
        # CASE 2: Reject_Summary with machine_code only (all sub-pages)
        elif screen_name == "Reject_Summary" and machine_code:
            if machine_code not in screen_config:
                return jsonify({"error": f"Machine code '{machine_code}' not found in Reject_Summary"}), 404
            
            return jsonify({
                "machine_type": machine_type,
                "screen_name": screen_name,
                "machine_code": machine_code,
                "decimal_config": screen_config[machine_code]
            }), 200
        
        # CASE 3: Standard screen or Reject_Summary without filters (full config)
        else:
            return jsonify({
                "machine_type": machine_type,
                "screen_name": screen_name,
                "decimal_config": screen_config
            }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/set_decimal_value', methods=['POST'])
def set_decimal_value():
    """Set decimal value for current machine/screen/ROI"""
    try:
        from utils.swagger_specs import get_set_decimal_value_spec
        set_decimal_value.__doc__ = get_set_decimal_value_spec().strip()
    except:
        pass
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        machine_code = data.get('machine_code')
        screen_id = data.get('screen_id')
        roi_index = data.get('roi_index')
        decimal_places = data.get('decimal_places')
        
        if not all([machine_code, screen_id, roi_index is not None, decimal_places is not None]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Read current config
        decimal_config_path = os.path.join(get_roi_data_folder(), 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Check if it's already a machine_type (F1, F41, F42)
        machine_type = machine_code if machine_code in config else None
        
        # Try to convert machine_code to machine_type if needed
        if not machine_type:
            machine_type = get_machine_type(machine_code)
            if not machine_type:
                return jsonify({"error": f"Could not determine machine_type from machine_code: {machine_code}"}), 400
        
        # Update config
        if machine_type not in config:
            config[machine_type] = {}
        if screen_id not in config[machine_type]:
            config[machine_type][screen_id] = {}
        
        config[machine_type][screen_id][roi_index] = int(decimal_places)
        
        # Save config
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        clear_cache('decimal')
        
        return jsonify({
            "message": "Decimal value set successfully",
            "machine_code": machine_code,
            "machine_type": machine_type,
            "screen_id": screen_id,
            "roi_index": roi_index,
            "decimal_places": decimal_places
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/set_all_decimal_values', methods=['POST'])
def set_all_decimal_values():
    """Set all decimal values for a screen"""
    try:
        from utils.swagger_specs import get_set_all_decimal_values_spec
        set_all_decimal_values.__doc__ = get_set_all_decimal_values_spec().strip()
    except:
        pass
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        machine_code = data.get('machine_code')
        screen_id = data.get('screen_id')
        decimal_config = data.get('decimal_config', {})
        
        if not machine_code or not screen_id:
            return jsonify({"error": "Missing machine_code or screen_id"}), 400
        
        # Read current config
        decimal_config_path = os.path.join(get_roi_data_folder(), 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Check if it's already a machine_type (F1, F41, F42)
        machine_type = machine_code if machine_code in config else None
        
        # Try to convert machine_code to machine_type if needed
        if not machine_type:
            machine_type = get_machine_type(machine_code)
            if not machine_type:
                return jsonify({"error": f"Could not determine machine_type from machine_code: {machine_code}"}), 400
        
        # Update config
        if machine_type not in config:
            config[machine_type] = {}
        
        config[machine_type][screen_id] = {}
        for roi_name, decimal_places in decimal_config.items():
            config[machine_type][screen_id][roi_name] = int(decimal_places)
        
        # Save config
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        clear_cache('decimal')
        
        return jsonify({
            "message": "All decimal values set successfully",
            "machine_type": machine_type,
            "config": config[machine_type][screen_id]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places/<machine_type>/<screen_name>', methods=['POST'])
def update_decimal_places_unified(machine_type, screen_name):
    """
    [UNIFIED API] Update decimal places configuration
    
    Required path parameters:
    - machine_type: Machine type (F1, F41, F42)
    - screen_name: Screen name (Production_Data, Reject_Summary, etc.)
    
    Optional query parameters:
    - machine_code: Machine code (e.g., IE-F1-CWA01) - required for Reject_Summary
    - sub_page: Sub-page number (1, 2, etc.) - optional for Reject_Summary
    
    Request body: JSON object with ROI names as keys and decimal places as values
    
    Examples:
    1. Update standard screen config:
       POST /api/decimal_places/F41/Injection
       Body: {"Injection speed": 1, "Charge speed": 1, ...}
       
    2. Update Reject_Summary for specific machine (all sub-pages):
       POST /api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01
       Body: {"1": {...}, "2": {...}}
       
    3. Update Reject_Summary for specific machine and sub-page:
       POST /api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01&sub_page=1
       Body: {"ST02_TESTED": 0, "ST02_REJECTS": 0, ...}
    """
    try:
        from utils.swagger_specs import get_decimal_places_unified_post_spec
        update_decimal_places_unified.__doc__ = get_decimal_places_unified_post_spec().strip()
    except:
        pass
        
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        # Get optional query parameters
        machine_code = request.args.get('machine_code')
        sub_page = request.args.get('sub_page')
        
        roi_config = request.json
        
        # Read current config
        decimal_config_path = os.path.join(get_roi_data_folder(), 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Initialize machine_type if not exists
        if machine_type not in config:
            config[machine_type] = {}
        
        # CASE 1: Reject_Summary with machine_code and sub_page
        if screen_name == "Reject_Summary" and machine_code and sub_page:
            if screen_name not in config[machine_type]:
                config[machine_type][screen_name] = {}
            if machine_code not in config[machine_type][screen_name]:
                config[machine_type][screen_name][machine_code] = {}
            if sub_page not in config[machine_type][screen_name][machine_code]:
                config[machine_type][screen_name][machine_code][sub_page] = {}
            
            # Update config
            for roi_name, decimal_places in roi_config.items():
                config[machine_type][screen_name][machine_code][sub_page][roi_name] = int(decimal_places)
            
            # Save and return
            with open(decimal_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            clear_cache('decimal')
            
            return jsonify({
                "message": "Decimal places updated successfully",
                "machine_type": machine_type,
                "screen_name": screen_name,
                "machine_code": machine_code,
                "sub_page": sub_page,
                "config": config[machine_type][screen_name][machine_code][sub_page]
            }), 200
        
        # CASE 2: Reject_Summary with machine_code only (update all sub-pages)
        elif screen_name == "Reject_Summary" and machine_code:
            if not machine_code:
                return jsonify({"error": "machine_code is required for Reject_Summary"}), 400
            
            if screen_name not in config[machine_type]:
                config[machine_type][screen_name] = {}
            if machine_code not in config[machine_type][screen_name]:
                config[machine_type][screen_name][machine_code] = {}
            
            # Update config (roi_config should contain sub-pages)
            for sub_page_key, sub_page_config in roi_config.items():
                config[machine_type][screen_name][machine_code][sub_page_key] = sub_page_config
            
            # Save and return
            with open(decimal_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            clear_cache('decimal')
            
            return jsonify({
                "message": "Decimal places updated successfully",
                "machine_type": machine_type,
                "screen_name": screen_name,
                "machine_code": machine_code,
                "config": config[machine_type][screen_name][machine_code]
            }), 200
        
        # CASE 3: Standard screen
        else:
            if screen_name not in config[machine_type]:
                config[machine_type][screen_name] = {}
            
            # Update config
            for roi_name, decimal_places in roi_config.items():
                config[machine_type][screen_name][roi_name] = int(decimal_places)
            
            # Save and return
            with open(decimal_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            clear_cache('decimal')
            
            return jsonify({
                "message": "Decimal places updated successfully",
                "machine_type": machine_type,
                "screen_name": screen_name,
                "config": config[machine_type][screen_name]
            }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

