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
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
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


@decimal_bp.route('/api/decimal_places/<machine_code>', methods=['GET'])
def get_decimal_places_for_machine(machine_code):
    """Get decimal places for specific machine (accepts machine_type or machine_code)"""
    try:
        from utils.swagger_specs import get_decimal_places_machine_spec
        get_decimal_places_for_machine.__doc__ = get_decimal_places_machine_spec().strip()
    except:
        pass
    try:
        config = get_decimal_places_config()
        
        # Check if it's already a machine_type (F1, F41, F42)
        if machine_code in config:
            return jsonify(config[machine_code]), 200
        
        # Try to convert machine_code to machine_type
        machine_type = get_machine_type(machine_code)
        if machine_type and machine_type in config:
            return jsonify(config[machine_type]), 200
        
        return jsonify({}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places/<machine_code>/<screen_name>', methods=['GET'])
def get_decimal_places_for_screen(machine_code, screen_name):
    """Get decimal places for specific screen"""
    try:
        from utils.swagger_specs import get_decimal_places_screen_spec
        get_decimal_places_for_screen.__doc__ = get_decimal_places_screen_spec().strip()
    except:
        pass
    try:
        config = get_decimal_places_config()
        
        # Check if it's already a machine_type (F1, F41, F42)
        machine_type = machine_code if machine_code in config else None
        
        # Try to convert machine_code to machine_type if needed
        if not machine_type:
            machine_type = get_machine_type(machine_code)
        
        if machine_type and machine_type in config and screen_name in config[machine_type]:
            return jsonify(config[machine_type][screen_name]), 200
        
        return jsonify({}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places/<machine_code>/<screen_name>', methods=['POST'])
def update_decimal_places_for_screen(machine_code, screen_name):
    """Update decimal places for specific screen"""
    try:
        from utils.swagger_specs import get_decimal_places_screen_post_spec
        update_decimal_places_for_screen.__doc__ = get_decimal_places_screen_post_spec().strip()
    except:
        pass
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        roi_config = request.json
        
        # Read current config
        decimal_config_path = os.path.join(get_roi_data_folder(), 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
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
        if screen_name not in config[machine_type]:
            config[machine_type][screen_name] = {}
        
        for roi_name, decimal_places in roi_config.items():
            config[machine_type][screen_name][roi_name] = int(decimal_places)
        
        # Save config
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        clear_cache('decimal')
        
        return jsonify({
            "message": "Decimal places updated successfully",
            "config": config[machine_type][screen_name]
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
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
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
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
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


@decimal_bp.route('/api/decimal_places/<machine_type>/Reject_Summary/<machine_code>', methods=['GET'])
def get_decimal_places_for_reject_summary_machine(machine_type, machine_code):
    """Get decimal places for all sub-pages of a specific machine_code in Reject_Summary"""
    try:
        from utils.swagger_specs import get_decimal_places_reject_summary_machine_spec
        get_decimal_places_for_reject_summary_machine.__doc__ = get_decimal_places_reject_summary_machine_spec().strip()
    except:
        pass
    try:
        config = get_decimal_places_config()
        
        if machine_type in config:
            if "Reject_Summary" in config[machine_type]:
                if machine_code in config[machine_type]["Reject_Summary"]:
                    return jsonify(config[machine_type]["Reject_Summary"][machine_code]), 200
        
        return jsonify({}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places/<machine_type>/Reject_Summary/<machine_code>/<sub_page>', methods=['GET'])
def get_decimal_places_for_reject_summary_subpage(machine_type, machine_code, sub_page):
    """Get decimal places for a specific sub-page of Reject_Summary"""
    try:
        from utils.swagger_specs import get_decimal_places_reject_summary_subpage_get_spec
        get_decimal_places_for_reject_summary_subpage.__doc__ = get_decimal_places_reject_summary_subpage_get_spec().strip()
    except:
        pass
    try:
        config = get_decimal_places_config()
        
        if machine_type in config:
            if "Reject_Summary" in config[machine_type]:
                if machine_code in config[machine_type]["Reject_Summary"]:
                    if sub_page in config[machine_type]["Reject_Summary"][machine_code]:
                        return jsonify(config[machine_type]["Reject_Summary"][machine_code][sub_page]), 200
        
        return jsonify({}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places/<machine_type>/Reject_Summary/<machine_code>/<sub_page>', methods=['POST'])
def update_decimal_places_for_reject_summary_subpage(machine_type, machine_code, sub_page):
    """Update decimal places for a specific sub-page of Reject_Summary"""
    try:
        from utils.swagger_specs import get_decimal_places_reject_summary_subpage_post_spec
        update_decimal_places_for_reject_summary_subpage.__doc__ = get_decimal_places_reject_summary_subpage_post_spec().strip()
    except:
        pass
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        roi_config = request.json
        
        # Read current config
        decimal_config_path = os.path.join(get_roi_data_folder(), 'decimal_places.json')
        if os.path.exists(decimal_config_path):
            with open(decimal_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update config - Cấu trúc: machine_type > "Reject_Summary" > machine_code > sub_page > roi_name
        if machine_type not in config:
            config[machine_type] = {}
        if "Reject_Summary" not in config[machine_type]:
            config[machine_type]["Reject_Summary"] = {}
        if machine_code not in config[machine_type]["Reject_Summary"]:
            config[machine_type]["Reject_Summary"][machine_code] = {}
        if sub_page not in config[machine_type]["Reject_Summary"][machine_code]:
            config[machine_type]["Reject_Summary"][machine_code][sub_page] = {}
        
        for roi_name, decimal_places in roi_config.items():
            config[machine_type]["Reject_Summary"][machine_code][sub_page][roi_name] = int(decimal_places)
        
        # Save config
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        clear_cache('decimal')
        
        return jsonify({
            "message": "Decimal places updated successfully",
            "machine_type": machine_type,
            "machine_code": machine_code,
            "sub_page": sub_page,
            "config": config[machine_type]["Reject_Summary"][machine_code][sub_page]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

