"""
Decimal Routes Module
Handles decimal places configuration endpoints
"""

from flask import Blueprint, request, jsonify
import os
import json
from utils import get_roi_data_folder, get_decimal_places_config
from utils.cache_manager import clear_cache

decimal_bp = Blueprint('decimal', __name__)


@decimal_bp.route('/api/decimal_places', methods=['GET'])
def get_decimal_places():
    """Get decimal places configuration"""
    try:
        config = get_decimal_places_config()
        return jsonify(config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places', methods=['POST'])
def update_decimal_places():
    """Update decimal places configuration"""
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
        
        # Update config
        if machine_code not in config:
            config[machine_code] = {}
        if screen_id not in config[machine_code]:
            config[machine_code][screen_id] = {}
        
        for roi_name, decimal_places in roi_config.items():
            config[machine_code][screen_id][roi_name] = int(decimal_places)
        
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
    """Get decimal places for specific machine"""
    try:
        config = get_decimal_places_config()
        
        if machine_code in config:
            return jsonify(config[machine_code]), 200
        return jsonify({}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places/<machine_code>/<screen_name>', methods=['GET'])
def get_decimal_places_for_screen(machine_code, screen_name):
    """Get decimal places for specific screen"""
    try:
        config = get_decimal_places_config()
        
        if machine_code in config and screen_name in config[machine_code]:
            return jsonify(config[machine_code][screen_name]), 200
        return jsonify({}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/decimal_places/<machine_code>/<screen_name>', methods=['POST'])
def update_decimal_places_for_screen(machine_code, screen_name):
    """Update decimal places for specific screen"""
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
        
        # Update config
        if machine_code not in config:
            config[machine_code] = {}
        if screen_name not in config[machine_code]:
            config[machine_code][screen_name] = {}
        
        for roi_name, decimal_places in roi_config.items():
            config[machine_code][screen_name][roi_name] = int(decimal_places)
        
        # Save config
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        clear_cache('decimal')
        
        return jsonify({
            "message": "Decimal places updated successfully",
            "config": config[machine_code][screen_name]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@decimal_bp.route('/api/set_decimal_value', methods=['POST'])
def set_decimal_value():
    """Set decimal value for current machine/screen/ROI"""
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
        
        # Update config
        if machine_code not in config:
            config[machine_code] = {}
        if screen_id not in config[machine_code]:
            config[machine_code][screen_id] = {}
        
        config[machine_code][screen_id][roi_index] = int(decimal_places)
        
        # Save config
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        clear_cache('decimal')
        
        return jsonify({
            "message": "Decimal value set successfully",
            "machine_code": machine_code,
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
        
        # Update config
        if machine_code not in config:
            config[machine_code] = {}
        
        config[machine_code][screen_id] = {}
        for roi_name, decimal_places in decimal_config.items():
            config[machine_code][screen_id][roi_name] = int(decimal_places)
        
        # Save config
        with open(decimal_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        clear_cache('decimal')
        
        return jsonify({
            "message": "All decimal values set successfully",
            "config": config[machine_code][screen_id]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

