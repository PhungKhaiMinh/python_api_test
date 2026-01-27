"""
Machine Routes Module  
Handles machine and screen management endpoints
"""

from flask import Blueprint, request, jsonify
import os
import json
from utils import (
    get_machine_type, get_area_for_machine, get_machine_name_from_code,
    save_current_machine_info, get_current_machine_info, get_roi_data_folder
)

machine_bp = Blueprint('machine', __name__)


@machine_bp.route('/api/machines', methods=['GET'])
def get_machine_info_route():
    """Get machine information"""
    try:
        from utils.swagger_specs import get_machines_spec
        get_machine_info_route.__doc__ = get_machines_spec().strip()
    except:
        pass
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens config not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Get specific machine if requested
        machine_id = request.args.get('machineid')
        
        if machine_id:
            for area_code, area_info in data.get('areas', {}).items():
                machines = area_info.get('machines', {})
                if machine_id in machines:
                    return jsonify({
                        "machine": machines[machine_id],
                        "area": area_code
                    }), 200
            return jsonify({"error": "Machine not found"}), 404
        
        # Return all machines
        return jsonify(data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@machine_bp.route('/api/machines/<area_code>', methods=['GET'])
def get_machines_by_area(area_code):
    """Get machines for specific area"""
    try:
        from utils.swagger_specs import get_machines_by_area_spec
        get_machines_by_area.__doc__ = get_machines_by_area_spec().strip()
    except:
        pass
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        
        with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        if area_code not in data.get('areas', {}):
            return jsonify({"error": f"Area {area_code} not found"}), 404
        
        area_data = data['areas'][area_code]
        
        return jsonify({
            "area_code": area_code,
            "area_name": area_data.get('name'),
            "machines": area_data.get('machines', {})
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@machine_bp.route('/api/machine_screens/<machine_code>', methods=['GET'])
def get_machine_screens(machine_code):
    """Get screens for specific machine"""
    try:
        from utils.swagger_specs import get_machine_screens_spec
        get_machine_screens.__doc__ = get_machine_screens_spec().strip()
    except:
        pass
    try:
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        
        with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Find machine in areas
        for area_code, area_info in data.get('areas', {}).items():
            machines = area_info.get('machines', {})
            if machine_code in machines:
                machine_info = machines[machine_code]
                machine_type = machine_info.get('type')
                
                # Get screens from machine_types
                screens = []
                if 'machine_types' in data and machine_type in data['machine_types']:
                    screens = data['machine_types'][machine_type].get('screens', [])
                
                return jsonify({
                    "machine_code": machine_code,
                    "machine_type": machine_type,
                    "machine_name": machine_info.get('name'),
                    "screens": screens
                }), 200
        
        return jsonify({"error": "Machine not found"}), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@machine_bp.route('/api/set_machine_screen', methods=['POST'])
def set_machine_screen():
    """Set current machine and screen"""
    try:
        from utils.swagger_specs import get_set_machine_screen_spec
        set_machine_screen.__doc__ = get_set_machine_screen_spec().strip()
    except:
        pass
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        machine_code = data.get('machine_code')
        screen_id = data.get('screen_id')
        
        if not machine_code or not screen_id:
            return jsonify({"error": "Missing machine_code or screen_id"}), 400
        
        # Save to file
        success = save_current_machine_info(machine_code, screen_id)
        
        if success:
            return jsonify({
                "message": "Machine and screen set successfully",
                "machine_code": machine_code,
                "screen_id": screen_id
            }), 200
        else:
            return jsonify({"error": "Failed to save machine info"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@machine_bp.route('/api/current_machine_screen', methods=['GET'])
def get_current_machine_screen():
    """Get current machine and screen"""
    try:
        from utils.swagger_specs import get_current_machine_screen_spec
        get_current_machine_screen.__doc__ = get_current_machine_screen_spec().strip()
    except:
        pass
    try:
        machine_info = get_current_machine_info()
        
        return jsonify(machine_info), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@machine_bp.route('/api/machine_screen_status', methods=['GET'])
def check_machine_screen_status():
    """Check if machine and screen are configured"""
    try:
        from utils.swagger_specs import get_machine_screen_status_spec
        check_machine_screen_status.__doc__ = get_machine_screen_status_spec().strip()
    except:
        pass
    try:
        machine_info = get_current_machine_info()
        
        machine_code = machine_info.get('machine_code')
        screen_id = machine_info.get('screen_id')
        
        is_configured = bool(machine_code and screen_id)
        
        return jsonify({
            "is_configured": is_configured,
            "machine_code": machine_code,
            "screen_id": screen_id
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@machine_bp.route('/api/update_machine_screen', methods=['POST'])
def update_machine_screen():
    """Update machine and screen with parameter_order_value.txt"""
    try:
        from utils.swagger_specs import get_update_machine_screen_spec
        update_machine_screen.__doc__ = get_update_machine_screen_spec().strip()
    except:
        pass
    try:
        if 'machine_code' not in request.form or 'screen_id' not in request.form:
            return jsonify({
                "error": "Missing required fields: machine_code and screen_id"
            }), 400
        
        machine_code = request.form['machine_code'].strip().upper()
        screen_name = request.form['screen_id'].strip()
        area = request.form.get('area', '').strip().upper()
        
        # Get area if not provided
        if not area:
            area = get_area_for_machine(machine_code)
            if not area:
                return jsonify({
                    "error": "Could not determine area for this machine_code"
                }), 400
        
        # Load machine_screens.json
        machine_screens_path = os.path.join(get_roi_data_folder(), 'machine_screens.json')
        if not os.path.exists(machine_screens_path):
            return jsonify({"error": "Machine screens configuration not found"}), 404
        
        with open(machine_screens_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Validate area
        if area not in data.get('areas', {}):
            return jsonify({"error": f"Area {area} not found"}), 404
        
        # Validate machine
        if machine_code not in data['areas'][area].get('machines', {}):
            return jsonify({"error": f"Machine {machine_code} not found in area {area}"}), 404
        
        # Get machine type
        machine_type = data['areas'][area]['machines'][machine_code].get('type')
        if not machine_type or machine_type not in data.get('machine_types', {}):
            return jsonify({"error": f"Machine type not found for machine {machine_code}"}), 404
        
        # Find screen by name
        screen_numeric_id = None
        selected_screen = None
        for screen in data['machine_types'][machine_type].get('screens', []):
            if screen['screen_id'] == screen_name:
                screen_numeric_id = screen['id']
                selected_screen = screen
                break
        
        if not screen_numeric_id:
            return jsonify({
                "error": f"Screen '{screen_name}' not found for machine {machine_code}"
            }), 404
        
        # Update parameter_order_value.txt
        parameter_order_file = os.path.join(get_roi_data_folder(), 'parameter_order_value.txt')
        with open(parameter_order_file, 'w', encoding='utf-8') as f:
            f.write(str(screen_numeric_id))
        
        # Also save to current_machine_screen.json for consistency
        save_current_machine_info(machine_code, screen_name)
        
        return jsonify({
            "message": "Machine and screen updated successfully",
            "area": {
                "area_code": area,
                "name": data['areas'][area]['name']
            },
            "machine": {
                "machine_code": machine_code,
                "name": data['areas'][area]['machines'][machine_code]['name'],
                "type": machine_type
            },
            "screen": {
                "id": screen_numeric_id,
                "screen_id": selected_screen['screen_id'],
                "description": selected_screen.get('description', '')
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to update: {str(e)}"}), 500

