"""
Script to inject Swagger documentation into all route functions
Run this after server starts to apply Swagger specs
"""

def inject_all_swagger_docs():
    """Inject Swagger documentation into all routes"""
    try:
        # Import all route modules
        from routes import machine_routes, reference_routes
        from utils import swagger_specs
        
        # Machine routes
        machine_routes.get_all_machines.__doc__ = swagger_specs.get_machines_spec().strip()
        machine_routes.get_machines_by_area.__doc__ = swagger_specs.get_machines_by_area_spec().strip()
        machine_routes.get_machine_screens.__doc__ = swagger_specs.get_machine_screens_spec().strip()
        machine_routes.set_machine_screen.__doc__ = swagger_specs.get_set_machine_screen_spec().strip()
        machine_routes.get_current_machine_screen.__doc__ = swagger_specs.get_current_machine_screen_spec().strip()
        machine_routes.get_machine_screen_status.__doc__ = swagger_specs.get_machine_screen_status_spec().strip()
        machine_routes.update_machine_screen.__doc__ = swagger_specs.get_update_machine_screen_spec().strip()
        
        # Reference routes
        reference_routes.upload_reference_image.__doc__ = swagger_specs.get_reference_images_post_spec().strip()
        reference_routes.get_reference_images.__doc__ = swagger_specs.get_reference_images_list_spec().strip()
        reference_routes.get_reference_image.__doc__ = swagger_specs.get_reference_image_spec().strip()
        reference_routes.delete_reference_image.__doc__ = swagger_specs.get_delete_reference_image_spec().strip()
        
        print("[OK] Swagger documentation injected into all routes")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to inject Swagger docs: {e}")
        return False


if __name__ == "__main__":
    inject_all_swagger_docs()

