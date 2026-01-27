"""
Swagger UI Configuration Module
Handles all Swagger/OpenAPI documentation and configuration
"""

def get_swagger_config():
    """Get Swagger configuration"""
    return {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs"
    }


def get_swagger_template():
    """Get Swagger template configuration"""
    return {
        "swagger": "2.0",
        "info": {
            "title": "HMI OCR API Server",
            "description": "API documentation for HMI OCR system with image processing, OCR, and configuration management",
            "version": "2.0",
            "contact": {
                "name": "API Support"
            }
        },
        "schemes": ["http", "https"],
        "basePath": "/",
        "tags": [
            {
                "name": "Image Processing",
                "description": "Endpoints for image upload, OCR processing, and HMI detection"
            },
            {
                "name": "Decimal Places",
                "description": "Endpoints for managing decimal places configuration"
            },
            {
                "name": "Machine Management",
                "description": "Endpoints for machine and screen information"
            },
            {
                "name": "History",
                "description": "Endpoints for querying OCR history"
            },
            {
                "name": "Reference Images",
                "description": "Endpoints for managing reference template images"
            },
            {
                "name": "System",
                "description": "System information and performance endpoints"
            }
        ]
    }


def init_swagger(app):
    """
    Initialize Swagger UI for Flask app
    
    Args:
        app: Flask application instance
        
    Returns:
        Swagger instance or None if Flasgger is not available
    """
    try:
        from flasgger import Swagger
        config = get_swagger_config()
        template = get_swagger_template()
        swagger = Swagger(app, config=config, template=template)
        print("[OK] Swagger UI initialized - Available at /apidocs")
        return swagger
    except ImportError:
        print("[WARNING] Flasgger not installed - Swagger UI unavailable. Install with: pip install flasgger")
        return None

