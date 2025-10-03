"""
Routes package for HMI OCR API
Contains API route handlers organized by functionality
"""

from .image_routes import image_bp
from .machine_routes import machine_bp
from .decimal_routes import decimal_bp
from .reference_routes import reference_bp

__all__ = [
    'image_bp',
    'machine_bp',
    'decimal_bp',
    'reference_bp',
]

