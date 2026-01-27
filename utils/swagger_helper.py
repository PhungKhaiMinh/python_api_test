"""
Swagger Helper Module
Helper functions for injecting Swagger documentation into routes
"""

def inject_docstring(func, spec_func):
    """
    Inject Swagger docstring into a function
    
    Args:
        func: Function to inject docstring into
        spec_func: Function that returns the Swagger spec string
    """
    try:
        func.__doc__ = spec_func()
    except Exception as e:
        # Fallback to default docstring if spec function fails
        if not func.__doc__:
            func.__doc__ = "API endpoint documentation"

