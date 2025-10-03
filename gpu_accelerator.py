#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[*] GPU Accelerator Module
Module tối ưu hóa các operations bằng GPU với CuPy và OpenCV CUDA

Features:
- Automatic GPU detection and fallback to CPU
- CuPy wrapper for NumPy operations
- OpenCV CUDA support
- Memory management for GPU
- Performance monitoring

Author: GPU Optimization Team
Updated: 2025
"""

import numpy as np
import cv2
import sys
import warnings
warnings.filterwarnings('ignore')

# GPU availability flags
HAS_CUPY = False
HAS_OPENCV_CUDA = False
HAS_TORCH_CUDA = False

# Try to import CuPy
try:
    import cupy as cp
    HAS_CUPY = True
    print("[OK] CuPy GPU acceleration available")
    print(f"   - GPU Device: {cp.cuda.Device().id}")
    print(f"   - GPU Memory: {cp.cuda.Device().mem_info[1] / (1024**3):.2f} GB")
except ImportError:
    print("[WARNING] CuPy not installed - Using CPU for NumPy operations")
    cp = None
except Exception as e:
    print(f"[WARNING] CuPy initialization error: {e}")
    cp = None

# Check OpenCV CUDA support
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        HAS_OPENCV_CUDA = True
        print(f"[OK] OpenCV CUDA available - {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
    else:
        print("[WARNING] OpenCV CUDA not available - Using CPU for OpenCV")
except:
    print("[WARNING] OpenCV not compiled with CUDA support")

# Check PyTorch CUDA
try:
    import torch
    if torch.cuda.is_available():
        HAS_TORCH_CUDA = True
        print(f"[OK] PyTorch CUDA available - {torch.cuda.device_count()} device(s)")
        print(f"   - Device: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] PyTorch CUDA not available")
except:
    print("[WARNING] PyTorch not installed")


class GPUAccelerator:
    """
    GPU Accelerator class với automatic fallback to CPU
    """
    
    def __init__(self, force_cpu=False):
        """
        Initialize GPU Accelerator
        
        Args:
            force_cpu: Bắt buộc dùng CPU ngay cả khi có GPU
        """
        self.use_gpu = not force_cpu and HAS_CUPY
        self.use_opencv_cuda = not force_cpu and HAS_OPENCV_CUDA
        self.device_type = "GPU" if self.use_gpu else "CPU"
        
        if self.use_gpu:
            print(f"[*] GPU Accelerator initialized - Using {self.device_type}")
        else:
            print(f"[CPU] GPU Accelerator initialized - Using {self.device_type} (GPU not available or disabled)")
    
    def to_gpu(self, array):
        """
        Chuyển numpy array lên GPU (CuPy array)
        
        Args:
            array: NumPy array
            
        Returns:
            CuPy array nếu GPU available, ngược lại trả về NumPy array
        """
        if self.use_gpu and array is not None:
            try:
                return cp.asarray(array)
            except:
                return array
        return array
    
    def to_cpu(self, array):
        """
        Chuyển array từ GPU về CPU
        
        Args:
            array: CuPy hoặc NumPy array
            
        Returns:
            NumPy array
        """
        if self.use_gpu and array is not None:
            try:
                if isinstance(array, cp.ndarray):
                    return cp.asnumpy(array)
            except:
                pass
        return array
    
    def array_op(self, operation, *args, **kwargs):
        """
        Thực hiện operation trên array với GPU acceleration
        
        Args:
            operation: Function name (e.g., 'mean', 'sum', 'std')
            *args: Arguments cho operation
            **kwargs: Keyword arguments
            
        Returns:
            Kết quả của operation
        """
        if self.use_gpu:
            try:
                # Chuyển args lên GPU
                gpu_args = [self.to_gpu(arg) if isinstance(arg, np.ndarray) else arg 
                           for arg in args]
                
                # Thực hiện operation trên GPU
                if hasattr(cp, operation):
                    result = getattr(cp, operation)(*gpu_args, **kwargs)
                    return self.to_cpu(result)
            except Exception as e:
                print(f"[WARNING] GPU operation failed, falling back to CPU: {e}")
        
        # Fallback to CPU (NumPy)
        if hasattr(np, operation):
            return getattr(np, operation)(*args, **kwargs)
        else:
            raise ValueError(f"Operation '{operation}' not found")
    
    def mean(self, array, axis=None):
        """Calculate mean with GPU acceleration"""
        return self.array_op('mean', array, axis=axis)
    
    def sum(self, array, axis=None):
        """Calculate sum with GPU acceleration"""
        return self.array_op('sum', array, axis=axis)
    
    def std(self, array, axis=None):
        """Calculate standard deviation with GPU acceleration"""
        return self.array_op('std', array, axis=axis)
    
    def max(self, array, axis=None):
        """Calculate max with GPU acceleration"""
        return self.array_op('max', array, axis=axis)
    
    def min(self, array, axis=None):
        """Calculate min with GPU acceleration"""
        return self.array_op('min', array, axis=axis)
    
    def resize_gpu(self, image, size, interpolation=cv2.INTER_LINEAR):
        """
        Resize image với GPU acceleration (OpenCV CUDA)
        
        Args:
            image: Input image (NumPy array)
            size: Target size (width, height)
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        if self.use_opencv_cuda and image is not None:
            try:
                # Upload to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                # Resize on GPU
                gpu_resized = cv2.cuda.resize(gpu_img, size, interpolation=interpolation)
                
                # Download from GPU
                return gpu_resized.download()
            except Exception as e:
                print(f"[WARNING] GPU resize failed, falling back to CPU: {e}")
        
        # Fallback to CPU
        return cv2.resize(image, size, interpolation=interpolation)
    
    def cvt_color_gpu(self, image, code):
        """
        Convert color space với GPU acceleration
        
        Args:
            image: Input image
            code: Color conversion code (e.g., cv2.COLOR_BGR2GRAY)
            
        Returns:
            Converted image
        """
        if self.use_opencv_cuda and image is not None:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                gpu_converted = cv2.cuda.cvtColor(gpu_img, code)
                
                return gpu_converted.download()
            except Exception as e:
                print(f"[WARNING] GPU color conversion failed, falling back to CPU: {e}")
        
        # Fallback to CPU
        return cv2.cvtColor(image, code)
    
    def gaussian_blur_gpu(self, image, ksize, sigmaX):
        """
        Gaussian blur với GPU acceleration
        
        Args:
            image: Input image
            ksize: Kernel size (width, height)
            sigmaX: Gaussian kernel standard deviation in X direction
            
        Returns:
            Blurred image
        """
        if self.use_opencv_cuda and image is not None:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                # Create Gaussian filter
                gaussian_filter = cv2.cuda.createGaussianFilter(
                    image.dtype, -1, ksize, sigmaX
                )
                
                # Apply filter
                gpu_blurred = gaussian_filter.apply(gpu_img)
                
                return gpu_blurred.download()
            except Exception as e:
                print(f"[WARNING] GPU Gaussian blur failed, falling back to CPU: {e}")
        
        # Fallback to CPU
        return cv2.GaussianBlur(image, ksize, sigmaX)
    
    def threshold_gpu(self, image, thresh, maxval, type):
        """
        Threshold với GPU acceleration
        
        Args:
            image: Input image (grayscale)
            thresh: Threshold value
            maxval: Maximum value
            type: Threshold type
            
        Returns:
            Thresholded image
        """
        if self.use_opencv_cuda and image is not None:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                _, gpu_thresh = cv2.cuda.threshold(gpu_img, thresh, maxval, type)
                
                return gpu_thresh.download()
            except Exception as e:
                print(f"[WARNING] GPU threshold failed, falling back to CPU: {e}")
        
        # Fallback to CPU
        _, result = cv2.threshold(image, thresh, maxval, type)
        return result
    
    def get_memory_info(self):
        """
        Lấy thông tin memory GPU
        
        Returns:
            Dict chứa thông tin memory
        """
        if self.use_gpu:
            try:
                free, total = cp.cuda.Device().mem_info
                return {
                    'device': 'GPU',
                    'free_mb': free / (1024**2),
                    'total_mb': total / (1024**2),
                    'used_mb': (total - free) / (1024**2),
                    'usage_percent': ((total - free) / total) * 100
                }
            except:
                pass
        
        # CPU memory info (using psutil if available)
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'device': 'CPU',
                'free_mb': mem.available / (1024**2),
                'total_mb': mem.total / (1024**2),
                'used_mb': mem.used / (1024**2),
                'usage_percent': mem.percent
            }
        except:
            return {'device': 'CPU', 'info': 'Not available'}
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        if self.use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                print("[CLEAN] GPU memory cache cleared")
            except:
                pass


# Global GPU accelerator instance
_gpu_accelerator = None

def get_gpu_accelerator(force_cpu=False):
    """
    Get global GPU accelerator instance (Singleton pattern)
    
    Args:
        force_cpu: Force CPU mode
        
    Returns:
        GPUAccelerator instance
    """
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(force_cpu=force_cpu)
    return _gpu_accelerator


def is_gpu_available():
    """
    Check if GPU is available
    
    Returns:
        Boolean indicating GPU availability
    """
    return HAS_CUPY or HAS_OPENCV_CUDA or HAS_TORCH_CUDA


def get_gpu_info():
    """
    Get detailed GPU information
    
    Returns:
        Dict with GPU information
    """
    info = {
        'has_cupy': HAS_CUPY,
        'has_opencv_cuda': HAS_OPENCV_CUDA,
        'has_torch_cuda': HAS_TORCH_CUDA,
        'gpu_available': is_gpu_available()
    }
    
    if HAS_CUPY:
        try:
            info['cupy_version'] = cp.__version__
            info['gpu_device_id'] = cp.cuda.Device().id
            free, total = cp.cuda.Device().mem_info
            info['gpu_memory_total_gb'] = total / (1024**3)
            info['gpu_memory_free_gb'] = free / (1024**3)
        except:
            pass
    
    if HAS_OPENCV_CUDA:
        try:
            info['opencv_cuda_devices'] = cv2.cuda.getCudaEnabledDeviceCount()
        except:
            pass
    
    if HAS_TORCH_CUDA:
        try:
            import torch
            info['torch_cuda_devices'] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                info['torch_gpu_name'] = torch.cuda.get_device_name(0)
        except:
            pass
    
    return info


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("GPU ACCELERATOR TEST")
    print("=" * 70)
    
    # Create accelerator
    gpu = get_gpu_accelerator()
    
    # Test basic operations
    print("\n📊 Testing basic operations...")
    test_array = np.random.rand(1000, 1000).astype(np.float32)
    
    import time
    
    # Mean operation
    start = time.time()
    result = gpu.mean(test_array)
    elapsed = time.time() - start
    print(f"   Mean: {result:.6f} (Time: {elapsed*1000:.2f}ms)")
    
    # Sum operation
    start = time.time()
    result = gpu.sum(test_array)
    elapsed = time.time() - start
    print(f"   Sum: {result:.2f} (Time: {elapsed*1000:.2f}ms)")
    
    # Test image operations
    print("\n🖼️  Testing image operations...")
    test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    # Resize
    start = time.time()
    resized = gpu.resize_gpu(test_image, (512, 512))
    elapsed = time.time() - start
    print(f"   Resize (1024x1024 -> 512x512): {elapsed*1000:.2f}ms")
    
    # Color conversion
    start = time.time()
    gray = gpu.cvt_color_gpu(test_image, cv2.COLOR_BGR2GRAY)
    elapsed = time.time() - start
    print(f"   Color conversion (BGR2GRAY): {elapsed*1000:.2f}ms")
    
    # Memory info
    print("\n💾 Memory information:")
    mem_info = gpu.get_memory_info()
    for key, value in mem_info.items():
        print(f"   {key}: {value}")
    
    # GPU info
    print("\n📋 GPU Information:")
    gpu_info = get_gpu_info()
    for key, value in gpu_info.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("[OK] GPU Accelerator test completed!")
    print("=" * 70)

