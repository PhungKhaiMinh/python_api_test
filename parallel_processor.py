#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[FAST] Parallel Processor Module
Module tối ưu hóa xử lý đa luồng và đa tiến trình

Features:
- Adaptive thread pool sizing
- Process pool for CPU-intensive tasks
- Batch processing optimization
- Task queue management
- Performance monitoring

Author: Parallel Processing Team
Updated: 2025
"""

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
import queue
import time
from typing import List, Callable, Any, Optional, Dict
import os

# Configuration
MAX_WORKERS = max(4, cpu_count() * 2)  # Số worker tối đa cho thread pool
PROCESS_POOL_SIZE = max(2, cpu_count())  # Số worker cho process pool
BATCH_SIZE = 8  # Kích thước batch mặc định

print(f"[*] Parallel Processor Configuration:")
print(f"   - CPU Cores: {cpu_count()}")
print(f"   - Thread Pool Workers: {MAX_WORKERS}")
print(f"   - Process Pool Workers: {PROCESS_POOL_SIZE}")
print(f"   - Default Batch Size: {BATCH_SIZE}")


class AdaptiveThreadPool:
    """
    Thread pool với khả năng tự động điều chỉnh số lượng worker
    """
    
    def __init__(self, min_workers=4, max_workers=None, name="ThreadPool"):
        """
        Initialize adaptive thread pool
        
        Args:
            min_workers: Số worker tối thiểu
            max_workers: Số worker tối đa (None = auto)
            name: Tên của thread pool
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or MAX_WORKERS
        self.name = name
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=name)
        self.active_tasks = 0
        self.lock = threading.Lock()
        
        print(f"[OK] {self.name} initialized: {self.min_workers}-{self.max_workers} workers")
    
    def submit(self, fn, *args, **kwargs):
        """
        Submit task to thread pool
        
        Args:
            fn: Function to execute
            *args, **kwargs: Arguments for function
            
        Returns:
            Future object
        """
        with self.lock:
            self.active_tasks += 1
        
        future = self.executor.submit(fn, *args, **kwargs)
        
        # Callback để giảm active_tasks khi hoàn thành
        def done_callback(f):
            with self.lock:
                self.active_tasks -= 1
        
        future.add_done_callback(done_callback)
        return future
    
    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """
        Map function over iterables with thread pool
        
        Args:
            fn: Function to execute
            *iterables: Input iterables
            timeout: Timeout in seconds
            chunksize: Chunk size for processing
            
        Returns:
            Iterator of results
        """
        return self.executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)
    
    def shutdown(self, wait=True):
        """Shutdown thread pool"""
        self.executor.shutdown(wait=wait)
    
    def get_stats(self):
        """Get thread pool statistics"""
        return {
            'name': self.name,
            'max_workers': self.max_workers,
            'active_tasks': self.active_tasks
        }


class BatchProcessor:
    """
    Processor để xử lý dữ liệu theo batch với parallelization
    """
    
    def __init__(self, batch_size=BATCH_SIZE, max_workers=None):
        """
        Initialize batch processor
        
        Args:
            batch_size: Kích thước batch
            max_workers: Số worker (None = auto)
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or MAX_WORKERS
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def process_batch(self, items: List[Any], process_fn: Callable, 
                     show_progress=False) -> List[Any]:
        """
        Process items in batches with parallelization
        
        Args:
            items: List of items to process
            process_fn: Function to process each item
            show_progress: Show progress indicator
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        # Nếu số lượng item ít, xử lý tuần tự
        if len(items) <= 2:
            return [process_fn(item) for item in items]
        
        # Tính số batch
        num_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        # Chia thành các batch
        batches = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(items))
            batches.append(items[start_idx:end_idx])
        
        # Xử lý các batch song song
        results = []
        
        def process_single_batch(batch):
            """Process a single batch"""
            return [process_fn(item) for item in batch]
        
        # Submit all batches
        futures = [self.executor.submit(process_single_batch, batch) 
                  for batch in batches]
        
        # Collect results
        for i, future in enumerate(as_completed(futures)):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                
                if show_progress:
                    progress = ((i + 1) / len(futures)) * 100
                    print(f"   Progress: {progress:.1f}% ({i+1}/{len(futures)} batches)")
            except Exception as e:
                print(f"[ERROR] Error processing batch: {e}")
        
        return results
    
    def process_parallel(self, items: List[Any], process_fn: Callable,
                        max_workers: Optional[int] = None) -> List[Any]:
        """
        Process all items in parallel (không batch)
        
        Args:
            items: List of items
            process_fn: Processing function
            max_workers: Override max workers
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        if len(items) == 1:
            return [process_fn(items[0])]
        
        # Sử dụng max_workers nếu được chỉ định
        workers = max_workers or min(len(items), self.max_workers)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_fn, item) for item in items]
            results = []
            
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"[ERROR] Error in parallel processing: {e}")
                    results.append(None)
        
        return results
    
    def shutdown(self):
        """Shutdown batch processor"""
        self.executor.shutdown(wait=True)


class TaskQueue:
    """
    Queue để quản lý tasks với priority
    """
    
    def __init__(self, max_workers=None):
        """
        Initialize task queue
        
        Args:
            max_workers: Số worker threads
        """
        self.queue = queue.PriorityQueue()
        self.max_workers = max_workers or MAX_WORKERS
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = False
        self.processed_count = 0
        self.lock = threading.Lock()
    
    def add_task(self, fn: Callable, args=(), kwargs=None, priority=5):
        """
        Add task to queue
        
        Args:
            fn: Function to execute
            args: Arguments
            kwargs: Keyword arguments
            priority: Priority (lower = higher priority)
        """
        if kwargs is None:
            kwargs = {}
        
        task = {
            'fn': fn,
            'args': args,
            'kwargs': kwargs,
            'priority': priority
        }
        
        self.queue.put((priority, time.time(), task))
    
    def process_queue(self, timeout=None):
        """
        Process all tasks in queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            List of results
        """
        self.running = True
        results = []
        
        start_time = time.time()
        
        while self.running:
            try:
                # Get task from queue với timeout
                try:
                    priority, timestamp, task = self.queue.get(timeout=0.1)
                except queue.Empty:
                    break
                
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    print("[WARNING] Queue processing timeout")
                    break
                
                # Execute task
                future = self.executor.submit(
                    task['fn'], *task['args'], **task['kwargs']
                )
                
                try:
                    result = future.result(timeout=30)  # 30s timeout per task
                    results.append(result)
                    
                    with self.lock:
                        self.processed_count += 1
                except Exception as e:
                    print(f"[ERROR] Task execution error: {e}")
                    results.append(None)
                
                self.queue.task_done()
                
            except Exception as e:
                print(f"[ERROR] Queue processing error: {e}")
                break
        
        self.running = False
        return results
    
    def get_stats(self):
        """Get queue statistics"""
        return {
            'queue_size': self.queue.qsize(),
            'processed_count': self.processed_count,
            'running': self.running
        }
    
    def clear(self):
        """Clear queue"""
        with self.queue.mutex:
            self.queue.queue.clear()
        self.processed_count = 0


class ParallelROIProcessor:
    """
    Specialized processor for ROI processing với intelligent batching
    """
    
    def __init__(self, max_workers=None):
        """
        Initialize ROI processor
        
        Args:
            max_workers: Max worker threads
        """
        self.max_workers = max_workers or MAX_WORKERS
        self.batch_processor = BatchProcessor(batch_size=BATCH_SIZE, max_workers=self.max_workers)
    
    def process_rois(self, roi_args_list: List[tuple], process_fn: Callable) -> List[Dict]:
        """
        Process multiple ROIs in parallel
        
        Args:
            roi_args_list: List of ROI arguments tuples
            process_fn: Function to process single ROI
            
        Returns:
            List of OCR results
        """
        if not roi_args_list:
            return []
        
        # Nếu ít ROI, xử lý tuần tự
        if len(roi_args_list) <= 2:
            return [process_fn(args) for args in roi_args_list]
        
        # Xử lý song song cho nhiều ROI
        return self.batch_processor.process_parallel(
            roi_args_list, 
            process_fn,
            max_workers=min(len(roi_args_list), self.max_workers)
        )
    
    def shutdown(self):
        """Shutdown processor"""
        self.batch_processor.shutdown()


# Global instances
_ocr_thread_pool = None
_image_thread_pool = None
_roi_processor = None

def get_ocr_thread_pool():
    """Get global OCR thread pool (Singleton)"""
    global _ocr_thread_pool
    if _ocr_thread_pool is None:
        _ocr_thread_pool = AdaptiveThreadPool(
            min_workers=4, 
            max_workers=MAX_WORKERS,
            name="OCR-ThreadPool"
        )
    return _ocr_thread_pool

def get_image_thread_pool():
    """Get global image processing thread pool (Singleton)"""
    global _image_thread_pool
    if _image_thread_pool is None:
        _image_thread_pool = AdaptiveThreadPool(
            min_workers=4,
            max_workers=MAX_WORKERS,
            name="Image-ThreadPool"
        )
    return _image_thread_pool

def get_roi_processor():
    """Get global ROI processor (Singleton)"""
    global _roi_processor
    if _roi_processor is None:
        _roi_processor = ParallelROIProcessor(max_workers=MAX_WORKERS)
    return _roi_processor


def parallel_map(func: Callable, items: List[Any], max_workers: Optional[int] = None,
                show_progress: bool = False) -> List[Any]:
    """
    Utility function để map function over items in parallel
    
    Args:
        func: Function to apply
        items: List of items
        max_workers: Max workers (None = auto)
        show_progress: Show progress
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    if len(items) == 1:
        return [func(items[0])]
    
    workers = max_workers or min(len(items), MAX_WORKERS)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(func, item): i for i, item in enumerate(items)}
        results = [None] * len(items)
        
        completed = 0
        total = len(items)
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
                completed += 1
                
                if show_progress and completed % max(1, total // 10) == 0:
                    print(f"   Progress: {(completed/total)*100:.1f}% ({completed}/{total})")
            except Exception as e:
                print(f"[ERROR] Error processing item {idx}: {e}")
                results[idx] = None
    
    return results


def get_system_stats():
    """
    Get system statistics
    
    Returns:
        Dict with system stats
    """
    stats = {
        'cpu_count': cpu_count(),
        'max_thread_workers': MAX_WORKERS,
        'max_process_workers': PROCESS_POOL_SIZE,
    }
    
    # Thread pool stats
    if _ocr_thread_pool:
        stats['ocr_pool'] = _ocr_thread_pool.get_stats()
    
    if _image_thread_pool:
        stats['image_pool'] = _image_thread_pool.get_stats()
    
    # Memory stats
    try:
        import psutil
        mem = psutil.virtual_memory()
        stats['memory'] = {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_percent': mem.percent
        }
        
        stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    except:
        pass
    
    return stats


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("PARALLEL PROCESSOR TEST")
    print("=" * 70)
    
    # Test batch processor
    print("\n[BATCH] Testing Batch Processor...")
    
    def dummy_process(x):
        """Dummy processing function"""
        time.sleep(0.01)  # Simulate work
        return x * 2
    
    items = list(range(20))
    batch_proc = BatchProcessor(batch_size=5)
    
    start = time.time()
    results = batch_proc.process_batch(items, dummy_process, show_progress=True)
    elapsed = time.time() - start
    
    print(f"   Processed {len(results)} items in {elapsed:.2f}s")
    print(f"   Results: {results[:5]}... (showing first 5)")
    
    # Test parallel map
    print("\n[FAST] Testing Parallel Map...")
    
    start = time.time()
    results = parallel_map(dummy_process, items, show_progress=True)
    elapsed = time.time() - start
    
    print(f"   Processed {len(results)} items in {elapsed:.2f}s")
    
    # Test task queue
    print("\n[QUEUE] Testing Task Queue...")
    
    task_queue = TaskQueue(max_workers=4)
    
    for i in range(10):
        task_queue.add_task(dummy_process, args=(i,), priority=i % 3)
    
    start = time.time()
    results = task_queue.process_queue()
    elapsed = time.time() - start
    
    print(f"   Processed {len(results)} tasks in {elapsed:.2f}s")
    print(f"   Queue stats: {task_queue.get_stats()}")
    
    # System stats
    print("\n[STATS] System Statistics:")
    stats = get_system_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("[OK] Parallel Processor test completed!")
    print("=" * 70)

