"""
Utility functions for monitoring and tracking model performance.
"""

import time
import psutil
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Track various performance metrics during model inference and training.
    """
    
    def __init__(self):
        self.metrics_history = {
            'inference_times': [],
            'memory_usage': [],
            'token_speeds': []
        }
    
    def start_tracking(self) -> float:
        """Start tracking a new operation. Returns start time."""
        return time.time()
    
    def record_inference(self, start_time: float, output_tokens: int) -> Dict[str, float]:
        """
        Record metrics for an inference operation.
        
        Args:
            start_time: Time when operation started
            output_tokens: Number of tokens generated
            
        Returns:
            Dictionary of recorded metrics
        """
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        memory_percent = psutil.virtual_memory().percent
        token_speed = output_tokens / duration if duration > 0 else 0
        
        # Record metrics
        self.metrics_history['inference_times'].append(duration)
        self.metrics_history['memory_usage'].append(memory_percent)
        self.metrics_history['token_speeds'].append(token_speed)
        
        # Return current metrics
        return {
            'duration': duration,
            'memory_percent': memory_percent,
            'tokens_per_second': token_speed
        }
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all tracked metrics.
        
        Returns:
            Dictionary with summary statistics for each metric
        """
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if not values:
                summary[metric_name] = {'count': 0}
                continue
                
            summary[metric_name] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }
        
        return summary
    
    def log_summary(self, level: int = logging.INFO) -> None:
        """
        Log summary statistics at the specified logging level.
        
        Args:
            level: Logging level (default: INFO)
        """
        summary = self.get_summary_stats()
        
        for metric_name, stats in summary.items():
            if stats.get('count', 0) == 0:
                continue
                
            logger.log(
                level,
                f"{metric_name}: mean={stats['mean']:.4f}, "
                f"median={stats['median']:.4f}, "
                f"min={stats['min']:.4f}, "
                f"max={stats['max']:.4f}, "
                f"std={stats['std']:.4f}, "
                f"count={stats['count']}"
            )
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        for key in self.metrics_history:
            self.metrics_history[key] = []


def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """
    Get current GPU memory usage if available.
    
    Returns:
        Dictionary with GPU memory stats or None if no GPU available
    """
    if not torch.cuda.is_available():
        return None
    
    memory_stats = {}
    
    # Get number of GPUs
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
        
        memory_stats[f'gpu_{i}'] = {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved
        }
    
    return memory_stats


def log_system_info() -> Dict[str, Any]:
    """
    Log system information including CPU, memory, and GPU if available.
    
    Returns:
        Dictionary with system information
    """
    # Get CPU info
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    
    # Get memory info
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024 ** 3)
    memory_available_gb = memory.available / (1024 ** 3)
    memory_used_gb = memory.used / (1024 ** 3)
    memory_percent = memory.percent
    
    # Get GPU info if available
    gpu_info = get_gpu_memory_usage()
    
    # Build system info dictionary
    system_info = {
        'cpu': {
            'percent': cpu_percent,
            'count': cpu_count
        },
        'memory': {
            'total_gb': memory_total_gb,
            'available_gb': memory_available_gb,
            'used_gb': memory_used_gb,
            'percent': memory_percent
        }
    }
    
    if gpu_info:
        system_info['gpu'] = gpu_info
    
    # Log the information
    logger.info(f"CPU Usage: {cpu_percent}% of {cpu_count} cores")
    logger.info(f"Memory: {memory_used_gb:.2f}GB / {memory_total_gb:.2f}GB ({memory_percent}%)")
    
    if gpu_info:
        for device, stats in gpu_info.items():
            logger.info(f"GPU {device}: Allocated: {stats['allocated_gb']:.2f}GB, Reserved: {stats['reserved_gb']:.2f}GB")
    
    return system_info


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Log system information
    system_info = log_system_info()
    
    # Create a performance tracker
    tracker = PerformanceTracker()
    
    # Simulate some operations
    for i in range(5):
        start = tracker.start_tracking()
        # Simulate work
        time.sleep(0.5)
        metrics = tracker.record_inference(start, 100)
        logger.info(f"Operation {i+1}: {metrics}")
    
    # Log summary
    tracker.log_summary()
