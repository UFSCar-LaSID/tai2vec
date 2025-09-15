import time
import functools
import threading
import psutil
import gc

# Try to import both frameworks
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def monitor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Check if GPU is available and determine framework
        gpu_available = False
        start_gpu = 0
        peak_gpu = 0
        framework = None
        
        # Try PyTorch first (since it's more commonly used now)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                start_gpu = torch.cuda.memory_allocated() / 1024**3
                peak_gpu = start_gpu
                gpu_available = True
                framework = 'pytorch'
            except Exception as e:
                print(f"[{func.__name__}] PyTorch GPU monitoring failed: {e}")
        
        # Fallback to TensorFlow if PyTorch not available or failed
        if not gpu_available and TF_AVAILABLE:
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                start_gpu = gpu_info['current'] / 1024**3
                peak_gpu = start_gpu
                gpu_available = True
                framework = 'tensorflow'
            except (RuntimeError, ValueError, AttributeError) as e:
                print(f"[{func.__name__}] TensorFlow GPU monitoring failed: {e}")
        
        if not gpu_available:
            print(f"[{func.__name__}] GPU monitoring unavailable: No compatible framework found")
        
        start_ram = psutil.virtual_memory().used / 1024**3
        peak_ram = start_ram
        
        monitoring = True
        
        def check_peak():
            nonlocal peak_gpu, peak_ram
            while monitoring:
                if gpu_available:
                    try:
                        if framework == 'pytorch':
                            torch.cuda.synchronize()
                            current_gpu = torch.cuda.memory_allocated() / 1024**3
                            peak_gpu = max(peak_gpu, current_gpu)
                        elif framework == 'tensorflow':
                            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                            current_gpu = gpu_info['current'] / 1024**3
                            peak_gpu = max(peak_gpu, current_gpu)
                    except:
                        pass  
                
                current_ram = psutil.virtual_memory().used / 1024**3
                peak_ram = max(peak_ram, current_ram)
                
                time.sleep(0.05)
        
        monitor_thread = threading.Thread(target=check_peak, daemon=True)
        monitor_thread.start()
        
        result = func(*args, **kwargs)
        
        monitoring = False
        monitor_thread.join(timeout=0.1)
        
        end_time = time.time()
        
        end_gpu = 0
        if gpu_available:
            try:
                if framework == 'pytorch':
                    torch.cuda.synchronize()
                    end_gpu = torch.cuda.memory_allocated() / 1024**3
                elif framework == 'tensorflow':
                    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                    end_gpu = gpu_info['current'] / 1024**3
            except:
                gpu_available = False
        
        end_ram = psutil.virtual_memory().used / 1024**3
        
        if gpu_available:
            framework_name = framework.capitalize()
            print(f"[{func.__name__}] Time: {end_time-start_time:.2f}s, GPU ({framework_name}): {end_gpu:.3f}GB ({end_gpu-start_gpu:+.3f}GB, Peak: {peak_gpu:.3f}GB), RAM: {end_ram:.1f}GB ({end_ram-start_ram:+.1f}GB, Peak: {peak_ram:.1f}GB)")
        else:
            print(f"[{func.__name__}] Time: {end_time-start_time:.2f}s, RAM: {end_ram:.1f}GB ({end_ram-start_ram:+.1f}GB, Peak: {peak_ram:.1f}GB) [GPU monitoring unavailable]")
        
        return result
    return wrapper

def memory_wipe(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Clear memory for both frameworks
        if TF_AVAILABLE:
            try:
                tf.keras.backend.clear_session()
                tf.config.experimental.reset_memory_stats('GPU:0')
            except (RuntimeError, ValueError, AttributeError):
                pass
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except (RuntimeError, ValueError, AttributeError):
                pass
        
        # General garbage collection
        for _ in range(3):
            gc.collect()
        
        result = func(*args, **kwargs)
        
        # Post-execution cleanup
        if TF_AVAILABLE:
            try:
                tf.keras.backend.clear_session()
                dummy = tf.zeros((1,), dtype=tf.float32)
                del dummy
            except (RuntimeError, ValueError, AttributeError):
                pass 
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except (RuntimeError, ValueError, AttributeError):
                pass
        
        gc.collect()
        
        return result
    return wrapper