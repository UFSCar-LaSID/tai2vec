import time
import functools
import threading
import tensorflow as tf
import psutil
import gc

def monitor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Check if GPU is available
        gpu_available = False
        start_gpu = 0
        peak_gpu = 0
        
        try:
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            start_gpu = gpu_info['current'] / 1024**3
            peak_gpu = start_gpu
            gpu_available = True
        except (RuntimeError, ValueError, AttributeError) as e:
            print(f"[{func.__name__}] GPU monitoring unavailable: {e}")
            gpu_available = False
        
        start_ram = psutil.virtual_memory().used / 1024**3
        peak_ram = start_ram
        
        monitoring = True
        
        def check_peak():
            nonlocal peak_gpu, peak_ram
            while monitoring:
                if gpu_available:
                    try:
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

        fragmentation_ratio = 0
        if gpu_available:
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                fragmentation_ratio = memory_info['peak'] / memory_info['current'] if memory_info['current'] > 0 else 1.0
                print(f"[{func.__name__}] Memory fragmentation ratio: {fragmentation_ratio:.2f}")
            except:
                pass
        
        monitoring = False
        monitor_thread.join(timeout=0.1)
        
        end_time = time.time()
        
        end_gpu = 0
        if gpu_available:
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                end_gpu = gpu_info['current'] / 1024**3
            except:
                gpu_available = False
        
        end_ram = psutil.virtual_memory().used / 1024**3
        
        if gpu_available:
            print(f"[{func.__name__}] Time: {end_time-start_time:.2f}s, GPU: {end_gpu:.3f}GB ({end_gpu-start_gpu:+.3f}GB, Peak: {peak_gpu:.3f}GB), RAM: {end_ram:.1f}GB ({end_ram-start_ram:+.1f}GB, Peak: {peak_ram:.1f}GB)")
        else:
            print(f"[{func.__name__}] Time: {end_time-start_time:.2f}s, RAM: {end_ram:.1f}GB ({end_ram-start_ram:+.1f}GB, Peak: {peak_ram:.1f}GB) [GPU monitoring unavailable]")
        
        return result
    return wrapper

def memory_wipe(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tf.keras.backend.clear_session()
        
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
        except (RuntimeError, ValueError, AttributeError):
            pass
        
        for _ in range(3):
            gc.collect()
        
        result = func(*args, **kwargs)
        
        tf.keras.backend.clear_session()
        
        try:
            dummy = tf.zeros((1,), dtype=tf.float32)
            del dummy
        except (RuntimeError, ValueError, AttributeError):
            pass 
        
        gc.collect()
        
        return result
    return wrapper