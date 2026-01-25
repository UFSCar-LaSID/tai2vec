import time
import functools
import threading
import psutil
import torch

def monitor(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        gpu_available = False
        start_gpu = 0
        peak_gpu = 0
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                start_gpu = torch.cuda.memory_allocated() / 1024**3
                peak_gpu = start_gpu
                gpu_available = True
            except Exception as e:
                print(f"[{func.__name__}] PyTorch GPU monitoring failed: {e}")
        
        if not gpu_available:
            print(f"[{func.__name__}] GPU monitoring unavailable: PyTorch not found or no GPU available")
        
        start_ram = psutil.virtual_memory().used / 1024**3
        peak_ram = start_ram
        
        monitoring = True
        
        def check_peak():
            nonlocal peak_gpu, peak_ram
            while monitoring:
                if gpu_available:
                    try:
                        torch.cuda.synchronize()
                        current_gpu = torch.cuda.memory_allocated() / 1024**3
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
                torch.cuda.synchronize()
                end_gpu = torch.cuda.memory_allocated() / 1024**3
            except:
                gpu_available = False
        
        end_ram = psutil.virtual_memory().used / 1024**3
        
        if gpu_available:
            print(f"[{func.__name__}] Time: {end_time-start_time:.2f}s, GPU (PyTorch): {end_gpu:.3f}GB ({end_gpu-start_gpu:+.3f}GB, Peak: {peak_gpu:.3f}GB), RAM: {end_ram:.1f}GB ({end_ram-start_ram:+.1f}GB, Peak: {peak_ram:.1f}GB)")
        else:
            print(f"[{func.__name__}] Time: {end_time-start_time:.2f}s, RAM: {end_ram:.1f}GB ({end_ram-start_ram:+.1f}GB, Peak: {peak_ram:.1f}GB) [GPU monitoring unavailable]")
        
        return result
    
    return wrapper