import torch
import platform
import subprocess

print("--- SYSTEM & PYTORCH SPECIFICATIONS ---")
# 1. OS Version
print(f"Operating System: {platform.system()} {platform.release()}")

# 2. PyTorch & CUDA Version
print(f"PyTorch Version: {torch.__version__}")
print(f"Compiled CUDA Version: {torch.version.cuda}")

# 3. GPU & VRAM Details
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM: {round(vram_gb)} GB")
else:
    print("CUDA is not available. GPU not detected by PyTorch.")

# 4. NVIDIA Driver Version (runs nvidia-smi in the background)
try:
    smi_output = subprocess.check_output("nvidia-smi", shell=True).decode()
    # Grabs the first few lines of nvidia-smi which contain the driver version
    print("\n--- NVIDIA DRIVER INFO ---")
    print("\n".join(smi_output.split("\n")[1:3]))
except Exception as e:
    print("Could not run nvidia-smi to get driver version.")