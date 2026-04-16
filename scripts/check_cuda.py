import sys
import os
import platform

print("="*60)
print("CINEMATCH AI — HARDWARE DIAGNOSTIC TOOL")
print("="*60)

# 1. Basic System Info
print(f"OS:               {platform.system()} {platform.release()}")
print(f"Python Version:   {sys.version.split()[0]}")
print(f"Execution Path:   {sys.executable}")
print("-" * 60)

# 2. Check for dependencies
try:
    import torch
    print(f"INFO: PyTorch:        {torch.__version__}")
except ImportError:
    print("ERR: PyTorch:        NOT INSTALLED")
    print("\n[!] Error: Core requirement 'torch' is missing.")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print(f"INFO: Transformers:   Installed")
except ImportError:
    print("ERR: Transformers:   NOT INSTALLED (sentence-transformers)")

print("-" * 60)

# 3. CUDA Availability Logic
cuda_available = torch.cuda.is_available()

if cuda_available:
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"STATUS:           GPU ACCELERATION ACTIVE")
    print(f"Device Name:      {device_name}")
    print(f"Device Count:     {device_count}")
    print(f"CUDA Version:     {torch.version.cuda}")
    print("-" * 60)
    print("Environment is correctly configured for high-speed AI inference.")
else:
    print(f"STATUS:           CPU MODE (DEGRADED PERFORMANCE)")
    print("-" * 60)
    print("DIAGNOSIS:")
    
    # Check if this is a CPU-only build
    if "+cpu" in torch.__version__:
        print("   -> You have the CPU-only build of PyTorch installed.")
        print("   -> Pip prioritized the standard registry instead of the CUDA-enabled one.")
    elif platform.system() == "Windows" and sys.version_info.major == 3 and sys.version_info.minor == 14:
        print("   -> You are using Python 3.14 on Windows.")
        print("   -> Python 3.14 often lacks pre-compiled CUDA wheels for this Torch version.")
    else:
        print("   -> PyTorch cannot see your NVIDIA drivers or compatible hardware.")

    print("\nFIX (Run these commands in your terminal):")
    print("   1. Activate your environment: .\\venv\\Scripts\\activate")
    print("   2. Force CUDA reinstall:")
    print("      pip uninstall torch torchvision torchaudio -y")
    print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("=" * 60)
