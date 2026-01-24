import sys
import platform

# environment for local and cloud execution
print("-- Environment Verification --")
print(f"Python version: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

try: 
    import torch # type: ignore
except ImportError as e:
    print("PyTorch not installed.")
    print(f"Import error: {e}")
    sys.exit(0)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f" Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available (expected on macOS).")

print("Environment verification finished")