import os
import sys
from pathlib import Path

# trae给出的一个修复老包不兼容问题的神奇的脚本

print("--- TensorBoard Patcher Script (Smart Version) ---")

# 1. 优先使用 CONDA_PREFIX 环境变量来定位 site-packages
conda_prefix = os.environ.get("CONDA_PREFIX")
site_packages = None

if conda_prefix and Path(conda_prefix).is_dir():
    print(f"Found active conda environment via CONDA_PREFIX: {conda_prefix}")
    # Conda 环境的 site-packages 通常在 <prefix>/lib/site-packages
    conda_sp = Path(conda_prefix) / "lib" / "site-packages"
    if conda_sp.is_dir():
        site_packages = conda_sp
        print(f"Using conda site-packages directory: {site_packages}")
    else:
        print(f"Warning: Could not find 'lib/site-packages' in {conda_prefix}. Will search sys.path.")
else:
    print("Info: CONDA_PREFIX environment variable not found. Searching sys.path...")

# 2. 如果 CONDA_PREFIX 失败，则回退到搜索 sys.path
if not site_packages:
    for p in sys.path:
        # 优先选择路径中包含 'conda' 或 'envs' 的，以避免系统全局路径
        if "site-packages" in p and ("conda" in p or "envs" in p) and Path(p).is_dir():
            site_packages = Path(p)
            print(f"Found a likely conda site-packages in sys.path: {site_packages}")
            break
    # 如果还是没找到，就只能用第一个找到的 site-packages
    if not site_packages:
         for p in sys.path:
            if "site-packages" in p and Path(p).is_dir():
                site_packages = Path(p)
                print(f"Warning: Could not find a clear conda path. Using first available site-packages: {site_packages}")
                break

if not site_packages:
    print("\nFatal Error: Could not find any site-packages directory.")
    sys.exit(1)

# 3. 构建并修复目标文件 (后续逻辑不变)
file_to_patch = site_packages / "torch" / "utils" / "tensorboard" / "__init__.py"
print(f"Target file: {file_to_patch}")

if not file_to_patch.exists():
    print(f"\nError: Target file does not exist at the expected location.")
    print(f"Please ensure PyTorch is installed correctly in the '{os.environ.get('CONDA_DEFAULT_ENV', 'current')}' environment.")
    sys.exit(1)

try:
    print("\nReading file content...")
    original_content = file_to_patch.read_text()

    old_line_1 = "from distutils.version import LooseVersion"
    old_line_2 = "LooseVersion = distutils.version.LooseVersion"
    new_line = "from packaging.version import Version as LooseVersion"

    if new_line in original_content:
        print("File already appears to be patched. No changes made.")
    elif old_line_1 in original_content:
        print(f"Found old line: '{old_line_1}'. Replacing...")
        modified_content = original_content.replace(old_line_1, new_line)
        file_to_patch.write_text(modified_content)
        print("Successfully patched the file!")
    elif old_line_2 in original_content:
        print(f"Found old line: '{old_line_2}'. Replacing...")
        modified_content = original_content.replace(old_line_2, new_line)
        file_to_patch.write_text(modified_content)
        print("Successfully patched the file!")
    else:
        print("\nError: Could not find the line to replace. The file might have a different structure than expected.")
        sys.exit(1)

except Exception as e:
    print(f"\nAn unexpected error occurred during patching: {e}")
    sys.exit(1)

print("\n--- Patching process finished. ---")