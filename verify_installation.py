"""
Installation Verification Script
--------------------------------
Verifies that all dependencies are installed correctly.
"""

import sys
from importlib import import_module


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_package(package_name, display_name=None):
    """Check if a package is installed."""
    if display_name is None:
        display_name = package_name
    
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name:20s} {version}")
        return True
    except ImportError:
        print(f"❌ {display_name:20s} NOT INSTALLED")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("CHECKING INSTALLATION")
    print("="*60)
    
    all_ok = True
    
    # Check Python version
    print("\n1. Python Version")
    print("-" * 60)
    all_ok &= check_python_version()
    
    # Check core packages
    print("\n2. Core ML Libraries")
    print("-" * 60)
    all_ok &= check_package('pandas', 'pandas')
    all_ok &= check_package('numpy', 'numpy')
    all_ok &= check_package('sklearn', 'scikit-learn')
    all_ok &= check_package('xgboost', 'xgboost')
    all_ok &= check_package('imblearn', 'imbalanced-learn')
    
    # Check API packages
    print("\n3. API Framework")
    print("-" * 60)
    all_ok &= check_package('fastapi', 'fastapi')
    all_ok &= check_package('uvicorn', 'uvicorn')
    all_ok &= check_package('pydantic', 'pydantic')
    
    # Check visualization packages
    print("\n4. Visualization")
    print("-" * 60)
    all_ok &= check_package('matplotlib', 'matplotlib')
    all_ok &= check_package('seaborn', 'seaborn')
    
    # Check utilities
    print("\n5. Utilities")
    print("-" * 60)
    all_ok &= check_package('joblib', 'joblib')
    
    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to go! Next steps:")
        print("1. python generate_sample_data.py")
        print("2. python train_pipeline.py")
        print("3. python api/main.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        print("\nThen run this script again.")
    
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
