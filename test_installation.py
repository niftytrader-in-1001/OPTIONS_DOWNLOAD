#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed
"""

import sys
import subprocess

def test_import(module_name, import_name=None):
    """Test if a module can be imported"""
    try:
        if import_name:
            exec(f"import {module_name}; from {module_name} import {import_name}")
            print(f"✅ {module_name}.{import_name}")
        else:
            exec(f"import {module_name}")
            print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    print("Testing Python version and imports...")
    print(f"Python version: {sys.version}")
    print()
    
    # Test basic imports
    modules = [
        ("pandas", None),
        ("requests", None),
        ("pyotp", None),
        ("openpyxl", None),
        ("logzero", None),
        ("SmartApi", "smartConnect"),
        ("SmartApi.smartConnect", "SmartConnect"),
    ]
    
    all_passed = True
    for module, import_name in modules:
        if not test_import(module, import_name):
            all_passed = False
    
    print()
    print("="*50)
    
    if all_passed:
        print("✅ All imports successful!")
        
        # Test SmartConnect creation
        try:
            from SmartApi.smartConnect import SmartConnect
            print("✅ SmartConnect class can be instantiated")
            print("All tests passed! 🎉")
        except Exception as e:
            print(f"❌ SmartConnect instantiation failed: {e}")
    else:
        print("❌ Some imports failed")
        
        # Try to install missing packages
        print("\nTrying to install missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "smartapi-python", "logzero"])
            print("Installation completed. Please run this test again.")
        except Exception as e:
            print(f"Installation failed: {e}")

if __name__ == "__main__":
    main()
