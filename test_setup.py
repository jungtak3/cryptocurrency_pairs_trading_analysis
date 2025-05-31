#!/usr/bin/env python3
"""
Test script to verify the cryptocurrency pairs trading analysis setup
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'scipy', 'statsmodels']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    return missing

def check_data_files():
    """Check if required data files exist"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    required_files = [
        'BTCUSDT_20170817_20250430.csv',
        'ETHUSDT_20170817_20250430.csv',
        'LTCUSDT_20171213_20250430.csv',
        'NEOUSDT_20171120_20250430.csv'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return missing_files

def test_basic_import():
    """Test if the analysis module can be imported"""
    try:
        from cryptocurrency_pairs_trading_analysis import run_analysis
        print("✅ Analysis module import - OK")
        return True
    except Exception as e:
        print(f"❌ Analysis module import - FAILED: {e}")
        return False

def test_report_generator():
    """Test if the report generator can be imported"""
    try:
        from markdown_to_pdf_generator import MarkdownPDFGenerator
        print("✅ Report generator import - OK")
        return True
    except Exception as e:
        print(f"❌ Report generator import - FAILED: {e}")
        return False

def main():
    print("=== Cryptocurrency Pairs Trading Analysis Setup Test ===\n")
    
    print("1. Checking Python dependencies...")
    missing_deps = check_dependencies()
    
    print("\n2. Checking data files...")
    missing_files = check_data_files()
    
    print("\n3. Testing module imports...")
    import_ok = test_basic_import() and test_report_generator()
    
    print("\n=== Test Results ===")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
    else:
        print("✅ All dependencies installed")
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        print("Place CSV files in the data/ directory (see data/README.md)")
    else:
        print("✅ All data files present")
    
    if import_ok:
        print("✅ Module imports working")
    else:
        print("❌ Module import issues")
    
    if not missing_deps and not missing_files and import_ok:
        print("\n🎉 Setup is complete! You can now run:")
        print("   python examples/generate_reports.py")
    else:
        print("\n⚠️  Setup incomplete. Please resolve the issues above.")

if __name__ == "__main__":
    main()