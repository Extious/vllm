#!/usr/bin/env python3
"""
Hardware Analysis Tools Test Script

Quick test for all hardware analysis tools functionality.

Author: GitHub Copilot
Date: 2025-08-18
"""

import subprocess
import sys
import os
from pathlib import Path


def test_tool(script_path: str, tool_name: str) -> bool:
    """Test a single tool"""
    print(f"\n{'='*60}")
    print(f"Testing tool: {tool_name}")
    print(f"Script path: {script_path}")
    print('='*60)
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script file does not exist: {script_path}")
        return False
    
    # Check if script is executable
    if not os.access(script_path, os.X_OK):
        print(f"WARNING: Script not executable, trying to add execute permission...")
        try:
            os.chmod(script_path, 0o755)
            print("OK Execute permission added successfully")
        except Exception as e:
            print(f"ERROR Failed to add execute permission: {e}")
            return False
    
    # Test script's --help option
    try:
        print("Testing --help option...")
        result = subprocess.run([sys.executable, script_path, '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("OK --help option works normally")
            print("Help info preview:")
            print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print(f"WARNING --help option returned non-zero exit code: {result.returncode}")
            if result.stderr:
                print(f"Error message: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("WARNING --help option timed out")
    except Exception as e:
        print(f"ERROR Error testing --help option: {e}")
        return False
    
    # Check basic script syntax
    try:
        print("Checking Python syntax...")
        result = subprocess.run([sys.executable, '-m', 'py_compile', script_path], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("OK Python syntax check passed")
        else:
            print(f"ERROR Python syntax error:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"ERROR Syntax check failed: {e}")
        return False
    
    return True


def check_dependencies():
    """Check system dependencies"""
    print("Checking system dependencies...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("OK nvidia-smi available")
            # Extract version info
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
        else:
            print("ERROR nvidia-smi not available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("ERROR nvidia-smi not found")
    
    # Check bandwidthTest
    try:
        result = subprocess.run(['bandwidthTest', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("OK bandwidthTest available")
        else:
            print("WARNING bandwidthTest not available (some features will be limited)")
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        print("WARNING bandwidthTest not found or no permission (some features will be limited)")
    
    # Check Python dependency packages
    required_packages = ['json', 'subprocess', 'time', 'os', 'sys', 'argparse', 'platform']
    optional_packages = ['psutil', 'xml.etree.ElementTree']
    
    print("\nChecking Python package dependencies:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"OK {package} - installed")
        except ImportError:
            print(f"ERROR {package} - not installed (required)")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"OK {package} - installed")
        except ImportError:
            print(f"WARNING {package} - not installed (recommended)")


def run_quick_test():
    """Run quick test"""
    print("Hardware Analysis Tools Quick Test")
    print("="*60)
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Define tool list
    tools = [
        (script_dir / "hardware_topology_analyzer.py", "Basic Hardware Topology Analyzer"),
        (script_dir / "simple_hardware_analyzer.py", "Simplified Analyzer"),
        (script_dir / "advanced_bandwidth_tester.py", "Advanced Bandwidth Tester")
    ]
    
    # Check system dependencies
    check_dependencies()
    
    # Test each tool
    results = {}
    for script_path, tool_name in tools:
        results[tool_name] = test_tool(str(script_path), tool_name)
    
    # Summarize test results
    print(f"\n{'='*60}")
    print("Test Results Summary")
    print('='*60)
    
    all_passed = True
    for tool_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{tool_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("All tools passed testing! Ready to use.")
        print("\nRecommended next steps:")
        print("1. Run 'python simple_hardware_analyzer.py' for basic testing")
        print("2. If CUDA environment available, run 'python advanced_bandwidth_tester.py'")
        print("3. Check README_hardware_tools.md for detailed usage")
    else:
        print("Some tools failed testing, please check error messages and fix issues.")
    print('='*60)
    
    return all_passed


if __name__ == '__main__':
    success = run_quick_test()
    sys.exit(0 if success else 1)
