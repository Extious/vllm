#!/usr/bin/env python3
"""
硬件分析工具测试脚本

快速测试所有硬件分析工具的基本功能。

作者: GitHub Copilot
日期: 2025-08-18
"""

import subprocess
import sys
import os
from pathlib import Path


def test_tool(script_path: str, tool_name: str) -> bool:
    """测试单个工具"""
    print(f"\n{'='*60}")
    print(f"测试工具: {tool_name}")
    print(f"脚本路径: {script_path}")
    print('='*60)
    
    if not os.path.exists(script_path):
        print(f"ERROR: 脚本文件不存在: {script_path}")
        return False
    
    # 检查脚本是否可执行
    if not os.access(script_path, os.X_OK):
        print(f"WARNING: 脚本不可执行，尝试添加执行权限...")
        try:
            os.chmod(script_path, 0o755)
            print("OK 执行权限添加成功")
        except Exception as e:
            print(f"ERROR 添加执行权限失败: {e}")
            return False
    
    # 测试脚本的--help选项
    try:
        print("测试--help选项...")
        result = subprocess.run([sys.executable, script_path, '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("OK --help选项工作正常")
            print("帮助信息预览:")
            print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print(f"WARNING --help选项返回非零退出码: {result.returncode}")
            if result.stderr:
                print(f"错误信息: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("WARNING --help选项超时")
    except Exception as e:
        print(f"ERROR 测试--help选项时出错: {e}")
        return False
    
    # 检查脚本的基本语法
    try:
        print("检查Python语法...")
        result = subprocess.run([sys.executable, '-m', 'py_compile', script_path], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("OK Python语法检查通过")
        else:
            print(f"ERROR Python语法错误:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"ERROR 语法检查失败: {e}")
        return False
    
    return True


def check_dependencies():
    """检查系统依赖"""
    print("检查系统依赖...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("OK nvidia-smi 可用")
            # 提取版本信息
            version_line = result.stdout.split('\n')[0]
            print(f"   版本: {version_line}")
        else:
            print("ERROR nvidia-smi 不可用")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("ERROR nvidia-smi 未找到")
    
    # 检查bandwidthTest
    try:
        result = subprocess.run(['bandwidthTest', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("OK bandwidthTest 可用")
        else:
            print("WARNING bandwidthTest 不可用（某些功能将受限）")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("WARNING bandwidthTest 未找到（某些功能将受限）")
    
    # 检查Python依赖包
    required_packages = ['json', 'subprocess', 'time', 'os', 'sys', 'argparse', 'platform']
    optional_packages = ['psutil', 'xml.etree.ElementTree']
    
    print("\n检查Python包依赖:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"OK {package} - 已安装")
        except ImportError:
            print(f"ERROR {package} - 未安装（必需）")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"OK {package} - 已安装")
        except ImportError:
            print(f"WARNING {package} - 未安装（推荐）")


def run_quick_test():
    """运行快速测试"""
    print("硬件分析工具快速测试")
    print("="*60)
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    
    # 定义工具列表
    tools = [
        (script_dir / "hardware_topology_analyzer.py", "基础硬件拓扑分析器"),
        (script_dir / "simple_hardware_analyzer.py", "简化版分析器"),
        (script_dir / "advanced_bandwidth_tester.py", "高级带宽测试器")
    ]
    
    # 检查系统依赖
    check_dependencies()
    
    # 测试每个工具
    results = {}
    for script_path, tool_name in tools:
        results[tool_name] = test_tool(str(script_path), tool_name)
    
    # 总结测试结果
    print(f"\n{'='*60}")
    print("测试结果总结")
    print('='*60)
    
    all_passed = True
    for tool_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{tool_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("所有工具测试通过！可以正常使用。")
        print("\n推荐的下一步操作:")
        print("1. 运行 python simple_hardware_analyzer.py 进行基础测试")
        print("2. 如果有CUDA环境，运行 python advanced_bandwidth_tester.py")
        print("3. 查看 README_hardware_tools.md 了解详细使用方法")
    else:
        print("部分工具测试失败，请检查错误信息并修复问题。")
    print('='*60)
    
    return all_passed


if __name__ == '__main__':
    success = run_quick_test()
    sys.exit(0 if success else 1)
