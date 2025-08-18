#!/usr/bin/env python3
"""
硬件拓扑结构分析器和带宽测试工具

该脚本使用 nvidia-smi 分析硬件拓扑结构，并使用 NVIDIA 的 bandwidthTest 工具
测试 CPU 和 GPU 之间以及 GPU 之间的传播速度。

作者: GitHub Copilot
日期: 2025-08-18
"""

import subprocess
import json
import re
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


class HardwareTopologyAnalyzer:
    def __init__(self):
        self.gpu_info = []
        self.topology_info = {}
        self.bandwidth_results = {}
        
    def check_dependencies(self) -> bool:
        """检查必要的依赖工具是否存在"""
        tools = {
            'nvidia-smi': 'NVIDIA系统管理接口',
            'bandwidthTest': 'NVIDIA带宽测试工具'
        }
        
        missing_tools = []
        for tool, description in tools.items():
            try:
                result = subprocess.run([tool, '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    missing_tools.append(f"{tool} ({description})")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing_tools.append(f"{tool} ({description})")
        
        if missing_tools:
            print("错误: 以下工具未找到或无法使用:")
            for tool in missing_tools:
                print(f"  - {tool}")
            print("\n请确保已安装NVIDIA驱动程序和CUDA工具包。")
            return False
        
        return True
    
    def get_gpu_info(self) -> List[Dict]:
        """获取GPU基本信息"""
        try:
            # 使用nvidia-smi获取GPU信息的XML格式
            result = subprocess.run(['nvidia-smi', '-q', '-x'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi执行失败: {result.stderr}")
            
            root = ET.fromstring(result.stdout)
            gpus = []
            
            for gpu in root.findall('gpu'):
                gpu_info = {
                    'id': gpu.get('id'),
                    'name': gpu.find('product_name').text if gpu.find('product_name') is not None else 'Unknown',
                    'uuid': gpu.find('uuid').text if gpu.find('uuid') is not None else 'Unknown',
                    'memory_total': gpu.find('.//memory_usage/total').text if gpu.find('.//memory_usage/total') is not None else 'Unknown',
                    'memory_used': gpu.find('.//memory_usage/used').text if gpu.find('.//memory_usage/used') is not None else 'Unknown',
                    'memory_free': gpu.find('.//memory_usage/free').text if gpu.find('.//memory_usage/free') is not None else 'Unknown',
                    'driver_version': gpu.find('.//driver_version').text if gpu.find('.//driver_version') is not None else 'Unknown',
                    'cuda_version': gpu.find('.//cuda_version').text if gpu.find('.//cuda_version') is not None else 'Unknown'
                }
                gpus.append(gpu_info)
            
            self.gpu_info = gpus
            return gpus
            
        except Exception as e:
            print(f"获取GPU信息时出错: {e}")
            return []
    
    def get_topology_info(self) -> Dict:
        """获取GPU拓扑结构信息"""
        try:
            # 获取GPU拓扑结构
            result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"获取拓扑信息失败: {result.stderr}")
                return {}
            
            topology_output = result.stdout
            self.topology_info = {'raw_output': topology_output}
            
            # 解析拓扑矩阵
            lines = topology_output.strip().split('\n')
            matrix_data = []
            gpu_ids = []
            
            for line in lines:
                if 'GPU' in line and 'mlx' not in line.lower():
                    # 这是标题行，提取GPU ID
                    parts = line.split()
                    for part in parts:
                        if part.startswith('GPU') and part != 'GPU':
                            gpu_ids.append(part)
                elif line.strip() and not line.startswith('Legend:') and 'X' in line:
                    # 这是数据行
                    matrix_data.append(line.strip())
            
            self.topology_info['gpu_ids'] = gpu_ids
            self.topology_info['matrix'] = matrix_data
            
            return self.topology_info
            
        except Exception as e:
            print(f"获取拓扑信息时出错: {e}")
            return {}
    
    def parse_bandwidth_test_output(self, output: str) -> Dict:
        """解析bandwidthTest的输出"""
        results = {
            'host_to_device': {},
            'device_to_host': {},
            'device_to_device': {}
        }
        
        lines = output.split('\n')
        current_test = None
        
        for line in lines:
            line = line.strip()
            
            # 识别测试类型
            if 'Host to Device Bandwidth' in line:
                current_test = 'host_to_device'
            elif 'Device to Host Bandwidth' in line:
                current_test = 'device_to_host'
            elif 'Device to Device Bandwidth' in line:
                current_test = 'device_to_device'
            
            # 提取带宽数据
            if current_test and 'MB/s' in line and 'Transfer Size' not in line:
                # 使用正则表达式提取传输大小和带宽
                match = re.search(r'(\d+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', line)
                if match:
                    transfer_size = match.group(1)
                    bandwidth = float(match.group(2))
                    
                    if current_test not in results:
                        results[current_test] = {}
                    results[current_test][transfer_size] = bandwidth
        
        return results
    
    def run_bandwidth_tests(self) -> Dict:
        """运行带宽测试"""
        print("正在运行带宽测试...")
        
        # 基本的Host-Device测试
        try:
            result = subprocess.run(['bandwidthTest'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                basic_results = self.parse_bandwidth_test_output(result.stdout)
                self.bandwidth_results.update(basic_results)
        except Exception as e:
            print(f"基本带宽测试失败: {e}")
        
        # GPU间P2P测试（如果有多个GPU）
        if len(self.gpu_info) > 1:
            try:
                result = subprocess.run(['bandwidthTest', '--device=all'], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    p2p_results = self.parse_bandwidth_test_output(result.stdout)
                    self.bandwidth_results.update(p2p_results)
            except Exception as e:
                print(f"P2P带宽测试失败: {e}")
        
        return self.bandwidth_results
    
    def analyze_pcie_info(self) -> Dict:
        """分析PCIe信息"""
        pcie_info = {}
        
        try:
            # 获取每个GPU的PCIe信息
            for i, gpu in enumerate(self.gpu_info):
                result = subprocess.run(['nvidia-smi', '--id=' + str(i), '--query-gpu=pci.bus_id,pci.link.gen.current,pci.link.width.current', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    data = result.stdout.strip().split(', ')
                    if len(data) >= 3:
                        pcie_info[f'GPU{i}'] = {
                            'bus_id': data[0],
                            'pcie_gen': data[1],
                            'pcie_width': data[2]
                        }
        except Exception as e:
            print(f"获取PCIe信息时出错: {e}")
        
        return pcie_info
    
    def print_analysis_report(self):
        """打印分析报告"""
        print("=" * 80)
        print("硬件拓扑结构分析报告")
        print("=" * 80)
        
        # GPU基本信息
        print("\n📊 GPU基本信息:")
        print("-" * 50)
        for i, gpu in enumerate(self.gpu_info):
            print(f"GPU {i}:")
            print(f"  名称: {gpu['name']}")
            print(f"  UUID: {gpu['uuid']}")
            print(f"  显存: {gpu['memory_used']} / {gpu['memory_total']}")
            print(f"  驱动版本: {gpu['driver_version']}")
            print(f"  CUDA版本: {gpu['cuda_version']}")
            print()
        
        # PCIe信息
        pcie_info = self.analyze_pcie_info()
        if pcie_info:
            print("🔌 PCIe连接信息:")
            print("-" * 50)
            for gpu_id, info in pcie_info.items():
                print(f"{gpu_id}:")
                print(f"  Bus ID: {info['bus_id']}")
                print(f"  PCIe Generation: {info['pcie_gen']}")
                print(f"  PCIe Width: x{info['pcie_width']}")
                print()
        
        # 拓扑结构
        if self.topology_info:
            print("🗺️  GPU拓扑结构:")
            print("-" * 50)
            if 'raw_output' in self.topology_info:
                print(self.topology_info['raw_output'])
        
        # 带宽测试结果
        if self.bandwidth_results:
            print("⚡ 带宽测试结果:")
            print("-" * 50)
            
            for test_type, results in self.bandwidth_results.items():
                if results:
                    print(f"\n{test_type.replace('_', ' ').title()}:")
                    for size, bandwidth in results.items():
                        print(f"  传输大小 {size} bytes: {bandwidth:.2f} MB/s")
        
        print("=" * 80)
    
    def save_results_to_file(self, filename: str):
        """保存结果到文件"""
        results = {
            'gpu_info': self.gpu_info,
            'topology_info': self.topology_info,
            'pcie_info': self.analyze_pcie_info(),
            'bandwidth_results': self.bandwidth_results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到文件: {filename}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
    
    def run_full_analysis(self, save_file: Optional[str] = None):
        """运行完整的硬件分析"""
        print("开始硬件拓扑结构和带宽分析...")
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 获取GPU信息
        print("正在获取GPU信息...")
        self.get_gpu_info()
        
        if not self.gpu_info:
            print("未检测到GPU设备!")
            return False
        
        # 获取拓扑信息
        print("正在分析拓扑结构...")
        self.get_topology_info()
        
        # 运行带宽测试
        self.run_bandwidth_tests()
        
        # 打印报告
        self.print_analysis_report()
        
        # 保存结果
        if save_file:
            self.save_results_to_file(save_file)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='硬件拓扑结构分析器和带宽测试工具')
    parser.add_argument('--save', '-s', type=str, help='保存结果到指定文件 (JSON格式)')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式，只显示关键信息')
    
    args = parser.parse_args()
    
    analyzer = HardwareTopologyAnalyzer()
    
    if not args.quiet:
        print("🚀 硬件拓扑结构分析器启动中...")
        print("此工具将分析GPU拓扑结构并测试带宽性能\n")
    
    success = analyzer.run_full_analysis(args.save)
    
    if not success:
        sys.exit(1)
    
    if not args.quiet:
        print("\n✅ 分析完成!")


if __name__ == '__main__':
    main()
