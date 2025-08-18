#!/usr/bin/env python3
"""
简化版硬件拓扑分析器

当NVIDIA bandwidthTest不可用时，使用替代方法进行基础性能测试。

作者: GitHub Copilot
日期: 2025-08-18
"""

import subprocess
import json
import time
import psutil
import platform
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET


class SimplifiedHardwareAnalyzer:
    def __init__(self):
        self.system_info = {}
        self.gpu_info = []
        self.topology_info = {}
        self.performance_info = {}
    
    def get_system_info(self) -> Dict:
        """获取系统基本信息"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available
        }
        
        # 获取CPU信息
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                # 提取CPU型号
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break
        except:
            pass
        
        self.system_info = info
        return info
    
    def get_gpu_detailed_info(self) -> List[Dict]:
        """获取详细的GPU信息"""
        try:
            # 获取基本GPU信息
            result = subprocess.run(['nvidia-smi', '-q', '-x'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                return []
            
            root = ET.fromstring(result.stdout)
            gpus = []
            
            for gpu in root.findall('gpu'):
                gpu_info = {
                    'id': gpu.get('id'),
                    'name': gpu.find('product_name').text if gpu.find('product_name') is not None else 'Unknown',
                    'uuid': gpu.find('uuid').text if gpu.find('uuid') is not None else 'Unknown',
                    'architecture': gpu.find('product_architecture').text if gpu.find('product_architecture') is not None else 'Unknown',
                    'cuda_version': gpu.find('.//cuda_version').text if gpu.find('.//cuda_version') is not None else 'Unknown',
                    'driver_version': gpu.find('.//driver_version').text if gpu.find('.//driver_version') is not None else 'Unknown',
                    'vbios_version': gpu.find('.//vbios_version').text if gpu.find('.//vbios_version') is not None else 'Unknown'
                }
                
                # 内存信息
                memory = gpu.find('.//memory_usage')
                if memory is not None:
                    gpu_info.update({
                        'memory_total': memory.find('total').text if memory.find('total') is not None else 'Unknown',
                        'memory_used': memory.find('used').text if memory.find('used') is not None else 'Unknown',
                        'memory_free': memory.find('free').text if memory.find('free') is not None else 'Unknown'
                    })
                
                # 温度信息
                temp = gpu.find('.//temperature/gpu_temp')
                if temp is not None:
                    gpu_info['temperature'] = temp.text
                
                # 功耗信息
                power = gpu.find('.//power_readings/power_draw')
                if power is not None:
                    gpu_info['power_draw'] = power.text
                
                # PCIe信息
                pci = gpu.find('.//pci')
                if pci is not None:
                    gpu_info.update({
                        'pci_bus_id': pci.find('pci_bus_id').text if pci.find('pci_bus_id') is not None else 'Unknown',
                        'pci_device_id': pci.find('pci_device_id').text if pci.find('pci_device_id') is not None else 'Unknown',
                        'pci_link_gen_current': pci.find('.//link_gen/current_link_gen').text if pci.find('.//link_gen/current_link_gen') is not None else 'Unknown',
                        'pci_link_width_current': pci.find('.//link_widths/current_link_width').text if pci.find('.//link_widths/current_link_width') is not None else 'Unknown'
                    })
                
                # 性能状态
                perf = gpu.find('.//performance_state')
                if perf is not None:
                    gpu_info['performance_state'] = perf.text
                
                # 时钟频率
                clocks = gpu.find('.//clocks')
                if clocks is not None:
                    gpu_info.update({
                        'graphics_clock': clocks.find('graphics_clock').text if clocks.find('graphics_clock') is not None else 'Unknown',
                        'sm_clock': clocks.find('sm_clock').text if clocks.find('sm_clock') is not None else 'Unknown',
                        'mem_clock': clocks.find('mem_clock').text if clocks.find('mem_clock') is not None else 'Unknown'
                    })
                
                gpus.append(gpu_info)
            
            self.gpu_info = gpus
            return gpus
            
        except Exception as e:
            print(f"获取GPU信息时出错: {e}")
            return []
    
    def get_nvidia_topology(self) -> Dict:
        """获取NVIDIA拓扑信息"""
        topology = {}
        
        try:
            # 拓扑矩阵
            result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                topology['matrix'] = result.stdout
            
            # 拓扑信息
            result = subprocess.run(['nvidia-smi', 'topo', '-i'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                topology['info'] = result.stdout
                
        except Exception as e:
            print(f"获取拓扑信息时出错: {e}")
        
        self.topology_info = topology
        return topology
    
    def test_basic_performance(self) -> Dict:
        """基础性能测试（当bandwidthTest不可用时）"""
        performance = {}
        
        try:
            # 使用nvidia-smi测试GPU利用率变化
            print("正在进行基础性能测试...")
            
            # 记录开始状态
            result_start = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], 
                                        capture_output=True, text=True)
            
            if result_start.returncode == 0:
                start_metrics = []
                for line in result_start.stdout.strip().split('\n'):
                    metrics = line.split(', ')
                    if len(metrics) >= 5:
                        start_metrics.append({
                            'gpu_util': float(metrics[0]) if metrics[0] != '[Not Supported]' else 0,
                            'mem_util': float(metrics[1]) if metrics[1] != '[Not Supported]' else 0,
                            'mem_used': int(metrics[2]) if metrics[2] != '[Not Supported]' else 0,
                            'power': float(metrics[3]) if metrics[3] != '[Not Supported]' else 0,
                            'temp': float(metrics[4]) if metrics[4] != '[Not Supported]' else 0
                        })
                
                performance['start_metrics'] = start_metrics
                
                # 简单的设备信息获取延迟测试
                latency_tests = []
                for i in range(5):
                    start_time = time.time()
                    subprocess.run(['nvidia-smi', '-q'], capture_output=True)
                    end_time = time.time()
                    latency_tests.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                performance['nvidia_smi_latency_ms'] = {
                    'min': min(latency_tests),
                    'max': max(latency_tests),
                    'avg': sum(latency_tests) / len(latency_tests)
                }
        
        except Exception as e:
            print(f"性能测试时出错: {e}")
        
        self.performance_info = performance
        return performance
    
    def analyze_connectivity(self) -> Dict:
        """分析GPU连接性"""
        connectivity = {}
        
        if len(self.gpu_info) > 1:
            # 分析PCIe配置
            pcie_analysis = {}
            for i, gpu in enumerate(self.gpu_info):
                pcie_analysis[f'GPU{i}'] = {
                    'bus_id': gpu.get('pci_bus_id', 'Unknown'),
                    'pcie_gen': gpu.get('pci_link_gen_current', 'Unknown'),
                    'pcie_width': gpu.get('pci_link_width_current', 'Unknown')
                }
            
            connectivity['pcie_analysis'] = pcie_analysis
            
            # 检查NVLink支持（通过拓扑矩阵）
            if self.topology_info and 'matrix' in self.topology_info:
                nvlink_count = self.topology_info['matrix'].count('NV')
                connectivity['nvlink_connections'] = nvlink_count
                
                # 解析拓扑矩阵中的连接类型
                lines = self.topology_info['matrix'].split('\n')
                connections = {}
                for line in lines:
                    if 'GPU' in line and any(conn in line for conn in ['NV', 'PHB', 'PIX', 'PXB']):
                        parts = line.split()
                        if len(parts) > 1:
                            gpu_id = parts[0]
                            connections[gpu_id] = parts[1:]
                
                connectivity['connection_matrix'] = connections
        
        return connectivity
    
    def print_detailed_report(self):
        """打印详细报告"""
        print("=" * 90)
        print("详细硬件拓扑结构分析报告")
        print("=" * 90)
        
        # 系统信息
        print("\n系统信息:")
        print("-" * 60)
        for key, value in self.system_info.items():
            if key == 'memory_total' or key == 'memory_available':
                # 转换字节为GB
                value_gb = value / (1024**3)
                print(f"  {key}: {value_gb:.2f} GB")
            else:
                print(f"  {key}: {value}")
        
        # 详细GPU信息
        print("\n详细GPU信息:")
        print("-" * 60)
        for i, gpu in enumerate(self.gpu_info):
            print(f"\nGPU {i} ({gpu['name']}):")
            for key, value in gpu.items():
                if key != 'name':
                    print(f"  {key}: {value}")
        
        # 拓扑信息
        if self.topology_info:
            print("\n拓扑结构:")
            print("-" * 60)
            if 'matrix' in self.topology_info:
                print("拓扑矩阵:")
                print(self.topology_info['matrix'])
            
            if 'info' in self.topology_info:
                print("\n拓扑详细信息:")
                print(self.topology_info['info'])
        
        # 连接性分析
        connectivity = self.analyze_connectivity()
        if connectivity:
            print("\n连接性分析:")
            print("-" * 60)
            if 'pcie_analysis' in connectivity:
                print("PCIe配置:")
                for gpu_id, config in connectivity['pcie_analysis'].items():
                    print(f"  {gpu_id}: Bus {config['bus_id']}, Gen {config['pcie_gen']}, x{config['pcie_width']}")
            
            if 'nvlink_connections' in connectivity:
                print(f"\nNVLink连接数: {connectivity['nvlink_connections']}")
        
        # 性能信息
        if self.performance_info:
            print("\n性能测试结果:")
            print("-" * 60)
            if 'nvidia_smi_latency_ms' in self.performance_info:
                latency = self.performance_info['nvidia_smi_latency_ms']
                print(f"nvidia-smi查询延迟: {latency['avg']:.2f}ms (最小: {latency['min']:.2f}ms, 最大: {latency['max']:.2f}ms)")
            
            if 'start_metrics' in self.performance_info:
                print("\nGPU当前状态:")
                for i, metrics in enumerate(self.performance_info['start_metrics']):
                    print(f"  GPU {i}: 利用率 {metrics['gpu_util']}%, 内存利用率 {metrics['mem_util']}%, "
                          f"温度 {metrics['temp']}°C, 功耗 {metrics['power']}W")
        
        print("=" * 90)
    
    def save_to_json(self, filename: str):
        """保存结果到JSON文件"""
        data = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'gpu_info': self.gpu_info,
            'topology_info': self.topology_info,
            'performance_info': self.performance_info,
            'connectivity_analysis': self.analyze_connectivity()
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n保存结果已保存到: {filename}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
    
    def run_analysis(self, save_file: Optional[str] = None):
        """运行完整分析"""
        print("启动详细硬件分析...")
        
        # 获取系统信息
        print("获取系统信息...")
        self.get_system_info()
        
        # 获取GPU信息
        print("获取GPU详细信息...")
        if not self.get_gpu_detailed_info():
            print("未检测到GPU设备或无法访问!")
            return False
        
        # 获取拓扑信息
        print("分析拓扑结构...")
        self.get_nvidia_topology()
        
        # 基础性能测试
        print("执行基础性能测试...")
        self.test_basic_performance()
        
        # 打印报告
        self.print_detailed_report()
        
        # 保存结果
        if save_file:
            self.save_to_json(save_file)
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='简化版硬件拓扑分析器')
    parser.add_argument('--save', '-s', type=str, help='保存结果到JSON文件')
    parser.add_argument('--simple', action='store_true', help='只显示简要信息')
    
    args = parser.parse_args()
    
    analyzer = SimplifiedHardwareAnalyzer()
    
    print("Simplified Hardware Topology Analyzer")
    print("For environments where bandwidthTest is not available\n")
    
    success = analyzer.run_analysis(args.save)
    
    if success:
        print("\nAnalysis completed!")
    else:
        print("\nAnalysis failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
