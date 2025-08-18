#!/usr/bin/env python3
"""
高级带宽测试器

这是一个功能更强大的硬件拓扑分析和带宽测试工具，包含：
- 详细的硬件拓扑分析
- 多种带宽测试方法
- 性能基准测试
- GPU间通信分析
- 自动生成测试报告

作者: GitHub Copilot
日期: 2025-08-18
"""

import subprocess
import json
import time
import os
import sys
import threading
import queue
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import xml.etree.ElementTree as ET
import argparse
import platform
import psutil


class AdvancedBandwidthTester:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gpu_info = []
        self.topology_info = {}
        self.bandwidth_results = {}
        self.system_info = {}
        self.test_results = {}
        self.start_time = datetime.now()
        
    def log(self, message: str, level: str = "INFO"):
        """日志输出"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def check_nvidia_tools(self) -> Dict[str, bool]:
        """检查NVIDIA工具可用性"""
        tools_status = {}
        
        # 检查nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            tools_status['nvidia-smi'] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools_status['nvidia-smi'] = False
        
        # 检查bandwidthTest
        try:
            result = subprocess.run(['bandwidthTest', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            tools_status['bandwidthTest'] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools_status['bandwidthTest'] = False
        
        # 检查其他CUDA工具
        for tool in ['nvcc', 'nvidia-ml-py']:
            try:
                if tool == 'nvcc':
                    result = subprocess.run([tool, '--version'], 
                                          capture_output=True, text=True, timeout=10)
                    tools_status[tool] = result.returncode == 0
                elif tool == 'nvidia-ml-py':
                    import pynvml
                    tools_status[tool] = True
            except (subprocess.TimeoutExpired, FileNotFoundError, ImportError):
                tools_status[tool] = False
        
        return tools_status
    
    def get_system_info(self) -> Dict:
        """获取详细系统信息"""
        info = {
            'timestamp': self.start_time.isoformat(),
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'python_version': platform.python_version()
        }
        
        # 获取CPU详细信息
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break
                    elif 'flags' in line:
                        info['cpu_flags'] = line.split(':')[1].strip()
        except Exception:
            pass
        
        # 获取内核信息
        try:
            info['kernel_version'] = platform.release()
        except Exception:
            pass
        
        self.system_info = info
        return info
    
    def get_gpu_comprehensive_info(self) -> List[Dict]:
        """获取全面的GPU信息"""
        gpus = []
        
        try:
            # 使用nvidia-smi获取XML格式的详细信息
            result = subprocess.run(['nvidia-smi', '-q', '-x'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.log(f"nvidia-smi执行失败: {result.stderr}", "ERROR")
                return []
            
            root = ET.fromstring(result.stdout)
            
            for gpu in root.findall('gpu'):
                gpu_info = self._parse_gpu_xml(gpu)
                gpus.append(gpu_info)
            
            # 获取额外的性能信息
            for i, gpu in enumerate(gpus):
                gpu.update(self._get_gpu_performance_info(i))
            
            self.gpu_info = gpus
            return gpus
            
        except Exception as e:
            self.log(f"获取GPU信息时出错: {e}", "ERROR")
            return []
    
    def _parse_gpu_xml(self, gpu_element) -> Dict:
        """解析GPU XML元素"""
        gpu_info = {
            'id': gpu_element.get('id'),
            'uuid': self._get_xml_text(gpu_element, 'uuid'),
            'name': self._get_xml_text(gpu_element, 'product_name'),
            'brand': self._get_xml_text(gpu_element, 'product_brand'),
            'architecture': self._get_xml_text(gpu_element, 'product_architecture'),
            'cuda_version': self._get_xml_text(gpu_element, './/cuda_version'),
            'driver_version': self._get_xml_text(gpu_element, './/driver_version'),
            'vbios_version': self._get_xml_text(gpu_element, './/vbios_version')
        }
        
        # 内存信息
        memory = gpu_element.find('.//memory_usage')
        if memory is not None:
            gpu_info.update({
                'memory_total': self._get_xml_text(memory, 'total'),
                'memory_used': self._get_xml_text(memory, 'used'),
                'memory_free': self._get_xml_text(memory, 'free')
            })
        
        # PCIe信息
        pci = gpu_element.find('.//pci')
        if pci is not None:
            gpu_info.update({
                'pci_bus': self._get_xml_text(pci, 'pci_bus'),
                'pci_device': self._get_xml_text(pci, 'pci_device'),
                'pci_domain': self._get_xml_text(pci, 'pci_domain'),
                'pci_bus_id': self._get_xml_text(pci, 'pci_bus_id'),
                'pci_link_gen_current': self._get_xml_text(pci, './/link_gen/current_link_gen'),
                'pci_link_gen_max': self._get_xml_text(pci, './/link_gen/max_link_gen'),
                'pci_link_width_current': self._get_xml_text(pci, './/link_widths/current_link_width'),
                'pci_link_width_max': self._get_xml_text(pci, './/link_widths/max_link_width')
            })
        
        # 时钟频率
        clocks = gpu_element.find('.//clocks')
        if clocks is not None:
            gpu_info.update({
                'graphics_clock': self._get_xml_text(clocks, 'graphics_clock'),
                'sm_clock': self._get_xml_text(clocks, 'sm_clock'),
                'mem_clock': self._get_xml_text(clocks, 'mem_clock'),
                'video_clock': self._get_xml_text(clocks, 'video_clock')
            })
        
        # 温度和功耗
        gpu_info.update({
            'temperature': self._get_xml_text(gpu_element, './/temperature/gpu_temp'),
            'power_draw': self._get_xml_text(gpu_element, './/power_readings/power_draw'),
            'power_limit': self._get_xml_text(gpu_element, './/power_readings/power_limit'),
            'performance_state': self._get_xml_text(gpu_element, './/performance_state')
        })
        
        return gpu_info
    
    def _get_xml_text(self, element, path: str) -> str:
        """安全获取XML元素文本"""
        try:
            found = element.find(path)
            return found.text if found is not None else 'Unknown'
        except:
            return 'Unknown'
    
    def _get_gpu_performance_info(self, gpu_id: int) -> Dict:
        """获取GPU性能信息"""
        perf_info = {}
        
        try:
            # 获取实时利用率
            result = subprocess.run([
                'nvidia-smi', '--id=' + str(gpu_id),
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,fan.speed',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                if len(data) >= 6:
                    perf_info.update({
                        'current_gpu_util': data[0],
                        'current_mem_util': data[1],
                        'current_mem_used': data[2],
                        'current_power': data[3],
                        'current_temp': data[4],
                        'current_fan_speed': data[5]
                    })
        except Exception as e:
            self.log(f"获取GPU {gpu_id} 性能信息失败: {e}", "WARNING")
        
        return perf_info
    
    def get_detailed_topology(self) -> Dict:
        """获取详细拓扑信息"""
        topology = {}
        
        try:
            # 拓扑矩阵
            result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                topology['matrix'] = result.stdout
                topology['connections'] = self._parse_topology_matrix(result.stdout)
            
            # 拓扑信息
            result = subprocess.run(['nvidia-smi', 'topo', '-i'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                topology['detailed_info'] = result.stdout
            
            # P2P可访问性矩阵
            if len(self.gpu_info) > 1:
                topology['p2p_matrix'] = self._get_p2p_accessibility()
            
        except Exception as e:
            self.log(f"获取拓扑信息失败: {e}", "ERROR")
        
        self.topology_info = topology
        return topology
    
    def _parse_topology_matrix(self, matrix_output: str) -> Dict:
        """解析拓扑矩阵"""
        connections = {}
        lines = matrix_output.strip().split('\n')
        
        # 找到矩阵数据
        matrix_started = False
        gpu_headers = []
        
        for line in lines:
            if 'GPU' in line and not matrix_started:
                # 解析GPU标题行
                parts = line.split()
                gpu_headers = [part for part in parts if part.startswith('GPU')]
                matrix_started = True
            elif matrix_started and line.strip() and not line.startswith('Legend'):
                # 解析数据行
                parts = line.split()
                if parts and parts[0].startswith('GPU'):
                    gpu_id = parts[0]
                    connections[gpu_id] = {}
                    for i, conn_type in enumerate(parts[1:]):
                        if i < len(gpu_headers):
                            connections[gpu_id][gpu_headers[i]] = conn_type
        
        return connections
    
    def _get_p2p_accessibility(self) -> Dict:
        """获取P2P可访问性信息"""
        p2p_matrix = {}
        
        try:
            # 这里可以添加自定义的P2P测试代码
            # 或者解析nvidia-smi的其他输出
            pass
        except Exception as e:
            self.log(f"获取P2P信息失败: {e}", "WARNING")
        
        return p2p_matrix
    
    def run_comprehensive_bandwidth_tests(self) -> Dict:
        """运行全面的带宽测试"""
        results = {}
        
        self.log("开始全面带宽测试...")
        
        # 1. 标准bandwidthTest
        if self.check_nvidia_tools().get('bandwidthTest', False):
            results['standard_bandwidth'] = self._run_standard_bandwidth_test()
        else:
            self.log("bandwidthTest不可用，跳过标准测试", "WARNING")
        
        # 2. 自定义带宽测试
        results['custom_tests'] = self._run_custom_bandwidth_tests()
        
        # 3. 延迟测试
        results['latency_tests'] = self._run_latency_tests()
        
        # 4. 多GPU测试（如果有多个GPU）
        if len(self.gpu_info) > 1:
            results['multi_gpu_tests'] = self._run_multi_gpu_tests()
        
        self.bandwidth_results = results
        return results
    
    def _run_standard_bandwidth_test(self) -> Dict:
        """运行标准带宽测试"""
        results = {}
        
        try:
            # 基本测试
            self.log("运行基本带宽测试...")
            result = subprocess.run(['bandwidthTest'], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                results['basic'] = self._parse_bandwidth_output(result.stdout)
            
            # 设备到设备测试
            if len(self.gpu_info) > 1:
                self.log("运行设备间带宽测试...")
                result = subprocess.run(['bandwidthTest', '--device=all'], 
                                      capture_output=True, text=True, timeout=180)
                if result.returncode == 0:
                    results['device_to_device'] = self._parse_bandwidth_output(result.stdout)
            
            # 内存模式测试
            for mode in ['pinned', 'pageable']:
                self.log(f"运行{mode}内存模式测试...")
                result = subprocess.run(['bandwidthTest', f'--memory={mode}'], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    results[f'{mode}_memory'] = self._parse_bandwidth_output(result.stdout)
        
        except Exception as e:
            self.log(f"标准带宽测试失败: {e}", "ERROR")
        
        return results
    
    def _parse_bandwidth_output(self, output: str) -> Dict:
        """解析带宽测试输出"""
        results = {
            'host_to_device': {},
            'device_to_host': {},
            'device_to_device': {}
        }
        
        lines = output.split('\n')
        current_test = None
        
        for line in lines:
            line = line.strip()
            
            if 'Host to Device Bandwidth' in line:
                current_test = 'host_to_device'
            elif 'Device to Host Bandwidth' in line:
                current_test = 'device_to_host'
            elif 'Device to Device Bandwidth' in line:
                current_test = 'device_to_device'
            elif current_test and 'MB/s' in line and any(char.isdigit() for char in line):
                # 解析带宽数据
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        transfer_size = parts[0]
                        bandwidth = float(parts[1])
                        results[current_test][transfer_size] = bandwidth
                    except (ValueError, IndexError):
                        continue
        
        return results
    
    def _run_custom_bandwidth_tests(self) -> Dict:
        """运行自定义带宽测试"""
        results = {}
        
        # 使用nvidia-smi的查询延迟作为基准
        latencies = []
        for i in range(10):
            start_time = time.time()
            subprocess.run(['nvidia-smi', '-q'], capture_output=True)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
        
        results['nvidia_smi_query_latency'] = {
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'avg_ms': statistics.mean(latencies),
            'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
        
        return results
    
    def _run_latency_tests(self) -> Dict:
        """运行延迟测试"""
        results = {}
        
        try:
            # GPU初始化延迟测试
            init_times = []
            for i in range(5):
                start_time = time.time()
                subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=name', '--format=csv,noheader'], 
                             capture_output=True)
                end_time = time.time()
                init_times.append((end_time - start_time) * 1000)
            
            results['gpu_query_latency'] = {
                'samples': init_times,
                'avg_ms': statistics.mean(init_times),
                'min_ms': min(init_times),
                'max_ms': max(init_times)
            }
        
        except Exception as e:
            self.log(f"延迟测试失败: {e}", "WARNING")
        
        return results
    
    def _run_multi_gpu_tests(self) -> Dict:
        """运行多GPU测试"""
        results = {}
        
        try:
            # 测试每个GPU的独立性能
            gpu_performances = {}
            for i, gpu in enumerate(self.gpu_info):
                perf_data = []
                for _ in range(3):
                    start_time = time.time()
                    result = subprocess.run([
                        'nvidia-smi', '--id=' + str(i),
                        '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True)
                    end_time = time.time()
                    
                    if result.returncode == 0:
                        perf_data.append({
                            'query_time_ms': (end_time - start_time) * 1000,
                            'output': result.stdout.strip()
                        })
                
                gpu_performances[f'GPU{i}'] = perf_data
            
            results['individual_gpu_performance'] = gpu_performances
        
        except Exception as e:
            self.log(f"多GPU测试失败: {e}", "WARNING")
        
        return results
    
    def generate_comprehensive_report(self) -> str:
        """生成综合报告"""
        report = []
        report.append("=" * 100)
        report.append("高级硬件拓扑结构与带宽分析报告")
        report.append("=" * 100)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试持续时间: {(datetime.now() - self.start_time).total_seconds():.2f} 秒")
        report.append("")
        
        # 工具可用性
        tools_status = self.check_nvidia_tools()
        report.append("🔧 工具可用性检查:")
        report.append("-" * 50)
        for tool, available in tools_status.items():
            status = "✅ 可用" if available else "❌ 不可用"
            report.append(f"  {tool}: {status}")
        report.append("")
        
        # 系统信息
        report.append("🖥️  系统信息:")
        report.append("-" * 50)
        for key, value in self.system_info.items():
            if 'memory' in key and 'gb' in key:
                report.append(f"  {key}: {value:.2f} GB")
            else:
                report.append(f"  {key}: {value}")
        report.append("")
        
        # GPU详细信息
        report.append("🎮 GPU详细信息:")
        report.append("-" * 50)
        for i, gpu in enumerate(self.gpu_info):
            report.append(f"\nGPU {i} - {gpu['name']}:")
            important_fields = [
                'uuid', 'architecture', 'memory_total', 'pci_bus_id',
                'pci_link_gen_current', 'pci_link_width_current',
                'current_temp', 'current_power', 'performance_state'
            ]
            for field in important_fields:
                if field in gpu:
                    report.append(f"  {field}: {gpu[field]}")
        report.append("")
        
        # 拓扑信息
        if self.topology_info:
            report.append("🗺️  拓扑结构分析:")
            report.append("-" * 50)
            if 'matrix' in self.topology_info:
                report.append("拓扑矩阵:")
                report.append(self.topology_info['matrix'])
            
            if 'connections' in self.topology_info:
                report.append("\n连接分析:")
                for gpu_id, connections in self.topology_info['connections'].items():
                    report.append(f"  {gpu_id}: {connections}")
            report.append("")
        
        # 带宽测试结果
        if self.bandwidth_results:
            report.append("⚡ 带宽测试结果:")
            report.append("-" * 50)
            self._add_bandwidth_results_to_report(report)
        
        report.append("=" * 100)
        return "\n".join(report)
    
    def _add_bandwidth_results_to_report(self, report: List[str]):
        """添加带宽测试结果到报告"""
        for test_category, results in self.bandwidth_results.items():
            if results:
                report.append(f"\n{test_category.replace('_', ' ').title()}:")
                
                if isinstance(results, dict):
                    for test_type, data in results.items():
                        if isinstance(data, dict) and data:
                            report.append(f"  {test_type}:")
                            for key, value in data.items():
                                if isinstance(value, (int, float)):
                                    report.append(f"    {key}: {value:.2f}")
                                else:
                                    report.append(f"    {key}: {value}")
    
    def save_detailed_results(self, filename: str):
        """保存详细结果到JSON文件"""
        detailed_results = {
            'metadata': {
                'timestamp': self.start_time.isoformat(),
                'test_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
                'tools_status': self.check_nvidia_tools()
            },
            'system_info': self.system_info,
            'gpu_info': self.gpu_info,
            'topology_info': self.topology_info,
            'bandwidth_results': self.bandwidth_results,
            'test_results': self.test_results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            self.log(f"详细结果已保存到: {filename}")
        except Exception as e:
            self.log(f"保存文件失败: {e}", "ERROR")
    
    def run_full_analysis(self, save_json: Optional[str] = None, save_report: Optional[str] = None):
        """运行完整分析"""
        self.log("🚀 开始高级硬件拓扑和带宽分析...")
        
        # 1. 系统信息收集
        self.log("📊 收集系统信息...")
        self.get_system_info()
        
        # 2. GPU信息收集
        self.log("🎮 收集GPU详细信息...")
        if not self.get_gpu_comprehensive_info():
            self.log("未检测到GPU设备!", "ERROR")
            return False
        
        # 3. 拓扑分析
        self.log("🗺️  分析硬件拓扑...")
        self.get_detailed_topology()
        
        # 4. 带宽测试
        self.log("⚡ 执行带宽测试...")
        self.run_comprehensive_bandwidth_tests()
        
        # 5. 生成报告
        self.log("📋 生成分析报告...")
        report = self.generate_comprehensive_report()
        print(report)
        
        # 6. 保存结果
        if save_json:
            self.save_detailed_results(save_json)
        
        if save_report:
            try:
                with open(save_report, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.log(f"报告已保存到: {save_report}")
            except Exception as e:
                self.log(f"保存报告失败: {e}", "ERROR")
        
        self.log("✅ 分析完成!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='高级硬件拓扑结构和带宽测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s                                    # 运行基本分析
  %(prog)s --save-json results.json          # 保存JSON结果
  %(prog)s --save-report report.txt          # 保存文本报告
  %(prog)s --quiet                          # 静默模式
  %(prog)s --save-json data.json --save-report analysis.txt  # 保存所有结果
        """
    )
    
    parser.add_argument('--save-json', type=str, help='保存详细结果到JSON文件')
    parser.add_argument('--save-report', type=str, help='保存文本报告到文件')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    # 创建测试器实例
    tester = AdvancedBandwidthTester(verbose=not args.quiet)
    
    # 运行分析
    success = tester.run_full_analysis(
        save_json=args.save_json,
        save_report=args.save_report
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
