#!/usr/bin/env python3
"""
Simplified Hardware Analyzer - English Version

For environments where NVIDIA bandwidthTest is not available.

Author: GitHub Copilot
Date: 2025-08-18
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
        """Get basic system information"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available
        }
        
        # Get CPU information
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                # Extract CPU model
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break
        except:
            pass
        
        self.system_info = info
        return info
    
    def get_gpu_detailed_info(self) -> List[Dict]:
        """Get detailed GPU information"""
        try:
            # Get basic GPU information
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
                
                # Memory information
                memory = gpu.find('.//memory_usage')
                if memory is not None:
                    gpu_info.update({
                        'memory_total': memory.find('total').text if memory.find('total') is not None else 'Unknown',
                        'memory_used': memory.find('used').text if memory.find('used') is not None else 'Unknown',
                        'memory_free': memory.find('free').text if memory.find('free') is not None else 'Unknown'
                    })
                
                # Temperature information
                temp = gpu.find('.//temperature/gpu_temp')
                if temp is not None:
                    gpu_info['temperature'] = temp.text
                
                # Power information
                power = gpu.find('.//power_readings/power_draw')
                if power is not None:
                    gpu_info['power_draw'] = power.text
                
                # PCIe information
                pci = gpu.find('.//pci')
                if pci is not None:
                    gpu_info.update({
                        'pci_bus_id': pci.find('pci_bus_id').text if pci.find('pci_bus_id') is not None else 'Unknown',
                        'pci_device_id': pci.find('pci_device_id').text if pci.find('pci_device_id') is not None else 'Unknown',
                        'pci_link_gen_current': pci.find('.//link_gen/current_link_gen').text if pci.find('.//link_gen/current_link_gen') is not None else 'Unknown',
                        'pci_link_width_current': pci.find('.//link_widths/current_link_width').text if pci.find('.//link_widths/current_link_width') is not None else 'Unknown'
                    })
                
                # Performance state
                perf = gpu.find('.//performance_state')
                if perf is not None:
                    gpu_info['performance_state'] = perf.text
                
                # Clock frequencies
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
            print(f"Error getting GPU information: {e}")
            return []
    
    def get_nvidia_topology(self) -> Dict:
        """Get NVIDIA topology information"""
        topology = {}
        
        try:
            # Topology matrix
            result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                topology['matrix'] = result.stdout
            
            # Topology information
            result = subprocess.run(['nvidia-smi', 'topo', '-i'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                topology['info'] = result.stdout
                
        except Exception as e:
            print(f"Error getting topology information: {e}")
        
        self.topology_info = topology
        return topology
    
    def test_basic_performance(self) -> Dict:
        """Basic performance testing (when bandwidthTest is not available)"""
        performance = {}
        
        try:
            # Use nvidia-smi to test GPU utilization changes
            print("Running basic performance tests...")
            
            # Record start state
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
                
                # Simple device information query latency test
                latency_tests = []
                for i in range(5):
                    start_time = time.time()
                    subprocess.run(['nvidia-smi', '-q'], capture_output=True)
                    end_time = time.time()
                    latency_tests.append((end_time - start_time) * 1000)  # Convert to milliseconds
                
                performance['nvidia_smi_latency_ms'] = {
                    'min': min(latency_tests),
                    'max': max(latency_tests),
                    'avg': sum(latency_tests) / len(latency_tests)
                }
        
        except Exception as e:
            print(f"Error during performance testing: {e}")
        
        self.performance_info = performance
        return performance
    
    def analyze_connectivity(self) -> Dict:
        """Analyze GPU connectivity"""
        connectivity = {}
        
        if len(self.gpu_info) > 1:
            # Analyze PCIe configuration
            pcie_analysis = {}
            for i, gpu in enumerate(self.gpu_info):
                pcie_analysis[f'GPU{i}'] = {
                    'bus_id': gpu.get('pci_bus_id', 'Unknown'),
                    'pcie_gen': gpu.get('pci_link_gen_current', 'Unknown'),
                    'pcie_width': gpu.get('pci_link_width_current', 'Unknown')
                }
            
            connectivity['pcie_analysis'] = pcie_analysis
            
            # Check NVLink support (through topology matrix)
            if self.topology_info and 'matrix' in self.topology_info:
                nvlink_count = self.topology_info['matrix'].count('NV')
                connectivity['nvlink_connections'] = nvlink_count
                
                # Parse connection types in topology matrix
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
        """Print detailed report"""
        print("=" * 90)
        print("Detailed Hardware Topology Analysis Report")
        print("=" * 90)
        
        # System information
        print("\nSystem Information:")
        print("-" * 60)
        for key, value in self.system_info.items():
            if key == 'memory_total' or key == 'memory_available':
                # Convert bytes to GB
                value_gb = value / (1024**3)
                print(f"  {key}: {value_gb:.2f} GB")
            else:
                print(f"  {key}: {value}")
        
        # Detailed GPU information
        print("\nDetailed GPU Information:")
        print("-" * 60)
        for i, gpu in enumerate(self.gpu_info):
            print(f"\nGPU {i} ({gpu['name']}):")
            for key, value in gpu.items():
                if key != 'name':
                    print(f"  {key}: {value}")
        
        # Topology information
        if self.topology_info:
            print("\nTopology Structure:")
            print("-" * 60)
            if 'matrix' in self.topology_info:
                print("Topology Matrix:")
                print(self.topology_info['matrix'])
            
            if 'info' in self.topology_info:
                print("\nDetailed Topology Information:")
                print(self.topology_info['info'])
        
        # Connectivity analysis
        connectivity = self.analyze_connectivity()
        if connectivity:
            print("\nConnectivity Analysis:")
            print("-" * 60)
            if 'pcie_analysis' in connectivity:
                print("PCIe Configuration:")
                for gpu_id, config in connectivity['pcie_analysis'].items():
                    print(f"  {gpu_id}: Bus {config['bus_id']}, Gen {config['pcie_gen']}, x{config['pcie_width']}")
            
            if 'nvlink_connections' in connectivity:
                print(f"\nNVLink Connections: {connectivity['nvlink_connections']}")
        
        # Performance information
        if self.performance_info:
            print("\nPerformance Test Results:")
            print("-" * 60)
            if 'nvidia_smi_latency_ms' in self.performance_info:
                latency = self.performance_info['nvidia_smi_latency_ms']
                print(f"nvidia-smi query latency: {latency['avg']:.2f}ms (min: {latency['min']:.2f}ms, max: {latency['max']:.2f}ms)")
            
            if 'start_metrics' in self.performance_info:
                print("\nCurrent GPU Status:")
                for i, metrics in enumerate(self.performance_info['start_metrics']):
                    print(f"  GPU {i}: Utilization {metrics['gpu_util']}%, Memory Utilization {metrics['mem_util']}%, "
                          f"Temperature {metrics['temp']}°C, Power {metrics['power']}W")
        
        print("=" * 90)
    
    def save_to_json(self, filename: str):
        """Save results to JSON file"""
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
            print(f"\nResults saved to: {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def run_analysis(self, save_file: Optional[str] = None):
        """Run complete analysis"""
        print("Starting detailed hardware analysis...")
        
        # Get system information
        print("Getting system information...")
        self.get_system_info()
        
        # Get GPU information
        print("Getting detailed GPU information...")
        if not self.get_gpu_detailed_info():
            print("No GPU devices detected or unable to access!")
            return False
        
        # Get topology information
        print("Analyzing topology structure...")
        self.get_nvidia_topology()
        
        # Basic performance testing
        print("Running basic performance tests...")
        self.test_basic_performance()
        
        # Print report
        self.print_detailed_report()
        
        # Save results
        if save_file:
            self.save_to_json(save_file)
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Hardware Topology Analyzer')
    parser.add_argument('--save', '-s', type=str, help='Save results to JSON file')
    parser.add_argument('--simple', action='store_true', help='Show only key information')
    
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
