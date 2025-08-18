# 硬件拓扑分析和带宽测试工具集

这个工具集包含三个Python脚本，用于分析GPU硬件拓扑结构并测试CPU和GPU之间以及GPU之间的传播速度。

## 工具说明

### 1. hardware_topology_analyzer.py - 基础硬件拓扑分析器

**功能特点：**
- 使用nvidia-smi分析GPU基本信息
- 使用NVIDIA bandwidthTest工具测试带宽
- 分析PCIe连接信息
- 生成JSON格式的结果报告

**使用方法：**
```bash
# 基本使用
python hardware_topology_analyzer.py

# 保存结果到文件
python hardware_topology_analyzer.py --save results.json

# 静默模式
python hardware_topology_analyzer.py --quiet
```

### 2. simple_hardware_analyzer.py - 简化版分析器

**功能特点：**
- 适用于bandwidthTest不可用的环境
- 详细的系统信息收集
- GPU状态监控
- 基础性能测试

**使用方法：**
```bash
# 运行分析
python simple_hardware_analyzer.py

# 保存结果
python simple_hardware_analyzer.py --save analysis.json

# 简要模式
python simple_hardware_analyzer.py --simple
```

### 3. advanced_bandwidth_tester.py - 高级带宽测试器

**功能特点：**
- 最全面的硬件分析工具
- 多种带宽测试方法
- 详细的拓扑结构分析
- 延迟测试和多GPU测试
- 生成详细的文本报告

**使用方法：**
```bash
# 基本分析
python advanced_bandwidth_tester.py

# 保存JSON结果和文本报告
python advanced_bandwidth_tester.py --save-json data.json --save-report report.txt

# 静默模式
python advanced_bandwidth_tester.py --quiet

# 查看帮助
python advanced_bandwidth_tester.py --help
```

## 前置要求

### 必需组件
- NVIDIA GPU和驱动程序
- nvidia-smi工具（通常随驱动程序安装）
- Python 3.6+

### 可选组件
- NVIDIA CUDA Toolkit（包含bandwidthTest工具）
- psutil库（用于系统信息收集）

### 安装依赖
```bash
# 安装Python依赖
pip install psutil

# 检查nvidia-smi是否可用
nvidia-smi --version

# 检查bandwidthTest是否可用（可选）
bandwidthTest --help
```

## 输出说明

### 拓扑结构信息
- **PCIe连接**: 显示每个GPU的PCIe总线ID、代数和宽度
- **拓扑矩阵**: 显示GPU之间的连接类型（NVLink、PCIe等）
- **NVLink连接**: 检测和分析高速GPU间连接

### 带宽测试结果
- **Host to Device**: CPU到GPU的传输带宽
- **Device to Host**: GPU到CPU的传输带宽
- **Device to Device**: GPU间直接传输带宽（如果支持）

### 性能指标
- **查询延迟**: nvidia-smi命令的响应时间
- **温度和功耗**: 实时GPU状态监控
- **内存使用**: GPU显存使用情况

## 示例输出

```
================================================================================
硬件拓扑结构分析报告
================================================================================

📊 GPU基本信息:
--------------------------------------------------
GPU 0:
  名称: NVIDIA GeForce RTX 4090
  UUID: GPU-12345678-1234-1234-1234-123456789012
  显存: 2048 MiB / 24564 MiB
  驱动版本: 535.86.10
  CUDA版本: 12.2

🔌 PCIe连接信息:
--------------------------------------------------
GPU0:
  Bus ID: 00000000:01:00.0
  PCIe Generation: 4
  PCIe Width: x16

⚡ 带宽测试结果:
--------------------------------------------------
Host to Device:
  传输大小 33554432 bytes: 25600.50 MB/s
Device to Host:
  传输大小 33554432 bytes: 26800.25 MB/s
```

## 故障排除

### 常见问题

1. **"nvidia-smi: command not found"**
   - 确保已安装NVIDIA驱动程序
   - 检查PATH环境变量

2. **"bandwidthTest: command not found"**
   - 安装CUDA Toolkit
   - 使用simple_hardware_analyzer.py作为替代

3. **权限错误**
   - 某些GPU信息可能需要管理员权限
   - 尝试使用sudo运行（在安全环境中）

4. **无法检测到GPU**
   - 确认GPU硬件正常工作
   - 检查驱动程序是否正确安装

### 调试建议

1. 首先运行简单版本确认基本功能
2. 检查工具可用性状态
3. 查看详细错误信息（使用--verbose或去掉--quiet）

## 脚本特性对比

| 特性 | hardware_topology_analyzer.py | simple_hardware_analyzer.py | advanced_bandwidth_tester.py |
|------|:----------------------------:|:--------------------------:|:---------------------------:|
| nvidia-smi支持 | ✅ | ✅ | ✅ |
| bandwidthTest支持 | ✅ | ❌ | ✅ |
| 系统信息收集 | 基础 | 详细 | 全面 |
| 拓扑分析 | 基础 | 详细 | 高级 |
| 性能测试 | 标准 | 简化 | 综合 |
| 报告格式 | 控制台+JSON | 控制台+JSON | 控制台+JSON+文本 |
| 适用场景 | 快速检查 | 无CUDA环境 | 深入分析 |

## 许可证

此工具集遵循MIT许可证。

## 作者

GitHub Copilot - 2025年8月18日
