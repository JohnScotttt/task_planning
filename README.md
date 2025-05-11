# 室内场景机器人任务规划系统

这是一个基于多模态输入的室内场景机器人任务规划系统。该系统能够理解视觉输入和自然语言指令，并生成可执行的任务序列。

## 功能特点

- 多模态输入处理（视觉+语言）
- 场景解析与理解
- 任务指令理解
- 任务分解与规划
- 冲突解决

## 系统架构

系统由以下主要模块组成：

1. **环境理解模块** (`src/environment/environment_understanding.py`)
   - 场景解析
   - 指令理解
   - 多模态对齐

2. **任务规划模块** (`src/planning/task_planner.py`)
   - 任务分解
   - 原子操作生成
   - 冲突解决

3. **主控制模块** (`src/main.py`)
   - 系统协调
   - 任务处理流程

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd task_planning
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备输入：
   - 准备场景图像
   - 准备自然语言指令

2. 运行系统：
```bash
python src/main.py
```

## 示例

输入：
- 图像：室内场景（包含桌子、杯子、厨房区域）
- 指令："把杯子放进厨房的柜子里"

输出：
```
执行计划：
1. {'type': 'navigate', 'target': '杯子', 'purpose': 'reach_object'}
2. {'type': 'grasp', 'target': '杯子'}
3. {'type': 'navigate', 'target': '柜子', 'purpose': 'reach_destination'}
4. {'type': 'place', 'target': '杯子', 'destination': '柜子'}
```

## 注意事项

1. 确保已安装所有必要的依赖
2. 需要有足够的 GPU 内存运行 VL 模型
3. 输入图像需要清晰可见目标物体

## 未来改进

1. 实现更复杂的场景解析逻辑
2. 添加更多的任务类型支持
3. 优化冲突解决算法
4. 添加实时反馈机制 