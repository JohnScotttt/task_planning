from typing import Dict, List
import os
from src.environment.environment_understanding import EnvironmentUnderstanding
from src.planning.task_planner import TaskPlanner

class TaskPlanningSystem:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """初始化任务规划系统
        
        Args:
            model_path: VL模型的路径
        """
        print("正在初始化任务规划系统...")
        self.environment_understanding = EnvironmentUnderstanding(model_path)
        self.task_planner = TaskPlanner()
        print("系统初始化完成")
    
    def process_task(self, image_path: str, instruction: str) -> List[Dict]:
        """处理任务
        
        Args:
            image_path: 输入图像的路径
            instruction: 自然语言指令
            
        Returns:
            List[Dict]: 执行计划
        """
        print(f"\n开始处理任务...")
        print(f"图像路径: {image_path}")
        print(f"任务指令: {instruction}")
        
        # 1. 环境理解
        print("\n1. 开始环境理解...")
        scene_info = self.environment_understanding.parse_scene(image_path)
        print(f"场景信息: {scene_info}")
        
        # 2. 指令理解
        print("\n2. 开始指令理解...")
        task_goal = self.environment_understanding.understand_instruction(
            instruction, scene_info
        )
        print(f"任务目标: {task_goal}")
        
        # 3. 任务规划
        print("\n3. 开始任务规划...")
        action_sequence = self.task_planner.plan(task_goal, scene_info)
        print(f"生成动作序列: {action_sequence}")
        
        return action_sequence

def main():
    # 示例使用
    system = TaskPlanningSystem()
    
    # 假设我们有一个测试图像和指令
    image_path = "test_image.png"
    instruction = "把杯子放进厨房的柜子里"
    
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 {image_path}")
        return
    
    # 处理任务
    action_sequence = system.process_task(image_path, instruction)
    
    # 打印执行计划
    print("\n执行计划：")
    for i, action in enumerate(action_sequence, 1):
        print(f"{i}. {action}")

if __name__ == "__main__":
    main() 