import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import TaskPlanningSystem

def main():
    try:
        # 创建系统实例
        system = TaskPlanningSystem()
        
        # 处理任务
        image_path = "image.png"
        instruction = "把锅放进厨房的柜子里"
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"错误：找不到图像文件 {image_path}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"目录内容: {os.listdir('.')}")
            return
        
        print(f"\n开始处理任务...")
        print(f"图像文件: {image_path}")
        print(f"任务指令: {instruction}")
        
        # 处理任务
        action_sequence = system.process_task(image_path, instruction)
        
        # 打印执行计划
        if action_sequence:
            print("\n执行计划：")
            for i, action in enumerate(action_sequence, 1):
                print(f"{i}. {action}")
        else:
            print("\n警告：未能生成执行计划")
            
    except Exception as e:
        print(f"\n发生错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
