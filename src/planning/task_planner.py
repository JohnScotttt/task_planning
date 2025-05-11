from typing import Dict, List, Tuple
from enum import Enum
import numpy as np

class ActionType(Enum):
    NAVIGATE = "navigate"
    GRASP = "grasp"
    PLACE = "place"
    MOVE = "move"
    ROTATE = "rotate"

class TaskPlanner:
    def __init__(self):
        """初始化任务规划器"""
        self.atomic_actions = {
            ActionType.NAVIGATE: self._plan_navigation,
            ActionType.GRASP: self._plan_grasp,
            ActionType.PLACE: self._plan_place,
            ActionType.MOVE: self._plan_move,
            ActionType.ROTATE: self._plan_rotate
        }
        
        # 定义动作优先级
        self.action_priorities = {
            ActionType.NAVIGATE: 1,
            ActionType.GRASP: 2,
            ActionType.PLACE: 3,
            ActionType.MOVE: 2,
            ActionType.ROTATE: 2
        }
    
    def plan(self, task_goal: Dict, scene_info: Dict) -> List[Dict]:
        """生成任务执行计划
        
        Args:
            task_goal: 任务目标
            scene_info: 场景信息
            
        Returns:
            List[Dict]: 原子操作序列
        """
        # 检查目标物体是否存在
        target_object = task_goal.get("target_object", "")
        if target_object:
            target_exists = False
            for obj in scene_info.get("objects", []):
                if target_object in obj.get("name", ""):
                    target_exists = True
                    break
            if not target_exists:
                raise ValueError(f"错误：在场景中未找到目标物体 '{target_object}'")
        
        # 检查目标位置是否存在
        destination = task_goal.get("destination", "")
        if destination:
            dest_exists = False
            for obj in scene_info.get("objects", []):
                if (destination in obj.get("name", "") or 
                    (obj.get("type") == "cabinet" and "柜" in destination)):
                    dest_exists = True
                    break
            if not dest_exists:
                raise ValueError(f"错误：在场景中未找到目标位置 '{destination}'")
        
        # 1. 任务分解
        sub_tasks = self._decompose_task(task_goal, scene_info)
        
        # 2. 生成原子操作序列
        action_sequence = []
        for sub_task in sub_tasks:
            try:
                actions = self._generate_actions(sub_task, scene_info)
                if not actions:
                    print(f"警告：无法为子任务生成动作：{sub_task}")
                action_sequence.extend(actions)
            except Exception as e:
                print(f"警告：处理子任务时出错：{str(e)}")
                continue
        
        # 3. 解决规划冲突
        try:
            action_sequence = self._resolve_conflicts(action_sequence, scene_info)
        except Exception as e:
            print(f"警告：解决规划冲突时出错：{str(e)}")
        
        if not action_sequence:
            raise ValueError("错误：无法生成有效的执行计划")
            
        return action_sequence
    
    def _decompose_task(self, task_goal: Dict, scene_info: Dict = None) -> List[Dict]:
        """将高层任务目标分解为子任务，支持通用物体和目标位置模糊匹配
        
        Args:
            task_goal: 任务目标
            scene_info: 场景信息（可选）
            
        Returns:
            List[Dict]: 子任务列表
        """
        sub_tasks = []
        action = task_goal.get("action", "")
        target_object = task_goal.get("target_object", "")
        destination = task_goal.get("destination", "")
        
        # 1. 智能查找目标物体的实际位置
        obj_location = None
        if scene_info and target_object:
            for obj in scene_info.get("objects", []):
                if target_object in obj.get("name", ""):
                    obj_location = obj.get("location", "")
                    break
        
        # 2. 智能查找目标位置的实际对象名
        dest_name = None
        if scene_info and destination:
            # 优先匹配类型为 cabinet 的物体
            for obj in scene_info.get("objects", []):
                if obj.get("type") == "cabinet":
                    dest_name = obj["name"]
                    break
            # 如果没找到 cabinet 类型，尝试模糊匹配
            if not dest_name:
                for obj in scene_info.get("objects", []):
                    if any(key in obj.get("name", "") for key in [destination, "柜", "橱柜", "cabinet"]):
                        dest_name = obj["name"]
                        break
        
        if not obj_location and target_object:
            print(f"警告：未找到目标物体 '{target_object}' 的具体位置")
            
        if not dest_name and destination:
            print(f"警告：未找到目标位置 '{destination}' 的具体位置")
        
        # 3. 针对不同动作类型分解
        if action == "place":
            if not target_object:
                raise ValueError("错误：放置动作需要指定目标物体")
            if not destination:
                raise ValueError("错误：放置动作需要指定目标位置")
                
            sub_tasks.extend([
                {
                    "action": ActionType.NAVIGATE,
                    "target": obj_location or target_object,
                    "purpose": "reach_object"
                },
                {
                    "action": ActionType.GRASP,
                    "target": target_object
                },
                {
                    "action": ActionType.NAVIGATE,
                    "target": dest_name or destination,
                    "purpose": "reach_destination"
                },
                {
                    "action": ActionType.PLACE,
                    "target": target_object,
                    "destination": dest_name or destination
                }
            ])
        elif action == "move":
            if not target_object:
                raise ValueError("错误：移动动作需要指定目标物体")
            if not destination:
                raise ValueError("错误：移动动作需要指定目标位置")
                
            sub_tasks.extend([
                {
                    "action": ActionType.NAVIGATE,
                    "target": obj_location or target_object,
                    "purpose": "reach_object"
                },
                {
                    "action": ActionType.GRASP,
                    "target": target_object
                },
                {
                    "action": ActionType.NAVIGATE,
                    "target": dest_name or destination,
                    "purpose": "reach_destination"
                },
                {
                    "action": ActionType.MOVE,
                    "target": target_object,
                    "destination": dest_name or destination
                }
            ])
        elif action == "grasp":
            if not target_object:
                raise ValueError("错误：抓取动作需要指定目标物体")
                
            sub_tasks.extend([
                {
                    "action": ActionType.NAVIGATE,
                    "target": obj_location or target_object,
                    "purpose": "reach_object"
                },
                {
                    "action": ActionType.GRASP,
                    "target": target_object
                }
            ])
        elif action == "rotate":
            if not target_object:
                raise ValueError("错误：旋转动作需要指定目标物体")
                
            sub_tasks.extend([
                {
                    "action": ActionType.NAVIGATE,
                    "target": obj_location or target_object,
                    "purpose": "reach_object"
                },
                {
                    "action": ActionType.ROTATE,
                    "target": target_object,
                    "angle": task_goal.get("angle", 90)
                }
            ])
        else:
            # 默认只导航到目标
            if not target_object:
                raise ValueError("错误：导航动作需要指定目标")
                
            sub_tasks.append({
                "action": ActionType.NAVIGATE,
                "target": obj_location or target_object,
                "purpose": "reach_object"
            })
        return sub_tasks
    
    def _generate_actions(self, sub_task: Dict, scene_info: Dict) -> List[Dict]:
        """为子任务生成具体的原子操作
        
        Args:
            sub_task: 子任务
            scene_info: 场景信息
            
        Returns:
            List[Dict]: 原子操作列表
        """
        action_type = sub_task["action"]
        if action_type in self.atomic_actions:
            return self.atomic_actions[action_type](sub_task, scene_info)
        return []
    
    def _resolve_conflicts(self, action_sequence: List[Dict], scene_info: Dict) -> List[Dict]:
        """解决规划冲突
        
        Args:
            action_sequence: 原始动作序列
            scene_info: 场景信息
            
        Returns:
            List[Dict]: 解决冲突后的动作序列
        """
        # 1. 检查资源冲突
        resource_conflicts = self._check_resource_conflicts(action_sequence)
        
        # 2. 检查空间冲突
        space_conflicts = self._check_space_conflicts(action_sequence, scene_info)
        
        # 3. 解决冲突
        resolved_sequence = action_sequence.copy()
        
        # 处理资源冲突
        for conflict in resource_conflicts:
            # 根据优先级重新排序
            action1, action2 = conflict
            if self.action_priorities[action1["type"]] > self.action_priorities[action2["type"]]:
                resolved_sequence.remove(action2)
                resolved_sequence.insert(resolved_sequence.index(action1) + 1, action2)
        
        # 处理空间冲突
        for conflict in space_conflicts:
            # 添加中间导航动作
            action1, action2 = conflict
            intermediate_nav = {
                "type": ActionType.NAVIGATE,
                "target": "safe_position",
                "purpose": "avoid_conflict"
            }
            resolved_sequence.insert(resolved_sequence.index(action1) + 1, intermediate_nav)
        
        return resolved_sequence
    
    def _check_resource_conflicts(self, action_sequence: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """检查资源冲突
        
        Args:
            action_sequence: 动作序列
            
        Returns:
            List[Tuple[Dict, Dict]]: 冲突的动作对列表
        """
        conflicts = []
        for i, action1 in enumerate(action_sequence):
            for action2 in action_sequence[i+1:]:
                # 检查是否使用相同的目标物体
                if (action1["type"] in [ActionType.GRASP, ActionType.MOVE] and
                    action2["type"] in [ActionType.GRASP, ActionType.MOVE] and
                    action1["target"] == action2["target"]):
                    conflicts.append((action1, action2))
        return conflicts
    
    def _check_space_conflicts(self, action_sequence: List[Dict], scene_info: Dict) -> List[Tuple[Dict, Dict]]:
        """检查空间冲突
        
        Args:
            action_sequence: 动作序列
            scene_info: 场景信息
            
        Returns:
            List[Tuple[Dict, Dict]]: 冲突的动作对列表
        """
        conflicts = []
        for i, action1 in enumerate(action_sequence):
            for action2 in action_sequence[i+1:]:
                # 检查动作是否在相近的空间位置
                if (action1["type"] == ActionType.PLACE and
                    action2["type"] == ActionType.PLACE and
                    self._is_nearby(action1["destination"], action2["destination"], scene_info)):
                    conflicts.append((action1, action2))
        return conflicts
    
    def _is_nearby(self, pos1: str, pos2: str, scene_info: Dict) -> bool:
        """检查两个位置是否相近
        
        Args:
            pos1: 位置1
            pos2: 位置2
            scene_info: 场景信息
            
        Returns:
            bool: 是否相近
        """
        # TODO: 实现更精确的空间关系判断
        return pos1 == pos2
    
    def _plan_navigation(self, sub_task: Dict, scene_info: Dict) -> List[Dict]:
        """规划导航动作
        
        Args:
            sub_task: 导航子任务
            scene_info: 场景信息
            
        Returns:
            List[Dict]: 导航动作序列
        """
        target = sub_task["target"]
        purpose = sub_task["purpose"]
        
        # 获取目标位置信息
        target_info = None
        for obj in scene_info["objects"]:
            if obj["name"] == target:
                target_info = obj
                break
        
        if not target_info:
            # 如果找不到精确匹配，尝试模糊匹配
            for obj in scene_info["objects"]:
                if target in obj["name"]:
                    target_info = obj
                    break
        
        if not target_info:
            return []
        
        # 获取目标位置
        target_location = None
        if "location" in target_info:
            target_location = target_info["location"]
        elif "type" in target_info:
            # 如果是特定类型的物体（如cabinet），使用其名称作为位置
            target_location = target_info["name"]
        else:
            # 如果既没有location也没有type，使用物体名称作为位置
            target_location = target_info["name"]
        
        # 生成导航路径点
        waypoints = self._generate_waypoints(target_location)
        
        # 构建导航动作序列
        nav_actions = []
        for waypoint in waypoints:
            nav_actions.append({
                "type": ActionType.NAVIGATE,
                "target": waypoint,
                "purpose": purpose
            })
        
        return nav_actions
    
    def _generate_waypoints(self, target_location: str) -> List[str]:
        """生成导航路径点
        
        Args:
            target_location: 目标位置
            
        Returns:
            List[str]: 路径点列表
        """
        # TODO: 实现更复杂的路径规划
        return [target_location]
    
    def _plan_grasp(self, sub_task: Dict, scene_info: Dict) -> List[Dict]:
        """规划抓取动作
        
        Args:
            sub_task: 抓取子任务
            scene_info: 场景信息
            
        Returns:
            List[Dict]: 抓取动作序列
        """
        target = sub_task["target"]
        
        # 获取目标物体信息
        target_info = None
        for obj in scene_info["objects"]:
            if obj["name"] == target:
                target_info = obj
                break
        
        if not target_info:
            return []
        
        # 根据物体属性调整抓取参数
        grasp_params = {
            "force": "normal",
            "orientation": "default"
        }
        
        if "fragile" in target_info.get("attributes", {}):
            grasp_params["force"] = "gentle"
        
        return [{
            "type": ActionType.GRASP,
            "target": target,
            "parameters": grasp_params
        }]
    
    def _plan_place(self, sub_task: Dict, scene_info: Dict) -> List[Dict]:
        """规划放置动作
        
        Args:
            sub_task: 放置子任务
            scene_info: 场景信息
            
        Returns:
            List[Dict]: 放置动作序列
        """
        target = sub_task["target"]
        destination = sub_task["destination"]
        
        # 获取目标位置信息
        dest_info = None
        for obj in scene_info["objects"]:
            if obj["name"] == destination:
                dest_info = obj
                break
        
        if not dest_info:
            return []
        
        # 根据目标位置属性调整放置参数
        place_params = {
            "height": "default",
            "orientation": "default",
            "stability": "normal"
        }
        
        if "shelf" in destination.lower():
            place_params["height"] = "shelf_height"
        elif "table" in destination.lower():
            place_params["height"] = "table_height"
        
        return [{
            "type": ActionType.PLACE,
            "target": target,
            "destination": destination,
            "parameters": place_params
        }]
    
    def _plan_move(self, sub_task: Dict, scene_info: Dict) -> List[Dict]:
        """规划移动动作
        
        Args:
            sub_task: 移动子任务
            scene_info: 场景信息
            
        Returns:
            List[Dict]: 移动动作序列
        """
        target = sub_task["target"]
        destination = sub_task["destination"]
        
        # 生成移动路径
        waypoints = self._generate_waypoints(destination)
        
        # 构建移动动作序列
        move_actions = []
        for waypoint in waypoints:
            move_actions.append({
                "type": ActionType.MOVE,
                "target": target,
                "destination": waypoint
            })
        
        return move_actions
    
    def _plan_rotate(self, sub_task: Dict, scene_info: Dict) -> List[Dict]:
        """规划旋转动作
        
        Args:
            sub_task: 旋转子任务
            scene_info: 场景信息
            
        Returns:
            List[Dict]: 旋转动作序列
        """
        target = sub_task["target"]
        angle = sub_task["angle"]
        
        # 根据角度生成旋转动作
        return [{
            "type": ActionType.ROTATE,
            "target": target,
            "angle": angle,
            "direction": "clockwise" if angle > 0 else "counterclockwise"
        }] 