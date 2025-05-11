from typing import Dict, List, Tuple
import torch
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class EnvironmentUnderstanding:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """初始化环境理解模块
        
        Args:
            model_path: VL模型的路径
        """
        print("正在加载VL模型...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )
        
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        print("VL模型加载完成")
        
    def parse_scene(self, image_path: str) -> Dict:
        """解析场景，识别物体及其属性
        
        Args:
            image_path: 输入图像的路径
            
        Returns:
            Dict: 包含场景中物体信息的字典
        """
        print(f"正在解析场景图像: {image_path}")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "请详细描述场景中的物体，包括它们的位置、状态和属性。"},
                ]
            }
        ]
        
        # 处理输入
        print("正在处理输入...")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # 生成描述
        print("正在生成场景描述...")
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        scene_description = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        print(f"场景描述: {scene_description}")
        
        # 解析场景描述
        print("正在解析场景描述...")
        scene_info = self._parse_scene_description(scene_description)
        print(f"解析结果: {scene_info}")
        
        return scene_info
    
    def _parse_scene_description(self, description: str) -> Dict:
        """将场景描述解析为结构化的物体信息
        
        Args:
            description: VL模型生成的场景描述
            
        Returns:
            Dict: 结构化的场景信息
        """
        objects = []
        relationships = []
        scene_graph = {}
        
        # 1. 先用原有正则提取
        object_patterns = [
            r"([^，。]+)(?:在|位于|上面|里面|旁边)([^，。]+)",
            r"([^，。]+)(?:上|里|中|旁)(?:有|放置着|放着)([^，。]+)",
            r"([^，。]+)(?:是|为)([^，。]+)"
        ]
        for pattern in object_patterns:
            matches = re.finditer(pattern, description)
            for match in matches:
                obj_name = match.group(1).strip()
                location = match.group(2).strip()
                attributes = {}
                if "白色" in match.group(0):
                    attributes["color"] = "white"
                if "粉色" in match.group(0):
                    attributes["color"] = "pink"
                if "黑色" in match.group(0):
                    attributes["color"] = "black"
                if "木质" in match.group(0):
                    attributes["material"] = "wood"
                if "光滑" in match.group(0):
                    attributes["texture"] = "smooth"
                obj_info = {
                    "name": obj_name,
                    "location": location,
                    "attributes": attributes
                }
                objects.append(obj_info)
                if obj_name not in scene_graph:
                    scene_graph[obj_name] = {}
                scene_graph[obj_name]["location"] = location
                relationship = {
                    "subject": obj_name,
                    "relation": "located_at",
                    "object": location
                }
                relationships.append(relationship)
        
        # 2. 针对"锅"做专门提取
        for line in description.splitlines():
            if '锅' in line:
                color = ''
                if '粉色' in line:
                    color = 'pink'
                elif '白色' in line:
                    color = 'white'
                elif '黑色' in line:
                    color = 'black'
                location = ''
                if '右侧' in line:
                    location = '台面右侧'
                elif '左侧' in line:
                    location = '台面左侧'
                elif '台面' in line:
                    location = '台面'
                elif '燃气灶' in line:
                    location = '燃气灶'
                obj_info = {
                    "name": "锅",
                    "location": location,
                    "attributes": {"color": color} if color else {}
                }
                # 避免重复添加
                if not any(o for o in objects if o["name"] == "锅" and o["location"] == location):
                    objects.append(obj_info)
                    scene_graph["锅"] = {"location": location}
        
        # 3. 特殊物体
        special_objects = {
            "橱柜": "cabinet",
            "台面": "counter",
            "燃气灶": "stove",
            "洗碗机": "dishwasher",
            "抽油烟机": "range_hood"
        }
        for obj_name, obj_type in special_objects.items():
            if obj_name in description:
                obj_info = {
                    "name": obj_name,
                    "type": obj_type,
                    "attributes": {}
                }
                if not any(o for o in objects if o["name"] == obj_name):
                    objects.append(obj_info)
                    scene_graph[obj_name] = {"type": obj_type}
        return {
            "objects": objects,
            "relationships": relationships,
            "scene_graph": scene_graph
        }
    
    def understand_instruction(self, instruction: str, scene_info: Dict) -> Dict:
        """理解任务指令
        
        Args:
            instruction: 自然语言指令
            scene_info: 场景信息
            
        Returns:
            Dict: 结构化的任务目标
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"场景信息：{scene_info}\n任务指令：{instruction}\n请解析这个任务指令，包括动作、目标对象和约束条件。"},
                ]
            }
        ]
        
        # 处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # 生成解析结果
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        instruction_understanding = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return self._parse_instruction(instruction_understanding)
    
    def _parse_instruction(self, understanding: str) -> Dict:
        """将指令理解结果解析为结构化的任务目标
        
        Args:
            understanding: VL模型生成的指令理解结果
            
        Returns:
            Dict: 结构化的任务目标
        """
        # 优先处理"把X放进Y"结构
        m = re.search(r'把(.+?)(?:放进|放到|放入|放在)(.+?)(?:[，。\n]|$)', understanding)
        if m:
            target_object = m.group(1).strip()
            destination = m.group(2).strip()
            action = 'place'
            return {
                "action": action,
                "target_object": target_object,
                "destination": destination,
                "constraints": []
            }
        # 兼容英文指令
        m2 = re.search(r'put the (.+?) (?:in|on|into|to) the (.+?)[\.,\n]', understanding, re.I)
        if m2:
            target_object = m2.group(1).strip()
            destination = m2.group(2).strip()
            action = 'place'
            return {
                "action": action,
                "target_object": target_object,
                "destination": destination,
                "constraints": []
            }
        # 回退原有逻辑
        return {
            "action": "",
            "target_object": "",
            "destination": "",
            "constraints": []
        } 