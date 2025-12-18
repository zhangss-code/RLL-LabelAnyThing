#!/usr/bin/env python3
"""
行为识别与分析模块
实现用户购买行为识别功能
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BehaviorAnalyzer:
    """行为分析器类"""
    
    def __init__(self, config: Dict):
        """
        初始化行为分析器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.min_confidence = config.get("analysis", {}).get("min_confidence", 0.7)
        
        # 行为状态跟踪
        self.user_tracking = {}  # 用户ID -> 行为状态
        self.behavior_history = []  # 行为历史记录
        
        # 加载模型（这里使用预训练的YOLO模型作为示例）
        self._load_models()
        
        logger.info("行为分析器初始化完成")
    
    def _load_models(self):
        """加载AI模型"""
        try:
            # 这里应该加载实际的模型文件
            # 示例：加载YOLO模型进行目标检测
            model_path = "models/yolov8n.onnx"  # 示例路径
            self.net = cv2.dnn.readNet(model_path)
            
            # 获取输出层名称
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # 类别标签（示例）
            self.classes = ["person", "hand", "product", "payment_device", "vending_machine"]
            
            logger.info(f"加载模型成功: {model_path}")
        except Exception as e:
            logger.warning(f"模型加载失败，使用基础检测方法: {e}")
            self.net = None
            self.output_layers = []
            self.classes = []
    
    def analyze(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """
        分析视频帧中的行为
        
        Args:
            frame: 视频帧
            timestamp: 时间戳
            
        Returns:
            行为事件列表
        """
        events = []
        
        try:
            # 1. 目标检测
            detected_objects = self._detect_objects(frame)
            
            # 2. 用户检测与跟踪
            users = self._detect_users(detected_objects)
            
            # 3. 行为识别
            for user in users:
                user_events = self._analyze_user_behavior(user, frame, timestamp)
                events.extend(user_events)
            
            # 4. 行为流程分析
            process_events = self._analyze_behavior_process(events, timestamp)
            events.extend(process_events)
            
        except Exception as e:
            logger.error(f"行为分析错误: {e}")
        
        return events
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """检测帧中的物体"""
        objects = []
        
        if self.net is not None:
            # 使用深度学习模型检测
            height, width = frame.shape[:2]
            
            # 预处理图像
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            # 解析检测结果
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.min_confidence:
                        # 计算边界框坐标
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # 左上角坐标
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        objects.append({
                            "class": self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}",
                            "confidence": float(confidence),
                            "bbox": (x, y, w, h),
                            "center": (center_x, center_y)
                        })
        else:
            # 使用基础检测方法（示例）
            # 这里可以使用Haar级联或HOG检测器
            pass
        
        return objects
    
    def _detect_users(self, objects: List[Dict]) -> List[Dict]:
        """从检测到的物体中识别用户"""
        users = []
        
        # 查找人物检测
        person_objects = [obj for obj in objects if obj["class"] == "person"]
        
        for person in person_objects:
            user_id = self._generate_user_id(person)
            
            # 查找关联的手部
            hand_objects = self._find_nearby_objects(person, objects, "hand", max_distance=100)
            
            # 查找附近的商品
            product_objects = self._find_nearby_objects(person, objects, "product", max_distance=200)
            
            # 查找附近的支付设备
            payment_objects = self._find_nearby_objects(person, objects, "payment_device", max_distance=150)
            
            user = {
                "user_id": user_id,
                "person": person,
                "hands": hand_objects,
                "nearby_products": product_objects,
                "nearby_payment": payment_objects,
                "position": person["center"],
                "timestamp": time.time()
            }
            
            # 更新用户跟踪
            self._update_user_tracking(user_id, user)
            
            users.append(user)
        
        return users
    
    def _generate_user_id(self, person_obj: Dict) -> str:
        """生成用户ID（基于位置和特征）"""
        # 简单实现：基于边界框中心位置
        center_x, center_y = person_obj["center"]
        return f"user_{int(center_x)}_{int(center_y)}"
    
    def _find_nearby_objects(self, reference_obj: Dict, objects: List[Dict], 
                            target_class: str, max_distance: float) -> List[Dict]:
        """查找附近的特定类别物体"""
        nearby = []
        ref_x, ref_y = reference_obj["center"]
        
        for obj in objects:
            if obj["class"] == target_class:
                obj_x, obj_y = obj["center"]
                distance = np.sqrt((obj_x - ref_x)**2 + (obj_y - ref_y)**2)
                
                if distance <= max_distance:
                    nearby.append(obj)
        
        return nearby
    
    def _update_user_tracking(self, user_id: str, user_data: Dict):
        """更新用户跟踪状态"""
        if user_id not in self.user_tracking:
            self.user_tracking[user_id] = {
                "first_seen": time.time(),
                "last_seen": time.time(),
                "behavior_stage": "approaching",  # approaching, selecting, paying, retrieving, leaving
                "selected_product": None,
                "payment_attempts": 0,
                "retrieval_attempts": 0,
                "position_history": [],
                "behavior_history": []
            }
        
        tracking = self.user_tracking[user_id]
        tracking["last_seen"] = time.time()
        tracking["position_history"].append({
            "position": user_data["position"],
            "timestamp": user_data["timestamp"]
        })
        
        # 保留最近100个位置
        if len(tracking["position_history"]) > 100:
            tracking["position_history"] = tracking["position_history"][-100:]
    
    def _analyze_user_behavior(self, user: Dict, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """分析单个用户的行为"""
        events = []
        user_id = user["user_id"]
        tracking = self.user_tracking.get(user_id, {})
        
        # 1. 检测选择行为
        selection_events = self._detect_selection_behavior(user, tracking)
        events.extend(selection_events)
        
        # 2. 检测支付行为
        payment_events = self._detect_payment_behavior(user, tracking)
        events.extend(payment_events)
        
        # 3. 检测取货行为
        retrieval_events = self._detect_retrieval_behavior(user, tracking)
        events.extend(retrieval_events)
        
        # 4. 更新行为阶段
        self._update_behavior_stage(user_id, events)
        
        # 记录行为历史
        if events:
            for event in events:
                tracking["behavior_history"].append(event)
                # 保留最近50个行为
                if len(tracking["behavior_history"]) > 50:
                    tracking["behavior_history"] = tracking["behavior_history"][-50:]
        
        return events
    
    def _detect_selection_behavior(self, user: Dict, tracking: Dict) -> List[Dict]:
        """检测商品选择行为"""
        events = []
        
        # 检查用户是否在看/指向商品
        if user["nearby_products"]:
            for product in user["nearby_products"]:
                # 检查手部是否指向商品
                hand_near_product = False
                for hand in user["hands"]:
                    hand_x, hand_y = hand["center"]
                    product_x, product_y = product["center"]
                    distance = np.sqrt((hand_x - product_x)**2 + (hand_y - product_y)**2)
                    
                    if distance < 50:  # 手部靠近商品
                        hand_near_product = True
                        break
                
                if hand_near_product and tracking.get("behavior_stage") in ["approaching", "selecting"]:
                    event = {
                        "action": "product_selection",
                        "user_id": user["user_id"],
                        "product": product,
                        "confidence": product["confidence"],
                        "timestamp": time.time()
                    }
                    events.append(event)
                    
                    # 更新跟踪状态
                    tracking["selected_product"] = product
                    tracking["behavior_stage"] = "selecting"
        
        return events
    
    def _detect_payment_behavior(self, user: Dict, tracking: Dict) -> List[Dict]:
        """检测支付行为"""
        events = []
        
        # 检查用户是否在进行支付
        if user["nearby_payment"] and tracking.get("behavior_stage") in ["selecting", "paying"]:
            for payment_device in user["nearby_payment"]:
                # 检查手部是否在支付设备附近
                hand_near_payment = False
                for hand in user["hands"]:
                    hand_x, hand_y = hand["center"]
                    payment_x, payment_y = payment_device["center"]
                    distance = np.sqrt((hand_x - payment_x)**2 + (hand_y - payment_y)**2)
                    
                    if distance < 30:  # 手部在支付设备上
                        hand_near_payment = True
                        break
                
                if hand_near_payment:
                    event = {
                        "action": "payment_attempt",
                        "user_id": user["user_id"],
                        "payment_device": payment_device,
                        "attempt_number": tracking.get("payment_attempts", 0) + 1,
                        "timestamp": time.time()
                    }
                    events.append(event)
                    
                    # 更新跟踪状态
                    tracking["payment_attempts"] = tracking.get("payment_attempts", 0) + 1
                    tracking["behavior_stage"] = "paying"
        
        return events
    
    def _detect_retrieval_behavior(self, user: Dict, tracking: Dict) -> List[Dict]:
        """检测取货行为"""
        events = []
        
        # 检查用户是否在取货口附近
        # 这里需要知道取货口的位置（假设在图像底部中心）
        height, width = 480, 640  # 假设图像尺寸
        retrieval_area = (width//2 - 50, height - 100, 100, 100)  # 取货口区域
        
        user_x, user_y = user["position"]
        retrieval_x, retrieval_y, retrieval_w, retrieval_h = retrieval_area
        
        # 检查用户是否在取货口附近
        in_retrieval_area = (retrieval_x <= user_x <= retrieval_x + retrieval_w and 
                            retrieval_y <= user_y <= retrieval_y + retrieval_h)
        
        if in_retrieval_area and tracking.get("behavior_stage") in ["paying", "retrieving"]:
            # 检查手部是否在取货动作
            hand_low = any(hand["center"][1] > height - 50 for hand in user["hands"])
            
            if hand_low:
                event = {
                    "action": "retrieval_attempt",
                    "user_id": user["user_id"],
                    "retrieval_area": retrieval_area,
                    "attempt_number": tracking.get("retrieval_attempts", 0) + 1,
                    "timestamp": time.time()
                }
                events.append(event)
                
                # 更新跟踪状态
                tracking["retrieval_attempts"] = tracking.get("retrieval_attempts", 0) + 1
                tracking["behavior_stage"] = "retrieving"
        
        return events
    
    def _update_behavior_stage(self, user_id: str, events: List[Dict]):
        """更新用户行为阶段"""
        if user_id not in self.user_tracking:
            return
        
        tracking = self.user_tracking[user_id]
        
        # 根据事件更新阶段
        event_actions = [event["action"] for event in events]
        
        if "product_selection" in event_actions:
            tracking["behavior_stage"] = "selecting"
        elif "payment_attempt" in event_actions:
            tracking["behavior_stage"] = "paying"
        elif "retrieval_attempt" in event_actions:
            tracking["behavior_stage"] = "retrieving"
        
        # 如果用户离开（长时间未检测到）
        time_since_last_seen = time.time() - tracking["last_seen"]
        if time_since_last_seen > 10.0:  # 10秒未见到用户
            tracking["behavior_stage"] = "leaving"
            
            # 清理过期的用户跟踪
            if time_since_last_seen > 60.0:  # 60秒后清理
                if user_id in self.user_tracking:
                    del self.user_tracking[user_id]
    
    def _analyze_behavior_process(self, events: List[Dict], timestamp: datetime) -> List[Dict]:
        """分析完整的行为流程"""
        process_events = []
        
        # 查找完整的购买流程
        for user_id, tracking in self.user_tracking.items():
            behavior_history = tracking.get("behavior_history", [])
            
            # 检查是否有完整的"选择-支付-取货"流程
            has_selection = any(event.get("action") == "product_selection" for event in behavior_history)
            has_payment = any(event.get("action") == "payment_attempt" for event in behavior_history)
            has_retrieval = any(event.get("action") == "retrieval_attempt" for event in behavior_history)
            
            if has_selection and has_payment and has_retrieval:
                # 创建完整的购买流程事件
                process_event = {
                    "action": "complete_purchase_process",
                    "user_id": user_id,
                    "stages": ["selection", "payment", "retrieval"],
                    "duration": time.time() - tracking.get("first_seen", time.time()),
                    "selected_product": tracking.get("selected_product"),
                    "timestamp": timestamp
                }
                process_events.append(process_event)
                
                # 重置该用户的跟踪（流程完成）
                tracking["behavior_stage"] = "completed"
                tracking["selected_product"] = None
                tracking["payment_attempts"] = 0
                tracking["retrieval_attempts"] = 0
        
        return process_events
    
    def get_status(self) -> Dict:
        """获取分析器状态"""
        return {
            "model_loaded": self.net is not None,
            "users_tracking": len(self.user_tracking),
            "min_confidence": self.min_confidence,
            "behavior_stages": {
                user_id: tracking["behavior_stage"]
                for user_id, tracking in self.user_tracking.items()
            }
        }
    
    def visualize_behavior(self, frame: np.ndarray, events: List[Dict]) -> np.ndarray:
        """可视化行为分析结果"""
        vis_frame = frame.copy()
        
        # 绘制检测到的物体
        for event in events:
            if "bbox" in event:
                x, y, w, h = event["bbox"]
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加标签
                label = f"{event.get('class', 'object')}: {event.get('confidence', 0):.2f}"
                cv2.putText(vis_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制行为事件
        for event in events:
            if "action" in event:
                action = event["action"]
                position = event.get("position", (50, 50))
                
                cv2.putText(vis_frame, action, position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return vis_frame
