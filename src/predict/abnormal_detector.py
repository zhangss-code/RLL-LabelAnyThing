#!/usr/bin/env python3
"""
异常行为检测模块
实现暴力破坏、异常停留、多次尝试等异常行为检测
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AbnormalDetector:
    """异常行为检测器类"""
    
    def __init__(self, config: Dict):
        """
        初始化异常检测器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.alerts_config = config.get("alerts", {})
        
        # 异常状态跟踪
        self.abnormal_states = {}  # 区域/用户ID -> 异常状态
        self.alert_history = []  # 警报历史记录
        
        # 阈值配置
        self.violent_threshold = self.alerts_config.get("violent_behavior_threshold", 0.8)
        self.loitering_threshold = self.alerts_config.get("loitering_time_threshold", 60)  # 秒
        self.multiple_attempts_threshold = self.alerts_config.get("multiple_attempts_threshold", 3)
        
        # 运动检测器（用于暴力行为检测）
        self.motion_detector = MotionDetector()
        
        # 停留时间跟踪
        self.loitering_tracker = LoiteringTracker(self.loitering_threshold)
        
        # 尝试次数跟踪
        self.attempt_tracker = AttemptTracker(self.multiple_attempts_threshold)
        
        logger.info("异常检测器初始化完成")
    
    def detect(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """
        检测视频帧中的异常行为
        
        Args:
            frame: 视频帧
            timestamp: 时间戳
            
        Returns:
            异常事件列表
        """
        abnormal_events = []
        
        try:
            # 1. 暴力破坏检测
            violent_events = self._detect_violent_behavior(frame, timestamp)
            abnormal_events.extend(violent_events)
            
            # 2. 异常停留检测
            loitering_events = self._detect_loitering_behavior(frame, timestamp)
            abnormal_events.extend(loitering_events)
            
            # 3. 多次尝试检测
            attempt_events = self._detect_multiple_attempts(frame, timestamp)
            abnormal_events.extend(attempt_events)
            
            # 4. 支付异常检测
            payment_events = self._detect_payment_anomalies(frame, timestamp)
            abnormal_events.extend(payment_events)
            
            # 5. 非法操作检测
            illegal_events = self._detect_illegal_operations(frame, timestamp)
            abnormal_events.extend(illegal_events)
            
            # 6. 更新异常状态
            self._update_abnormal_states(abnormal_events, timestamp)
            
        except Exception as e:
            logger.error(f"异常检测错误: {e}")
        
        return abnormal_events
    
    def _detect_violent_behavior(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """检测暴力破坏行为"""
        events = []
        
        # 使用运动检测器分析帧
        motion_result = self.motion_detector.analyze(frame)
        
        # 检查是否有剧烈运动
        if motion_result["violent_motion_detected"]:
            violence_score = motion_result["violence_score"]
            
            if violence_score >= self.violent_threshold:
                event = {
                    "type": "violent_behavior",
                    "severity": "high",
                    "score": violence_score,
                    "location": motion_result["motion_center"],
                    "description": "检测到可能的暴力破坏行为",
                    "timestamp": timestamp,
                    "evidence": motion_result.get("motion_mask", None)
                }
                events.append(event)
                
                logger.warning(f"暴力行为检测: 分数={violence_score:.2f}, 位置={motion_result['motion_center']}")
        
        return events
    
    def _detect_loitering_behavior(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """检测异常停留行为"""
        events = []
        
        # 检测人物
        persons = self._detect_persons(frame)
        
        for person in persons:
            person_id = person.get("id", "unknown")
            position = person.get("position", (0, 0))
            
            # 更新停留跟踪
            loitering_result = self.loitering_tracker.update(person_id, position, timestamp)
            
            # 检查是否超过停留阈值
            if loitering_result["is_loitering"]:
                event = {
                    "type": "abnormal_loitering",
                    "severity": "medium",
                    "person_id": person_id,
                    "duration": loitering_result["duration"],
                    "position": position,
                    "description": f"异常停留 {loitering_result['duration']:.1f}秒",
                    "timestamp": timestamp
                }
                events.append(event)
                
                logger.warning(f"异常停留检测: 用户={person_id}, 时长={loitering_result['duration']:.1f}秒")
        
        return events
    
    def _detect_multiple_attempts(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """检测多次尝试行为"""
        events = []
        
        # 检测支付行为（这里简化处理，实际应该从行为分析器获取）
        payment_actions = self._detect_payment_actions(frame)
        
        for payment in payment_actions:
            user_id = payment.get("user_id", "unknown")
            attempt_type = payment.get("type", "payment")
            
            # 更新尝试跟踪
            attempt_result = self.attempt_tracker.update(user_id, attempt_type, timestamp)
            
            # 检查是否超过尝试阈值
            if attempt_result["exceeded_threshold"]:
                event = {
                    "type": "multiple_attempts",
                    "severity": "medium",
                    "user_id": user_id,
                    "attempt_type": attempt_type,
                    "attempt_count": attempt_result["count"],
                    "threshold": self.multiple_attempts_threshold,
                    "description": f"多次{attempt_type}尝试 ({attempt_result['count']}次)",
                    "timestamp": timestamp
                }
                events.append(event)
                
                logger.warning(f"多次尝试检测: 用户={user_id}, 类型={attempt_type}, 次数={attempt_result['count']}")
        
        return events
    
    def _detect_payment_anomalies(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """检测支付异常"""
        events = []
        
        # 检测支付失败模式
        # 这里可以检查支付设备的状态、用户行为模式等
        
        # 示例：检测长时间支付操作
        payment_duration = self._check_payment_duration(frame)
        if payment_duration > 30:  # 支付操作超过30秒
            event = {
                "type": "payment_anomaly",
                "severity": "low",
                "duration": payment_duration,
                "description": f"支付操作异常长时间 ({payment_duration}秒)",
                "timestamp": timestamp
            }
            events.append(event)
        
        # 检测支付后无取货行为
        # 这需要与行为分析器集成
        
        return events
    
    def _detect_illegal_operations(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """检测非法操作"""
        events = []
        
        # 检测非正常访问售货机内部
        internal_access = self._detect_internal_access(frame)
        if internal_access["detected"]:
            event = {
                "type": "illegal_operation",
                "severity": "high",
                "location": internal_access["location"],
                "description": "检测到非法访问售货机内部",
                "timestamp": timestamp,
                "evidence": internal_access.get("evidence", None)
            }
            events.append(event)
            
            logger.warning(f"非法操作检测: 位置={internal_access['location']}")
        
        # 检测破坏性操作（如摇晃机器）
        shaking_detected = self._detect_machine_shaking(frame)
        if shaking_detected:
            event = {
                "type": "machine_shaking",
                "severity": "high",
                "description": "检测到摇晃售货机行为",
                "timestamp": timestamp
            }
            events.append(event)
        
        return events
    
    def _detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """检测帧中的人物"""
        persons = []
        
        # 使用Haar级联检测器检测人物
        try:
            # 加载预训练的人物检测器
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # 检测人物
            boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            
            for i, (x, y, w, h) in enumerate(boxes):
                if weights[i] > 0.5:  # 置信度阈值
                    person_id = f"person_{x}_{y}"
                    persons.append({
                        "id": person_id,
                        "bbox": (x, y, w, h),
                        "position": (x + w//2, y + h//2),
                        "confidence": float(weights[i])
                    })
        except Exception as e:
            logger.error(f"人物检测错误: {e}")
        
        return persons
    
    def _detect_payment_actions(self, frame: np.ndarray) -> List[Dict]:
        """检测支付行为（简化实现）"""
        payments = []
        
        # 这里应该与行为分析器集成
        # 简化实现：检测支付设备区域的活动
        
        # 假设支付设备在图像右侧
        height, width = frame.shape[:2]
        payment_region = (width - 200, 100, 150, 200)  # x, y, w, h
        
        # 检查支付区域是否有运动
        payment_roi = frame[payment_region[1]:payment_region[1]+payment_region[3],
                           payment_region[0]:payment_region[0]+payment_region[2]]
        
        if payment_roi.size > 0:
            # 计算运动强度
            gray = cv2.cvtColor(payment_roi, cv2.COLOR_BGR2GRAY)
            if hasattr(self, 'prev_payment_gray'):
                diff = cv2.absdiff(gray, self.prev_payment_gray)
                motion_intensity = np.mean(diff)
                
                if motion_intensity > 10:  # 运动阈值
                    payments.append({
                        "user_id": "detected_user",
                        "type": "payment",
                        "intensity": motion_intensity,
                        "region": payment_region
                    })
            
            self.prev_payment_gray = gray
        
        return payments
    
    def _check_payment_duration(self, frame: np.ndarray) -> float:
        """检查支付操作持续时间"""
        # 这里应该跟踪支付操作的开始时间
        # 简化实现：返回一个模拟值
        if not hasattr(self, 'payment_start_time'):
            self.payment_start_time = time.time()
        
        return time.time() - self.payment_start_time
    
    def _detect_internal_access(self, frame: np.ndarray) -> Dict:
        """检测非法访问售货机内部"""
        result = {"detected": False, "location": (0, 0)}
        
        # 检测是否有人手伸入售货机内部区域
        # 内部区域通常在上部或侧面
        
        height, width = frame.shape[:2]
        internal_regions = [
            (50, 50, 100, 150),      # 左上内部区域
            (width - 150, 50, 100, 150)  # 右上内部区域
        ]
        
        for region in internal_regions:
            x, y, w, h = region
            roi = frame[y:y+h, x:x+w]
            
            # 使用肤色检测
            skin_mask = self._detect_skin(roi)
            skin_area = np.sum(skin_mask) / 255
            
            if skin_area > 100:  # 检测到足够大的肤色区域
                result["detected"] = True
                result["location"] = (x + w//2, y + h//2)
                result["evidence"] = skin_mask
                break
        
        return result
    
    def _detect_machine_shaking(self, frame: np.ndarray) -> bool:
        """检测机器摇晃"""
        # 通过分析连续帧的运动模式来检测
        
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False
        
        # 计算光流或帧差
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # 分析运动模式
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        
        self.prev_frame = curr_gray
        
        # 如果平均运动幅度很大且方向混乱，可能是摇晃
        return avg_magnitude > 5.0
    
    def _detect_skin(self, image: np.ndarray) -> np.ndarray:
        """检测肤色"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义肤色范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 创建肤色掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        return skin_mask
    
    def _update_abnormal_states(self, events: List[Dict], timestamp: datetime):
        """更新异常状态"""
        for event in events:
            event_type = event.get("type")
            severity = event.get("severity", "medium")
            
            # 记录到警报历史
            self.alert_history.append({
                "type": event_type,
                "severity": severity,
                "timestamp": timestamp,
                "data": event
            })
            
            # 保留最近1000个警报
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
    
    def get_status(self) -> Dict:
        """获取检测器状态"""
        return {
            "active_alerts": len([a for a in self.alert_history 
                                 if time.time() - a["timestamp"].timestamp() < 3600]),  # 最近1小时的警报
            "loitering_tracking": self.loitering_tracker.get_status(),
            "attempt_tracking": self.attempt_tracker.get_status(),
            "thresholds": {
                "violent_behavior": self.violent_threshold,
                "loitering_time": self.loitering_threshold,
                "multiple_attempts": self.multiple_attempts_threshold
            }
        }
    
    def visualize_abnormalities(self, frame: np.ndarray, events: List[Dict]) -> np.ndarray:
        """可视化异常检测结果"""
        vis_frame = frame.copy()
        
        # 颜色映射：根据严重程度使用不同颜色
        severity_colors = {
            "high": (0, 0, 255),    # 红色
            "medium": (0, 165, 255), # 橙色
            "low": (0, 255, 255)     # 黄色
        }
        
        for event in events:
            severity = event.get("severity", "medium")
            color = severity_colors.get(severity, (0, 165, 255))
            
            # 绘制异常位置
            location = event.get("location")
            if location:
                x, y = location
                cv2.circle(vis_frame, (x, y), 20, color, 3)
                cv2.putText(vis_frame, "!", (x-5, y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # 添加异常描述
            description = event.get("description", "异常行为")
            cv2.putText(vis_frame, description, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_frame


class MotionDetector:
    """运动检测器类"""
    
    def __init__(self):
        self.prev_frame = None
        self.motion_history = []
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """分析帧中的运动"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return {
                "violent_motion_detected": False,
                "violence_score": 0.0,
                "motion_center": (0, 0)
            }
        
        # 计算帧差
        frame_diff = cv2.absdiff(gray, self.prev_frame)
        
        # 二值化
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 计算运动区域
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算暴力运动分数
        violence_score = 0.0
        motion_center = (0, 0)
        motion_mask = None
        
        if contours:
            # 找到最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            # 计算运动强度
            motion_intensity = np.mean(frame_diff)
            
            # 计算暴力分数（基于面积和强度）
            violence_score = min(1.0, (area / 10000) * (motion_intensity / 50))
            
            # 计算运动中心
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                motion_center = (cx, cy)
            
            motion_mask = thresh
        
        # 更新前一帧
        self.prev_frame = gray
        
        # 记录运动历史
        self.motion_history.append(violence_score)
        if len(self.motion_history) > 100:
            self.motion_history = self.motion_history[-100:]
        
        return {
            "violent_motion_detected": violence_score > 0.3,
            "violence_score": violence_score,
            "motion_center": motion_center,
            "motion_mask": motion_mask
        }
    
    def get_status(self) -> Dict:
        """获取运动检测器状态"""
        if not self.motion_history:
            return {"average_violence_score": 0.0, "recent_detections": 0}
        
        avg_score = np.mean(self.motion_history)
        recent_detections = len([s for s in self.motion_history[-10:] if s > 0.3])
        
        return {
            "average_violence_score": avg_score,
            "recent_detections": recent_detections
        }


class LoiteringTracker:
    """停留时间跟踪器类"""
    
    def __init__(self, threshold_seconds: float = 60):
        self.threshold = threshold_seconds
        self.tracking_data = {}  # person_id -> {start_time, last_position, last_update}
    
    def update(self, person_id: str, position: Tuple[int, int], timestamp: datetime) -> Dict:
        """更新停留跟踪"""
        current_time = time.time()
        
        if person_id not in self.tracking_data:
            # 新用户，开始跟踪
            self.tracking_data[person_id] = {
                "start_time": current_time,
                "last_position": position,
                "last_update": current_time,
                "position_history": [position]
            }
            duration = 0.0
        else:
            # 更新现有用户
            tracking = self.tracking_data[person_id]
            
            # 检查用户是否移动了（超过一定距离）
            last_x, last_y = tracking["last_position"]
            curr_x, curr_y = position
            distance = np.sqrt((curr_x - last_x)**2 + (curr_y - last_y)**2)
            
            if distance > 50:  # 移动超过50像素，重置计时
                tracking["start_time"] = current_time
                tracking["position_history"] = [position]
            else:
                # 仍在同一区域，更新位置历史
                tracking["position_history"].append(position)
                if len(tracking["position_history"]) > 100:
                    tracking["position_history"] = tracking["position_history"][-100:]
            
            tracking["last_position"] = position
            tracking["last_update"] = current_time
            
            duration = current_time - tracking["start_time"]
        
        # 检查是否超过阈值
        is_loitering = duration >= self.threshold
        
        # 清理过期的跟踪数据（超过阈值2倍时间）
        to_remove = []
        for pid, data in self.tracking_data.items():
            if current_time - data["last_update"] > self.threshold * 2:
                to_remove.append(pid)
        
        for pid in to_remove:
            del self.tracking_data[pid]
        
        return {
            "is_loitering": is_loitering,
            "duration": duration,
            "threshold": self.threshold
        }
    
    def get_status(self) -> Dict:
        """获取跟踪器状态"""
        current_time = time.time()
        loitering_count = 0
        
        for data in self.tracking_data.values():
            duration = current_time - data["start_time"]
            if duration >= self.threshold:
                loitering_count += 1
        
        return {
            "tracking_count": len(self.tracking_data),
            "loitering_count": loitering_count,
            "threshold": self.threshold
        }


class AttemptTracker:
    """尝试次数跟踪器类"""
    
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.attempt_data = {}  # user_id -> {attempt_type: count, last_attempt_time}
    
    def update(self, user_id: str, attempt_type: str, timestamp: datetime) -> Dict:
        """更新尝试跟踪"""
        current_time = time.time()
        
        if user_id not in self.attempt_data:
            self.attempt_data[user_id] = {}
        
        if attempt_type not in self.attempt_data[user_id]:
            self.attempt_data[user_id][attempt_type] = {
                "count": 1,
                "first_attempt": current_time,
                "last_attempt": current_time
            }
        else:
            attempt_info = self.attempt_data[user_id][attempt_type]
            
            # 检查是否在时间窗口内（例如最近5分钟）
            if current_time - attempt_info["last_attempt"] > 300:  # 5分钟
                # 重置计数
                attempt_info["count"] = 1
                attempt_info["first_attempt"] = current_time
            else:
                # 增加计数
                attempt_info["count"] += 1
            
            attempt_info["last_attempt"] = current_time
        
        attempt_info = self.attempt_data[user_id][attempt_type]
        exceeded = attempt_info["count"] >= self.threshold
        
        # 清理过期的尝试数据（超过1小时）
        to_remove_users = []
        for uid, attempts in self.attempt_data.items():
            to_remove_types = []
            for atype, info in attempts.items():
                if current_time - info["last_attempt"] > 3600:  # 1小时
                    to_remove_types.append(atype)
            
            for atype in to_remove_types:
                del attempts[atype]
            
            if not attempts:  # 如果没有尝试类型了，删除用户
                to_remove_users.append(uid)
        
        for uid in to_remove_users:
            del self.attempt_data[uid]
        
        return {
            "exceeded_threshold": exceeded,
            "count": attempt_info["count"],
            "first_attempt": attempt_info["first_attempt"],
            "last_attempt": attempt_info["last_attempt"]
        }
    
    def get_status(self) -> Dict:
        """获取跟踪器状态"""
        total_attempts = 0
        exceeded_count = 0
        
        for user_attempts in self.attempt_data.values():
            for attempt_info in user_attempts.values():
                total_attempts += attempt_info["count"]
                if attempt_info["count"] >= self.threshold:
                    exceeded_count += 1
        
        return {
            "tracking_users": len(self.attempt_data),
            "total_attempts": total_attempts,
            "exceeded_threshold_count": exceeded_count,
            "threshold": self.threshold
        }
