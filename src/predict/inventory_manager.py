#!/usr/bin/env python3
"""
库存管理模块
实现售货机库存的视觉识别和管理
"""

import cv2
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import os

logger = logging.getLogger(__name__)

class InventoryManager:
    """库存管理器类"""
    
    def __init__(self, config: Dict):
        """
        初始化库存管理器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.inventory_config = config.get("inventory", {})
        
        # 库存阈值
        self.low_stock_threshold = self.inventory_config.get("low_stock_threshold", 5)
        self.empty_slot_threshold = self.inventory_config.get("empty_slot_threshold", 0)
        
        # 库存状态
        self.inventory_state = self._load_inventory_state()
        
        # 补货历史
        self.restock_history = []
        
        # 槽位异常状态
        self.slot_anomalies = {}
        
        logger.info("库存管理器初始化完成")
    
    def _load_inventory_state(self) -> Dict:
        """加载库存状态"""
        state_path = "data/inventory_state.json"
        
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载库存状态失败: {e}")
        
        # 返回默认库存状态
        return self._create_default_inventory_state()
    
    def _create_default_inventory_state(self) -> Dict:
        """创建默认库存状态"""
        # 假设有12个槽位（3行4列）
        inventory_state = {
            "last_updated": datetime.now().isoformat(),
            "total_slots": 12,
            "occupied_slots": 10,
            "empty_slots": 2,
            "low_stock_slots": 1,
            "anomaly_slots": 0,
            "slots": {}
        }
        
        # 初始化每个槽位的状态
        for row in range(1, 4):
            for col in range(1, 5):
                slot_id = f"slot_{row}_{col}"
                
                # 随机分配初始库存（5-10个）
                initial_stock = np.random.randint(5, 11)
                
                inventory_state["slots"][slot_id] = {
                    "slot_id": slot_id,
                    "row": row,
                    "col": col,
                    "product_id": f"product_{(row-1)*4 + col}",
                    "product_name": f"商品{(row-1)*4 + col}",
                    "capacity": 15,
                    "current_stock": initial_stock,
                    "last_restock": (datetime.now() - timedelta(days=np.random.randint(1, 7))).isoformat(),
                    "status": "normal",  # normal, low_stock, empty, anomaly
                    "anomaly_type": None,
                    "position": (100 + (col-1)*100, 100 + (row-1)*220),  # 示例位置
                    "size": (80, 200)  # 宽度, 高度
                }
        
        return inventory_state
    
    def check_inventory(self, frame: np.ndarray) -> Dict:
        """
        检查库存状态
        
        Args:
            frame: 视频帧
            
        Returns:
            库存状态报告
        """
        inventory_report = {
            "timestamp": datetime.now().isoformat(),
            "total_slots": 0,
            "occupied_slots": 0,
            "empty_slots": 0,
            "low_stock_slots": 0,
            "anomaly_slots": 0,
            "slot_details": [],
            "alerts": []
        }
        
        try:
            # 1. 检测槽位
            detected_slots = self._detect_slots(frame)
            inventory_report["total_slots"] = len(detected_slots)
            
            # 2. 分析每个槽位
            for slot in detected_slots:
                slot_analysis = self._analyze_slot(frame, slot)
                slot_detail = self._create_slot_detail(slot, slot_analysis)
                inventory_report["slot_details"].append(slot_detail)
                
                # 更新计数
                if slot_analysis["is_empty"]:
                    inventory_report["empty_slots"] += 1
                elif slot_analysis["is_low_stock"]:
                    inventory_report["low_stock_slots"] += 1
                else:
                    inventory_report["occupied_slots"] += 1
                
                if slot_analysis["has_anomaly"]:
                    inventory_report["anomaly_slots"] += 1
                
                # 检查是否需要警报
                alerts = self._check_slot_alerts(slot_detail)
                inventory_report["alerts"].extend(alerts)
            
            # 3. 更新库存状态
            self._update_inventory_state(inventory_report)
            
            # 4. 保存库存状态
            self._save_inventory_state()
            
        except Exception as e:
            logger.error(f"库存检查错误: {e}")
        
        return inventory_report
    
    def _detect_slots(self, frame: np.ndarray) -> List[Dict]:
        """检测售货机槽位"""
        slots = []
        
        # 使用边缘检测找到槽位
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 边缘检测
        edges = cv2.Canny(enhanced, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选可能是槽位的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 槽位通常有特定的面积范围
            if 5000 < area < 50000:
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算宽高比（槽位通常是竖直的矩形）
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.2 < aspect_ratio < 0.8:  # 竖直矩形
                    # 计算轮廓的矩形度
                    rect_area = w * h
                    rectangularity = area / rect_area if rect_area > 0 else 0
                    
                    if rectangularity > 0.6:  # 相对规则的矩形
                        slot = {
                            "bbox": (x, y, w, h),
                            "position": (x + w//2, y + h//2),
                            "size": (w, h),
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                            "rectangularity": rectangularity
                        }
                        slots.append(slot)
        
        # 如果没有检测到槽位，使用配置的槽位
        if not slots:
            slots = self._get_configured_slots(frame)
        
        # 按位置排序（从上到下，从左到右）
        slots.sort(key=lambda s: (s["position"][1], s["position"][0]))
        
        # 分配槽位ID
        for i, slot in enumerate(slots):
            row = i // 4 + 1  # 假设每行4个槽位
            col = i % 4 + 1
            slot["slot_id"] = f"slot_{row}_{col}"
            slot["row"] = row
            slot["col"] = col
        
        return slots
    
    def _get_configured_slots(self, frame: np.ndarray) -> List[Dict]:
        """获取配置的槽位"""
        slots = []
        
        # 从库存状态获取槽位配置
        for slot_id, slot_info in self.inventory_state.get("slots", {}).items():
            if "position" in slot_info and "size" in slot_info:
                x, y = slot_info["position"]
                w, h = slot_info["size"]
                
                slot = {
                    "slot_id": slot_id,
                    "row": slot_info.get("row", 1),
                    "col": slot_info.get("col", 1),
                    "bbox": (x, y, w, h),
                    "position": (x + w//2, y + h//2),
                    "size": (w, h),
                    "area": w * h,
                    "aspect_ratio": w / h if h > 0 else 0,
                    "rectangularity": 0.8  # 假设规则
                }
                slots.append(slot)
        
        return slots
    
    def _analyze_slot(self, frame: np.ndarray, slot: Dict) -> Dict:
        """分析槽位状态"""
        x, y, w, h = slot["bbox"]
        
        # 提取槽位区域
        slot_roi = frame[y:y+h, x:x+w]
        
        if slot_roi.size == 0:
            return {
                "is_empty": True,
                "is_low_stock": False,
                "has_anomaly": False,
                "estimated_stock": 0,
                "anomaly_type": None,
                "confidence": 1.0
            }
        
        # 分析槽位内容
        analysis = self._analyze_slot_content(slot_roi)
        
        # 检查异常
        anomaly_detection = self._detect_slot_anomaly(slot_roi)
        
        # 估算库存
        estimated_stock = self._estimate_stock(slot_roi, analysis)
        
        # 判断是否为空
        is_empty = estimated_stock <= self.empty_slot_threshold
        
        # 判断是否为低库存
        slot_capacity = self._get_slot_capacity(slot["slot_id"])
        is_low_stock = not is_empty and estimated_stock <= self.low_stock_threshold
        
        return {
            "is_empty": is_empty,
            "is_low_stock": is_low_stock,
            "has_anomaly": anomaly_detection["has_anomaly"],
            "estimated_stock": estimated_stock,
            "anomaly_type": anomaly_detection["anomaly_type"],
            "confidence": analysis.get("confidence", 0.7),
            "content_analysis": analysis,
            "anomaly_details": anomaly_detection
        }
    
    def _analyze_slot_content(self, slot_roi: np.ndarray) -> Dict:
        """分析槽位内容"""
        if slot_roi.size == 0:
            return {"confidence": 0.0, "content_type": "empty"}
        
        # 转换为灰度图
        gray = cv2.cvtColor(slot_roi, cv2.COLOR_BGR2GRAY)
        
        # 计算平均亮度
        avg_brightness = np.mean(gray)
        
        # 计算对比度
        contrast = np.std(gray)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析轮廓
        contour_count = len(contours)
        contour_areas = [cv2.contourArea(c) for c in contours]
        total_contour_area = sum(contour_areas)
        
        # 计算填充率
        fill_ratio = total_contour_area / (slot_roi.shape[0] * slot_roi.shape[1])
        
        # 判断内容类型
        if fill_ratio < 0.1:
            content_type = "empty"
            confidence = 1.0 - fill_ratio
        elif fill_ratio < 0.3:
            content_type = "low_stock"
            confidence = 0.7
        else:
            content_type = "normal"
            confidence = 0.8
        
        return {
            "confidence": confidence,
            "content_type": content_type,
            "avg_brightness": avg_brightness,
            "contrast": contrast,
            "edge_density": edge_density,
            "contour_count": contour_count,
            "total_contour_area": total_contour_area,
            "fill_ratio": fill_ratio,
            "max_contour_area": max(contour_areas) if contour_areas else 0
        }
    
    def _detect_slot_anomaly(self, slot_roi: np.ndarray) -> Dict:
        """检测槽位异常"""
        anomaly_result = {
            "has_anomaly": False,
            "anomaly_type": None,
            "confidence": 0.0,
            "details": {}
        }
        
        if slot_roi.size == 0:
            return anomaly_result
        
        # 1. 检测堵塞（商品卡住）
        jam_detected = self._detect_jam(slot_roi)
        if jam_detected["detected"]:
            anomaly_result["has_anomaly"] = True
            anomaly_result["anomaly_type"] = "jam"
            anomaly_result["confidence"] = jam_detected["confidence"]
            anomaly_result["details"]["jam"] = jam_detected
        
        # 2. 检测倾斜
        tilt_detected = self._detect_tilt(slot_roi)
        if tilt_detected["detected"]:
            anomaly_result["has_anomaly"] = True
            anomaly_result["anomaly_type"] = "tilt"
            anomaly_result["confidence"] = max(anomaly_result["confidence"], tilt_detected["confidence"])
            anomaly_result["details"]["tilt"] = tilt_detected
        
        # 3. 检测异物
        foreign_object = self._detect_foreign_object(slot_roi)
        if foreign_object["detected"]:
            anomaly_result["has_anomaly"] = True
            anomaly_result["anomaly_type"] = "foreign_object"
            anomaly_result["confidence"] = max(anomaly_result["confidence"], foreign_object["confidence"])
            anomaly_result["details"]["foreign_object"] = foreign_object
        
        return anomaly_result
    
    def _detect_jam(self, slot_roi: np.ndarray) -> Dict:
        """检测商品堵塞"""
        gray = cv2.cvtColor(slot_roi, cv2.COLOR_BGR2GRAY)
        
        # 使用Hough变换检测直线
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        jam_detected = False
        confidence = 0.0
        
        if lines is not None:
            # 检查是否有水平线（堵塞的典型特征）
            horizontal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # 接近水平的线
                if 0 < angle < 15 or 165 < angle < 180:
                    horizontal_lines += 1
            
            # 如果有多条水平线，可能是堵塞
            if horizontal_lines >= 3:
                jam_detected = True
                confidence = min(horizontal_lines / 10.0, 1.0)
        
        return {
            "detected": jam_detected,
            "confidence": confidence,
            "line_count": len(lines) if lines is not None else 0,
            "horizontal_lines": horizontal_lines if 'horizontal_lines' in locals() else 0
        }
    
    def _detect_tilt(self, slot_roi: np.ndarray) -> Dict:
        """检测商品倾斜"""
        gray = cv2.cvtColor(slot_roi, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"detected": False, "confidence": 0.0}
        
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(max_contour)
        angle = rect[2]  # 旋转角度
        
        # 检查角度是否异常（不是接近0或90度）
        tilt_detected = False
        confidence = 0.0
        
        if abs(angle) > 10 and abs(angle) < 80:
            tilt_detected = True
            confidence = min(abs(angle) / 45.0, 1.0)
        
        return {
            "detected": tilt_detected,
            "confidence": confidence,
            "tilt_angle": angle,
            "contour_area": cv2.contourArea(max_contour)
        }
    
    def _detect_foreign_object(self, slot_roi: np.ndarray) -> Dict:
        """检测异物"""
        # 使用颜色异常检测
        hsv = cv2.cvtColor(slot_roi, cv2.COLOR_BGR2HSV)
        
        # 计算颜色直方图
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # 计算颜色分布的熵（异常颜色会有不同的分布）
        hist_h_normalized = hist_h / (np.sum(hist_h) + 1e-7)
        hist_s_normalized = hist_s / (np.sum(hist_s) + 1e-7)
        
        entropy_h = -np.sum(hist_h_normalized * np.log2(hist_h_normalized + 1e-7))
        entropy_s = -np.sum(hist_s_normalized * np.log2(hist_s_normalized + 1e-7))
        
        # 高熵值可能表示颜色异常
        foreign_detected = entropy_h > 5.0 or entropy_s > 4.0
        confidence = min(max(entropy_h / 8.0, entropy_s / 6.0), 1.0)
        
        return {
            "detected": foreign_detected,
            "confidence": confidence,
            "entropy_h": float(entropy_h),
            "entropy_s": float(entropy_s)
        }
    
    def _estimate_stock(self, slot_roi: np.ndarray, analysis: Dict) -> int:
        """估算库存数量"""
        if analysis.get("content_type") == "empty":
            return 0
        
        # 基于填充率估算
        fill_ratio = analysis.get("fill_ratio", 0)
        contour_count = analysis.get("contour_count", 0)
        
        # 简单估算：基于填充率和轮廓数量
        if fill_ratio < 0.2:
            estimated = max(1, int(contour_count * 0.5))
        elif fill_ratio < 0.4:
            estimated = max(3, int(contour_count * 0.7))
        elif fill_ratio < 0.6:
            estimated = max(5, int(contour_count * 0.9))
        else:
            estimated = max(7, int(contour_count * 1.1))
        
        # 考虑槽位容量
        slot_capacity = 15  # 默认容量
        estimated = min(estimated, slot_capacity)
        
        return estimated
    
    def _get_slot_capacity(self, slot_id: str) -> int:
        """获取槽位容量"""
        slot_info = self.inventory_state.get("slots", {}).get(slot_id, {})
        return slot_info.get("capacity", 15)
    
    def _create_slot_detail(self, slot: Dict, analysis: Dict) -> Dict:
        """创建槽位详细信息"""
        slot_detail = {
            "slot_id": slot["slot_id"],
            "row": slot["row"],
            "col": slot["col"],
            "position": slot["position"],
            "size": slot["size"],
            "is_empty": analysis["is_empty"],
            "is_low_stock": analysis["is_low_stock"],
            "has_anomaly": analysis["has_anomaly"],
            "estimated_stock": analysis["estimated_stock"],
            "anomaly_type": analysis["anomaly_type"],
            "confidence": analysis["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加产品信息（如果可用）
        slot_info = self.inventory_state.get("slots", {}).get(slot["slot_id"], {})
        if slot_info:
            slot_detail["product_id"] = slot_info.get("product_id")
            slot_detail["product_name"] = slot_info.get("product_name")
            slot_detail["capacity"] = slot_info.get("capacity", 15)
            slot_detail["last_restock"] = slot_info.get("last_restock")
        
        return slot_detail
    
    def _check_slot_alerts(self, slot_detail: Dict) -> List[Dict]:
        """检查槽位警报"""
        alerts = []
        
        # 空槽位警报
        if slot_detail["is_empty"]:
            alert = {
                "type": "empty_slot",
                "severity": "medium",
                "slot_id": slot_detail["slot_id"],
                "product_name": slot_detail.get("product_name", "未知商品"),
                "description": f"槽位 {slot_detail['slot_id']} 已空",
                "timestamp": slot_detail["timestamp"]
            }
            alerts.append(alert)
        
        # 低库存警报
        elif slot_detail["is_low_stock"]:
            alert = {
                "type": "low_stock",
                "severity": "low",
                "slot_id": slot_detail["slot_id"],
                "product_name": slot_detail.get("product_name", "未知商品"),
                "current_stock": slot_detail["estimated_stock"],
                "threshold": self.low_stock_threshold,
                "description": f"槽位 {slot_detail['slot_id']} 库存低 ({slot_detail['estimated_stock']}个)",
                "timestamp": slot_detail["timestamp"]
            }
            alerts.append(alert)
        
        # 异常警报
        if slot_detail["has_anomaly"]:
            alert = {
                "type": "slot_anomaly",
                "severity": "high",
                "slot_id": slot_detail["slot_id"],
                "anomaly_type": slot_detail["anomaly_type"],
                "product_name": slot_detail.get("product_name", "未知商品"),
                "description": f"槽位 {slot_detail['slot_id']} 检测到{slot_detail['anomaly_type']}异常",
                "timestamp": slot_detail["timestamp"]
            }
            alerts.append(alert)
        
        return alerts
    
    def _update_inventory_state(self, report: Dict):
        """更新库存状态"""
        self.inventory_state["last_updated"] = datetime.now().isoformat()
        self.inventory_state["total_slots"] = report["total_slots"]
        self.inventory_state["occupied_slots"] = report["occupied_slots"]
        self.inventory_state["empty_slots"] = report["empty_slots"]
        self.inventory_state["low_stock_slots"] = report["low_stock_slots"]
        self.inventory_state["anomaly_slots"] = report["anomaly_slots"]
        
        # 更新每个槽位的状态
        for slot_detail in report["slot_details"]:
            slot_id = slot_detail["slot_id"]
            
            if slot_id not in self.inventory_state["slots"]:
                self.inventory_state["slots"][slot_id] = {
                    "slot_id": slot_id,
                    "row": slot_detail["row"],
                    "col": slot_detail["col"],
                    "product_id": f"product_{slot_id}",
                    "product_name": f"商品{slot_id}",
                    "capacity": 15,
                    "position": slot_detail["position"],
                    "size": slot_detail["size"]
                }
            
            slot_state = self.inventory_state["slots"][slot_id]
            slot_state["current_stock"] = slot_detail["estimated_stock"]
            
            # 更新状态
            if slot_detail["is_empty"]:
                slot_state["status"] = "empty"
            elif slot_detail["is_low_stock"]:
                slot_state["status"] = "low_stock"
            elif slot_detail["has_anomaly"]:
                slot_state["status"] = "anomaly"
                slot_state["anomaly_type"] = slot_detail["anomaly_type"]
            else:
                slot_state["status"] = "normal"
                slot_state["anomaly_type"] = None
            
            slot_state["last_updated"] = slot_detail["timestamp"]
    
    def _save_inventory_state(self):
        """保存库存状态"""
        try:
            os.makedirs("data", exist_ok=True)
            state_path = "data/inventory_state.json"
            
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(self.inventory_state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"库存状态已保存到 {state_path}")
        except Exception as e:
            logger.error(f"保存库存状态失败: {e}")
    
    def check_low_stock(self, inventory_data: Dict) -> List[Dict]:
        """检查低库存商品"""
        low_stock_items = []
        
        for slot_detail in inventory_data.get("slot_details", []):
            if slot_detail.get("is_low_stock", False):
                item = {
                    "slot_id": slot_detail["slot_id"],
                    "product_name": slot_detail.get("product_name", "未知商品"),
                    "current_stock": slot_detail.get("estimated_stock", 0),
                    "threshold": self.low_stock_threshold,
                    "position": slot_detail.get("position"),
                    "timestamp": slot_detail.get("timestamp")
                }
                low_stock_items.append(item)
        
        return low_stock_items
    
    def record_restock(self, slot_id: str, restock_amount: int, restocker: str = "系统"):
        """记录补货操作"""
        restock_record = {
            "slot_id": slot_id,
            "restock_amount": restock_amount,
            "restocker": restocker,
            "timestamp": datetime.now().isoformat(),
            "previous_stock": self.inventory_state["slots"].get(slot_id, {}).get("current_stock", 0)
        }
        
        self.restock_history.append(restock_record)
        
        # 更新库存状态
        if slot_id in self.inventory_state["slots"]:
            slot_state = self.inventory_state["slots"][slot_id]
            slot_state["current_stock"] = min(
                slot_state.get("current_stock", 0) + restock_amount,
                slot_state.get("capacity", 15)
            )
            slot_state["last_restock"] = restock_record["timestamp"]
            slot_state["status"] = "normal"
            slot_state["anomaly_type"] = None
        
        # 保存更新
        self._save_inventory_state()
        
        logger.info(f"记录补货: 槽位 {slot_id}, 数量 {restock_amount}, 操作员 {restocker}")
        
        return restock_record
    
    def get_inventory_summary(self) -> Dict:
        """获取库存摘要"""
        total_capacity = 0
        total_stock = 0
        empty_slots = 0
        low_stock_slots = 0
        anomaly_slots = 0
        
        for slot_id, slot_state in self.inventory_state.get("slots", {}).items():
            total_capacity += slot_state.get("capacity", 0)
            total_stock += slot_state.get("current_stock", 0)
            
            status = slot_state.get("status", "normal")
            if status == "empty":
                empty_slots += 1
            elif status == "low_stock":
                low_stock_slots += 1
            elif status == "anomaly":
                anomaly_slots += 1
        
        utilization = total_stock / total_capacity if total_capacity > 0 else 0
        
        return {
            "total_slots": len(self.inventory_state.get("slots", {})),
            "total_capacity": total_capacity,
            "total_stock": total_stock,
            "utilization_rate": utilization,
            "empty_slots": empty_slots,
            "low_stock_slots": low_stock_slots,
            "anomaly_slots": anomaly_slots,
            "last_updated": self.inventory_state.get("last_updated"),
            "restock_count": len(self.restock_history)
        }
    
    def get_status(self) -> Dict:
        """获取管理器状态"""
        summary = self.get_inventory_summary()
        
        return {
            "inventory_summary": summary,
            "thresholds": {
                "low_stock": self.low_stock_threshold,
                "empty_slot": self.empty_slot_threshold
            },
            "slot_count": len(self.inventory_state.get("slots", {})),
            "restock_history_count": len(self.restock_history)
        }
    
    def visualize_inventory(self, frame: np.ndarray, report: Dict) -> np.ndarray:
        """可视化库存状态"""
        vis_frame = frame.copy()
        
        # 颜色定义
        colors = {
            "normal": (0, 255, 0),    # 绿色
            "low_stock": (0, 255, 255),  # 黄色
            "empty": (0, 0, 255),      # 红色
            "anomaly": (255, 0, 0)     # 蓝色
        }
        
        # 绘制每个槽位
        for slot_detail in report.get("slot_details", []):
            if "position" in slot_detail and "size" in slot_detail:
                x, y = slot_detail["position"]
                w, h = slot_detail["size"]
                
                # 确定颜色
                if slot_detail["has_anomaly"]:
                    color = colors["anomaly"]
                elif slot_detail["is_empty"]:
                    color = colors["empty"]
                elif slot_detail["is_low_stock"]:
                    color = colors["low_stock"]
                else:
                    color = colors["normal"]
                
                # 绘制矩形
                cv2.rectangle(vis_frame, (x - w//2, y - h//2), 
                             (x + w//2, y + h//2), color, 2)
                
                # 添加标签
                label = f"{slot_detail.get('slot_id', '')}: {slot_detail.get('estimated_stock', 0)}"
                cv2.putText(vis_frame, label, (x - w//2, y - h//2 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 添加摘要信息
        summary_text = f"库存: {report.get('occupied_slots', 0)}/{report.get('total_slots', 0)}"
        cv2.putText(vis_frame, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        alert_count = len(report.get("alerts", []))
        if alert_count > 0:
            alert_text = f"警报: {alert_count}"
            cv2.putText(vis_frame, alert_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame
