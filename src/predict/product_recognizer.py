#!/usr/bin/env python3
"""
商品识别模块
实现售货机中商品的视觉识别
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import os
import json

logger = logging.getLogger(__name__)

class ProductRecognizer:
    """商品识别器类"""
    
    def __init__(self, config: Dict):
        """
        初始化商品识别器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.min_confidence = config.get("analysis", {}).get("min_confidence", 0.7)
        
        # 商品数据库
        self.product_database = self._load_product_database()
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 商品槽位配置
        self.slot_configurations = self._load_slot_configurations()
        
        # 识别历史
        self.recognition_history = []
        
        logger.info(f"商品识别器初始化完成，加载了 {len(self.product_database)} 种商品")
    
    def _load_product_database(self) -> Dict:
        """加载商品数据库"""
        database_path = "data/products_database.json"
        
        if os.path.exists(database_path):
            try:
                with open(database_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载商品数据库失败: {e}")
        
        # 返回默认数据库
        return self._create_default_database()
    
    def _create_default_database(self) -> Dict:
        """创建默认商品数据库"""
        return {
            "cola": {
                "name": "可口可乐",
                "category": "饮料",
                "price": 3.0,
                "features": {
                    "color": "red",
                    "shape": "cylindrical",
                    "logo": "coca_cola",
                    "size": (65, 180)  # 直径x高度(mm)
                },
                "image_paths": [],
                "template_features": None
            },
            "sprite": {
                "name": "雪碧",
                "category": "饮料",
                "price": 3.0,
                "features": {
                    "color": "green",
                    "shape": "cylindrical",
                    "logo": "sprite",
                    "size": (65, 180)
                },
                "image_paths": [],
                "template_features": None
            },
            "water": {
                "name": "矿泉水",
                "category": "饮料",
                "price": 2.0,
                "features": {
                    "color": "transparent",
                    "shape": "cylindrical",
                    "logo": "none",
                    "size": (65, 180)
                },
                "image_paths": [],
                "template_features": None
            },
            "chips": {
                "name": "薯片",
                "category": "零食",
                "price": 5.0,
                "features": {
                    "color": "yellow",
                    "shape": "bag",
                    "logo": "lays",
                    "size": (150, 250, 30)  # 长x宽x厚(mm)
                },
                "image_paths": [],
                "template_features": None
            },
            "chocolate": {
                "name": "巧克力",
                "category": "零食",
                "price": 4.0,
                "features": {
                    "color": "brown",
                    "shape": "rectangular",
                    "logo": "dove",
                    "size": (80, 120, 10)
                },
                "image_paths": [],
                "template_features": None
            }
        }
    
    def _load_slot_configurations(self) -> List[Dict]:
        """加载商品槽位配置"""
        # 默认槽位配置（假设售货机有3行4列）
        slots = []
        
        # 槽位位置（在图像中的相对位置）
        slot_positions = [
            # 第一行
            {"row": 1, "col": 1, "position": (100, 100), "size": (80, 200)},
            {"row": 1, "col": 2, "position": (200, 100), "size": (80, 200)},
            {"row": 1, "col": 3, "position": (300, 100), "size": (80, 200)},
            {"row": 1, "col": 4, "position": (400, 100), "size": (80, 200)},
            
            # 第二行
            {"row": 2, "col": 1, "position": (100, 320), "size": (80, 200)},
            {"row": 2, "col": 2, "position": (200, 320), "size": (80, 200)},
            {"row": 2, "col": 3, "position": (300, 320), "size": (80, 200)},
            {"row": 2, "col": 4, "position": (400, 320), "size": (80, 200)},
            
            # 第三行
            {"row": 3, "col": 1, "position": (100, 540), "size": (80, 200)},
            {"row": 3, "col": 2, "position": (200, 540), "size": (80, 200)},
            {"row": 3, "col": 3, "position": (300, 540), "size": (80, 200)},
            {"row": 3, "col": 4, "position": (400, 540), "size": (80, 200)},
        ]
        
        # 为每个槽位分配商品（示例）
        products = list(self.product_database.keys())
        for i, pos in enumerate(slot_positions):
            product_id = products[i % len(products)] if products else "unknown"
            slot = {
                "slot_id": f"slot_{pos['row']}_{pos['col']}",
                "row": pos["row"],
                "col": pos["col"],
                "position": pos["position"],  # (x, y)
                "size": pos["size"],  # (width, height)
                "assigned_product": product_id,
                "current_product": product_id,
                "capacity": 10,  # 最大容量
                "current_stock": 8,  # 当前库存
                "last_restock": time.time() - 86400  # 24小时前
            }
            slots.append(slot)
        
        return slots
    
    def recognize(self, frame: np.ndarray) -> List[Dict]:
        """
        识别视频帧中的商品
        
        Args:
            frame: 视频帧
            
        Returns:
            商品识别结果列表
        """
        recognition_results = []
        
        try:
            # 1. 检测商品槽位
            detected_slots = self._detect_product_slots(frame)
            
            # 2. 对每个槽位进行商品识别
            for slot in detected_slots:
                slot_result = self._recognize_slot_content(frame, slot)
                if slot_result:
                    recognition_results.append(slot_result)
            
            # 3. 检测用户手中的商品
            hand_products = self._detect_hand_products(frame)
            recognition_results.extend(hand_products)
            
            # 4. 记录识别历史
            if recognition_results:
                self._update_recognition_history(recognition_results)
            
        except Exception as e:
            logger.error(f"商品识别错误: {e}")
        
        return recognition_results
    
    def _detect_product_slots(self, frame: np.ndarray) -> List[Dict]:
        """检测商品槽位"""
        detected_slots = []
        
        # 使用槽位配置作为参考
        for slot_config in self.slot_configurations:
            x, y = slot_config["position"]
            w, h = slot_config["size"]
            
            # 检查槽位区域是否在图像范围内
            height, width = frame.shape[:2]
            if x < width and y < height:
                # 提取槽位区域
                slot_roi = frame[y:y+h, x:x+w]
                
                if slot_roi.size > 0:
                    # 分析槽位内容
                    slot_analysis = self._analyze_slot_content(slot_roi)
                    
                    detected_slot = {
                        "slot_id": slot_config["slot_id"],
                        "position": (x, y),
                        "size": (w, h),
                        "assigned_product": slot_config["assigned_product"],
                        "content_analysis": slot_analysis,
                        "is_empty": slot_analysis.get("is_empty", False),
                        "product_count": slot_analysis.get("product_count", 0)
                    }
                    
                    detected_slots.append(detected_slot)
        
        return detected_slots
    
    def _analyze_slot_content(self, slot_roi: np.ndarray) -> Dict:
        """分析槽位内容"""
        if slot_roi.size == 0:
            return {"is_empty": True, "product_count": 0}
        
        # 转换为灰度图
        gray = cv2.cvtColor(slot_roi, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析轮廓
        product_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 忽略小面积噪声
                product_contours.append(contour)
        
        # 判断是否为空
        is_empty = len(product_contours) == 0
        
        # 估算商品数量（基于轮廓数量和面积）
        product_count = len(product_contours)
        
        # 计算平均颜色
        avg_color = np.mean(slot_roi, axis=(0, 1))
        
        # 计算纹理特征
        texture_score = self._calculate_texture_score(gray)
        
        return {
            "is_empty": is_empty,
            "product_count": product_count,
            "contour_count": len(product_contours),
            "avg_color": avg_color.tolist(),
            "texture_score": texture_score,
            "slot_area": slot_roi.shape[0] * slot_roi.shape[1]
        }
    
    def _calculate_texture_score(self, gray_image: np.ndarray) -> float:
        """计算纹理分数"""
        if gray_image.size == 0:
            return 0.0
        
        # 使用LBP（局部二值模式）计算纹理
        try:
            # 计算LBP
            radius = 1
            n_points = 8 * radius
            lbp = np.zeros_like(gray_image, dtype=np.uint8)
            
            for i in range(gray_image.shape[0] - 2*radius):
                for j in range(gray_image.shape[1] - 2*radius):
                    center = gray_image[i+radius, j+radius]
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(j + radius * np.cos(angle))
                        y = int(i + radius * np.sin(angle))
                        if gray_image[y, x] >= center:
                            code |= 1 << (n_points - k - 1)
                    lbp[i+radius, j+radius] = code
            
            # 计算LBP直方图
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)  # 归一化
            
            # 计算纹理分数（熵）
            texture_score = -np.sum(hist * np.log2(hist + 1e-7))
            
            return texture_score
        except Exception:
            return 0.0
    
    def _recognize_slot_content(self, frame: np.ndarray, slot: Dict) -> Optional[Dict]:
        """识别槽位中的具体商品"""
        x, y = slot["position"]
        w, h = slot["size"]
        
        # 提取槽位区域
        slot_roi = frame[y:y+h, x:x+w]
        
        if slot_roi.size == 0 or slot["is_empty"]:
            return None
        
        # 提取特征
        features = self.feature_extractor.extract(slot_roi)
        
        # 与数据库中的商品进行匹配
        best_match = None
        best_score = 0.0
        
        for product_id, product_info in self.product_database.items():
            # 获取商品特征模板
            template_features = product_info.get("template_features")
            
            if template_features is None:
                # 如果没有模板特征，使用基于属性的匹配
                match_score = self._match_by_attributes(features, product_info)
            else:
                # 使用特征匹配
                match_score = self._match_features(features, template_features)
            
            # 考虑槽位分配的商品
            if product_id == slot["assigned_product"]:
                match_score *= 1.2  # 增加分配商品的权重
            
            if match_score > best_score and match_score >= self.min_confidence:
                best_score = match_score
                best_match = product_id
        
        if best_match:
            product_info = self.product_database[best_match]
            
            # 估算库存数量
            estimated_stock = self._estimate_stock_count(slot_roi, product_info)
            
            result = {
                "slot_id": slot["slot_id"],
                "product_id": best_match,
                "product_name": product_info["name"],
                "category": product_info["category"],
                "price": product_info["price"],
                "confidence": best_score,
                "position": slot["position"],
                "estimated_stock": estimated_stock,
                "slot_size": slot["size"],
                "timestamp": datetime.now()
            }
            
            return result
        
        return None
    
    def _match_by_attributes(self, features: Dict, product_info: Dict) -> float:
        """基于属性匹配商品"""
        match_score = 0.0
        total_weight = 0.0
        
        product_features = product_info.get("features", {})
        
        # 颜色匹配
        if "color" in features and "color" in product_features:
            color_similarity = self._compare_colors(features["color"], product_features["color"])
            match_score += color_similarity * 0.3
            total_weight += 0.3
        
        # 形状匹配
        if "shape" in features and "shape" in product_features:
            shape_similarity = self._compare_shapes(features["shape"], product_features["shape"])
            match_score += shape_similarity * 0.3
            total_weight += 0.3
        
        # 纹理匹配
        if "texture" in features:
            texture_similarity = 1.0 - abs(features["texture"] - 0.5)  # 简化处理
            match_score += texture_similarity * 0.2
            total_weight += 0.2
        
        # 大小匹配
        if "size" in features and "size" in product_features:
            size_similarity = self._compare_sizes(features["size"], product_features["size"])
            match_score += size_similarity * 0.2
            total_weight += 0.2
        
        # 归一化分数
        if total_weight > 0:
            match_score /= total_weight
        
        return match_score
    
    def _match_features(self, features1: Dict, features2: Dict) -> float:
        """匹配特征向量"""
        # 这里应该实现具体的特征匹配算法
        # 简化实现：计算特征向量的余弦相似度
        
        if "feature_vector" not in features1 or "feature_vector" not in features2:
            return 0.0
        
        vec1 = np.array(features1["feature_vector"])
        vec2 = np.array(features2["feature_vector"])
        
        if vec1.shape != vec2.shape:
            return 0.0
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def _compare_colors(self, color1, color2) -> float:
        """比较颜色相似度"""
        # 简化实现
        if color1 == color2:
            return 1.0
        
        # 颜色相似度表
        color_similarities = {
            ("red", "dark_red"): 0.8,
            ("green", "dark_green"): 0.8,
            ("blue", "dark_blue"): 0.8,
            ("yellow", "orange"): 0.7,
            ("brown", "dark_brown"): 0.9,
        }
        
        for (c1, c2), similarity in color_similarities.items():
            if (color1 == c1 and color2 == c2) or (color1 == c2 and color2 == c1):
                return similarity
        
        return 0.3  # 默认相似度
    
    def _compare_shapes(self, shape1, shape2) -> float:
        """比较形状相似度"""
        if shape1 == shape2:
            return 1.0
        
        # 形状相似度表
        shape_similarities = {
            ("cylindrical", "bottle"): 0.8,
            ("rectangular", "square"): 0.7,
            ("bag", "pouch"): 0.9,
        }
        
        for (s1, s2), similarity in shape_similarities.items():
            if (shape1 == s1 and shape2 == s2) or (shape1 == s2 and shape2 == s1):
                return similarity
        
        return 0.3  # 默认相似度
    
    def _compare_sizes(self, size1, size2) -> float:
        """比较大小相似度"""
        # 简化实现
        if size1 == size2:
            return 1.0
        
        # 如果都是元组/列表，计算相对差异
        if isinstance(size1, (tuple, list)) and isinstance(size2, (tuple, list)):
            if len(size1) != len(size2):
                return 0.3
            
            # 计算平均相对差异
            diffs = []
            for s1, s2 in zip(size1, size2):
                if s2 == 0:
                    diffs.append(1.0)
                else:
                    diffs.append(abs(s1 - s2) / max(s1, s2))
            
            avg_diff = np.mean(diffs)
            similarity = 1.0 - min(avg_diff, 1.0)
            return similarity
        
        return 0.3
    
    def _estimate_stock_count(self, slot_roi: np.ndarray, product_info: Dict) -> int:
        """估算库存数量"""
        # 基于槽位区域分析和商品大小估算
        
        slot_area = slot_roi.shape[0] * slot_roi.shape[1]
        
        # 获取商品大小信息
        product_size = product_info.get("features", {}).get("size", (100, 100))
        
        if isinstance(product_size, (tuple, list)):
            if len(product_size) >= 2:
                # 估算商品投影面积
                if len(product_size) == 2:
                    product_area = product_size[0] * product_size[1]
                else:
                    product_area = product_size[0] * product_size[1]
            else:
                product_area = 10000  # 默认面积
        else:
            product_area = 10000
        
        # 估算最大容量
        max_capacity = int(slot_area / product_area * 0.7)  # 考虑包装和空隙
        
        # 基于轮廓分析估算当前库存
        gray = cv2.cvtColor(slot_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算有效轮廓数量
        valid_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > product_area * 0.3:  # 面积大于商品面积的30%
                valid_contours += 1
        
        estimated_stock = min(valid_contours, max_capacity)
        
        return estimated_stock
    
    def _detect_hand_products(self, frame: np.ndarray) -> List[Dict]:
        """检测用户手中的商品"""
        hand_products = []
        
        # 使用肤色检测找到手部区域
        skin_mask = self._detect_skin(frame)
        
        # 查找手部轮廓
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # 手部区域大小范围
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 提取手部区域
                hand_roi = frame[y:y+h, x:x+w]
                
                if hand_roi.size > 0:
                    # 检测手中是否有商品
                    product_result = self._detect_product_in_hand(hand_roi)
                    if product_result:
                        product_result["hand_position"] = (x + w//2, y + h//2)
                        product_result["hand_bbox"] = (x, y, w, h)
                        hand_products.append(product_result)
        
        return hand_products
    
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
    
    def _detect_product_in_hand(self, hand_roi: np.ndarray) -> Optional[Dict]:
        """检测手中的商品"""
        # 提取特征
        features = self.feature_extractor.extract(hand_roi)
        
        # 与数据库中的商品进行匹配
        best_match = None
        best_score = 0.0
        
        for product_id, product_info in self.product_database.items():
            # 使用基于属性的匹配
            match_score = self._match_by_attributes(features, product_info)
            
            if match_score > best_score and match_score >= self.min_confidence:
                best_score = match_score
                best_match = product_id
        
        if best_match:
            product_info = self.product_database[best_match]
            
            result = {
                "product_id": best_match,
                "product_name": product_info["name"],
                "category": product_info["category"],
                "price": product_info["price"],
                "confidence": best_score,
                "in_hand": True,
                "timestamp": datetime.now()
            }
            
            return result
        
        return None
    
    def _update_recognition_history(self, results: List[Dict]):
        """更新识别历史"""
        for result in results:
            self.recognition_history.append({
                "timestamp": datetime.now(),
                "result": result
            })
        
        # 保留最近1000条记录
        if len(self.recognition_history) > 1000:
            self.recognition_history = self.recognition_history[-1000:]
    
    def get_status(self) -> Dict:
        """获取识别器状态"""
        recent_results = [r for r in self.recognition_history 
                         if (datetime.now() - r["timestamp"]).total_seconds() < 3600]
        
        return {
            "product_count": len(self.product_database),
            "slot_count": len(self.slot_configurations),
            "recent_recognitions": len(recent_results),
            "average_confidence": np.mean([r["result"].get("confidence", 0) 
                                          for r in recent_results]) if recent_results else 0.0
        }
    
    def visualize_products(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """可视化商品识别结果"""
        vis_frame = frame.copy()
        
        for result in results:
            # 绘制槽位边界
            if "position" in result and "slot_size" in result:
                x, y = result["position"]
                w, h = result["slot_size"]
                
                # 绘制矩形
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加标签
                label = f"{result.get('product_name', 'Unknown')}: {result.get('estimated_stock', 0)}"
                cv2.putText(vis_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制手中的商品
            if result.get("in_hand", False) and "hand_position" in result:
                x, y = result["hand_position"]
                cv2.circle(vis_frame, (x, y), 15, (255, 0, 0), 3)
                
                label = f"Hand: {result.get('product_name', 'Unknown')}"
                cv2.putText(vis_frame, label, (x - 50, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return vis_frame


class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
    
    def extract(self, image: np.ndarray) -> Dict:
        """从图像中提取特征"""
        features = {}
        
        if image.size == 0:
            return features
        
        # 颜色特征
        features["color"] = self._extract_color_features(image)
        
        # 纹理特征
        features["texture"] = self._extract_texture_features(image)
        
        # 形状特征
        features["shape"] = self._extract_shape_features(image)
        
        # 大小特征
        features["size"] = image.shape[:2]  # (height, width)
        
        # SIFT特征
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is not None:
            # 使用PCA降维或特征聚合
            features["feature_vector"] = self._aggregate_descriptors(descriptors)
        
        return features
    
    def _extract_color_features(self, image: np.ndarray) -> str:
        """提取颜色特征"""
        # 计算平均颜色
        avg_color = np.mean(image, axis=(0, 1))
        
        # 转换为HSV
        avg_color_bgr = np.uint8([[avg_color]])
        avg_color_hsv = cv2.cvtColor(avg_color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # 根据HSV值判断颜色
        h, s, v = avg_color_hsv
        
        if s < 50:  # 低饱和度
            if v > 200:
                return "white"
            elif v < 50:
                return "black"
            else:
                return "gray"
        else:
            if h < 15 or h > 165:
                return "red"
            elif h < 45:
                return "orange"
            elif h < 75:
                return "yellow"
            elif h < 105:
                return "green"
            elif h < 135:
                return "cyan"
            elif h < 165:
                return "blue"
            else:
                return "purple"
    
    def _extract_texture_features(self, image: np.ndarray) -> float:
        """提取纹理特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用LBP计算纹理
        radius = 1
        n_points = 8 * radius
        lbp = np.zeros_like(gray, dtype=np.uint8)
        
        for i in range(gray.shape[0] - 2*radius):
            for j in range(gray.shape[1] - 2*radius):
                center = gray[i+radius, j+radius]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(j + radius * np.cos(angle))
                    y = int(i + radius * np.sin(angle))
                    if gray[y, x] >= center:
                        code |= 1 << (n_points - k - 1)
                lbp[i+radius, j+radius] = code
        
        # 计算LBP直方图
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        # 计算纹理分数（熵）
        texture_score = -np.sum(hist * np.log2(hist + 1e-7))
        
        # 归一化到0-1范围
        normalized_score = min(texture_score / 10.0, 1.0)
        
        return normalized_score
    
    def _extract_shape_features(self, image: np.ndarray) -> str:
        """提取形状特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "unknown"
        
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓特征
        area = cv2.contourArea(max_contour)
        perimeter = cv2.arcLength(max_contour, True)
        
        if perimeter == 0:
            return "unknown"
        
        # 计算圆形度
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 计算矩形度
        x, y, w, h = cv2.boundingRect(max_contour)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # 判断形状
        if circularity > 0.8:
            return "cylindrical"  # 接近圆形
        elif rectangularity > 0.7:
            # 计算宽高比
            aspect_ratio = w / h if h > 0 else 0
            if 0.8 < aspect_ratio < 1.2:
                return "square"
            else:
                return "rectangular"
        else:
            return "irregular"
    
    def _aggregate_descriptors(self, descriptors: np.ndarray) -> List[float]:
        """聚合SIFT描述符"""
        if descriptors is None or len(descriptors) == 0:
            return [0.0] * 128  # 返回零向量
        
        # 简单实现：计算描述符的平均值
        avg_descriptor = np.mean(descriptors, axis=0)
        
        return avg_descriptor.tolist()
