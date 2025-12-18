#!/usr/bin/env python3
"""
数据报告模块
实现系统数据的收集、存储和报告生成
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
import sqlite3
from dataclasses import dataclass, asdict
import csv

logger = logging.getLogger(__name__)

@dataclass
class BehaviorEvent:
    """行为事件数据类"""
    event_id: str
    user_id: str
    action: str
    timestamp: datetime
    confidence: float
    position: Optional[tuple] = None
    product_info: Optional[Dict] = None
    duration: Optional[float] = None

@dataclass
class AbnormalEvent:
    """异常事件数据类"""
    event_id: str
    event_type: str
    severity: str
    timestamp: datetime
    description: str
    location: Optional[tuple] = None
    confidence: float = 0.0
    evidence: Optional[str] = None

@dataclass
class InventoryStatus:
    """库存状态数据类"""
    slot_id: str
    product_id: str
    product_name: str
    current_stock: int
    capacity: int
    status: str  # normal, low_stock, empty, anomaly
    timestamp: datetime
    anomaly_type: Optional[str] = None

@dataclass
class ProductRecognition:
    """商品识别数据类"""
    recognition_id: str
    product_id: str
    product_name: str
    slot_id: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = None
    in_hand: bool = False
    hand_position: Optional[tuple] = None

class DataReporter:
    """数据报告器类"""
    
    def __init__(self, config: Dict):
        """
        初始化数据报告器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.reporting_config = config.get("reporting", {})
        
        # 数据库连接
        self.db_connection = None
        self.db_path = "data/system_data.db"
        
        # 初始化数据库
        self._initialize_database()
        
        # 实时数据缓存
        self.realtime_data_cache = {
            "behavior_events": [],
            "abnormal_events": [],
            "inventory_status": [],
            "product_recognitions": [],
            "system_status": {}
        }
        
        # 报告历史
        self.report_history = []
        
        logger.info("数据报告器初始化完成")
    
    def _initialize_database(self):
        """初始化数据库"""
        try:
            # 确保数据目录存在
            os.makedirs("data", exist_ok=True)
            
            # 连接数据库
            self.db_connection = sqlite3.connect(self.db_path)
            self.db_connection.row_factory = sqlite3.Row
            
            # 创建表
            self._create_tables()
            
            logger.info(f"数据库初始化完成: {self.db_path}")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            self.db_connection = None
    
    def _create_tables(self):
        """创建数据库表"""
        if self.db_connection is None:
            return
        
        cursor = self.db_connection.cursor()
        
        # 行为事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT,
                action TEXT,
                timestamp DATETIME,
                confidence REAL,
                position_x INTEGER,
                position_y INTEGER,
                product_id TEXT,
                product_name TEXT,
                duration REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 异常事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS abnormal_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                severity TEXT,
                timestamp DATETIME,
                description TEXT,
                location_x INTEGER,
                location_y INTEGER,
                confidence REAL,
                evidence TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 库存状态表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory_status (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot_id TEXT,
                product_id TEXT,
                product_name TEXT,
                current_stock INTEGER,
                capacity INTEGER,
                status TEXT,
                timestamp DATETIME,
                anomaly_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 商品识别表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS product_recognitions (
                recognition_id TEXT PRIMARY KEY,
                product_id TEXT,
                product_name TEXT,
                slot_id TEXT,
                confidence REAL,
                timestamp DATETIME,
                in_hand BOOLEAN,
                hand_position_x INTEGER,
                hand_position_y INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 系统状态表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_status (
                status_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                module_name TEXT,
                status_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 报告历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS report_history (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT,
                report_date DATE,
                report_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.db_connection.commit()
        logger.info("数据库表创建完成")
    
    def save_behavior_event(self, event_data: Dict):
        """保存行为事件"""
        try:
            # 生成事件ID
            event_id = f"behavior_{int(time.time() * 1000)}_{hash(str(event_data)) % 10000}"
            
            # 创建行为事件对象
            event = BehaviorEvent(
                event_id=event_id,
                user_id=event_data.get("user_id", "unknown"),
                action=event_data.get("action", "unknown"),
                timestamp=datetime.now(),
                confidence=event_data.get("confidence", 0.0),
                position=event_data.get("position"),
                product_info=event_data.get("product"),
                duration=event_data.get("duration")
            )
            
            # 保存到数据库
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                position_x, position_y = event.position if event.position else (None, None)
                product_id = event.product_info.get("product_id") if event.product_info else None
                product_name = event.product_info.get("product_name") if event.product_info else None
                
                cursor.execute('''
                    INSERT INTO behavior_events 
                    (event_id, user_id, action, timestamp, confidence, 
                     position_x, position_y, product_id, product_name, duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.user_id,
                    event.action,
                    event.timestamp.isoformat(),
                    event.confidence,
                    position_x,
                    position_y,
                    product_id,
                    product_name,
                    event.duration
                ))
                
                self.db_connection.commit()
            
            # 更新实时缓存
            self.realtime_data_cache["behavior_events"].append(asdict(event))
            
            # 保留最近100个事件
            if len(self.realtime_data_cache["behavior_events"]) > 100:
                self.realtime_data_cache["behavior_events"] = self.realtime_data_cache["behavior_events"][-100:]
            
            logger.debug(f"行为事件已保存: {event.action} - {event.user_id}")
            
        except Exception as e:
            logger.error(f"保存行为事件失败: {e}")
    
    def save_abnormal_event(self, event_data: Dict):
        """保存异常事件"""
        try:
            # 生成事件ID
            event_id = f"abnormal_{int(time.time() * 1000)}_{hash(str(event_data)) % 10000}"
            
            # 创建异常事件对象
            event = AbnormalEvent(
                event_id=event_id,
                event_type=event_data.get("type", "unknown"),
                severity=event_data.get("severity", "medium"),
                timestamp=datetime.now(),
                description=event_data.get("description", "未知异常"),
                location=event_data.get("location"),
                confidence=event_data.get("confidence", 0.0),
                evidence=json.dumps(event_data.get("evidence")) if event_data.get("evidence") else None
            )
            
            # 保存到数据库
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                location_x, location_y = event.location if event.location else (None, None)
                
                cursor.execute('''
                    INSERT INTO abnormal_events 
                    (event_id, event_type, severity, timestamp, description,
                     location_x, location_y, confidence, evidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.event_type,
                    event.severity,
                    event.timestamp.isoformat(),
                    event.description,
                    location_x,
                    location_y,
                    event.confidence,
                    event.evidence
                ))
                
                self.db_connection.commit()
            
            # 更新实时缓存
            self.realtime_data_cache["abnormal_events"].append(asdict(event))
            
            # 保留最近50个异常事件
            if len(self.realtime_data_cache["abnormal_events"]) > 50:
                self.realtime_data_cache["abnormal_events"] = self.realtime_data_cache["abnormal_events"][-50:]
            
            logger.info(f"异常事件已保存: {event.event_type} - {event.severity}")
            
        except Exception as e:
            logger.error(f"保存异常事件失败: {e}")
    
    def save_inventory_status(self, inventory_data: Dict):
        """保存库存状态"""
        try:
            # 处理库存数据
            for slot_detail in inventory_data.get("slot_details", []):
                # 生成记录ID
                record_id = f"inventory_{slot_detail.get('slot_id', 'unknown')}_{int(time.time() * 1000)}"
                
                # 创建库存状态对象
                status = InventoryStatus(
                    slot_id=slot_detail.get("slot_id", "unknown"),
                    product_id=slot_detail.get("product_id", "unknown"),
                    product_name=slot_detail.get("product_name", "未知商品"),
                    current_stock=slot_detail.get("estimated_stock", 0),
                    capacity=slot_detail.get("capacity", 15),
                    status=self._determine_inventory_status(slot_detail),
                    timestamp=datetime.fromisoformat(slot_detail.get("timestamp", datetime.now().isoformat())),
                    anomaly_type=slot_detail.get("anomaly_type")
                )
                
                # 保存到数据库
                if self.db_connection:
                    cursor = self.db_connection.cursor()
                    
                    cursor.execute('''
                        INSERT INTO inventory_status 
                        (slot_id, product_id, product_name, current_stock, capacity,
                         status, timestamp, anomaly_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        status.slot_id,
                        status.product_id,
                        status.product_name,
                        status.current_stock,
                        status.capacity,
                        status.status,
                        status.timestamp.isoformat(),
                        status.anomaly_type
                    ))
                    
                    self.db_connection.commit()
                
                # 更新实时缓存
                self.realtime_data_cache["inventory_status"].append(asdict(status))
            
            # 保留最近100个库存状态
            if len(self.realtime_data_cache["inventory_status"]) > 100:
                self.realtime_data_cache["inventory_status"] = self.realtime_data_cache["inventory_status"][-100:]
            
            logger.debug("库存状态已保存")
            
        except Exception as e:
            logger.error(f"保存库存状态失败: {e}")
    
    def save_product_recognition(self, recognition_data: Dict):
        """保存商品识别结果"""
        try:
            # 生成识别ID
            recognition_id = f"product_{int(time.time() * 1000)}_{hash(str(recognition_data)) % 10000}"
            
            # 创建商品识别对象
            recognition = ProductRecognition(
                recognition_id=recognition_id,
                product_id=recognition_data.get("product_id", "unknown"),
                product_name=recognition_data.get("product_name", "未知商品"),
                slot_id=recognition_data.get("slot_id"),
                confidence=recognition_data.get("confidence", 0.0),
                timestamp=recognition_data.get("timestamp", datetime.now()),
                in_hand=recognition_data.get("in_hand", False),
                hand_position=recognition_data.get("hand_position")
            )
            
            # 保存到数据库
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                hand_position_x, hand_position_y = recognition.hand_position if recognition.hand_position else (None, None)
                
                cursor.execute('''
                    INSERT INTO product_recognitions 
                    (recognition_id, product_id, product_name, slot_id, confidence,
                     timestamp, in_hand, hand_position_x, hand_position_y)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    recognition.recognition_id,
                    recognition.product_id,
                    recognition.product_name,
                    recognition.slot_id,
                    recognition.confidence,
                    recognition.timestamp.isoformat() if isinstance(recognition.timestamp, datetime) else recognition.timestamp,
                    recognition.in_hand,
                    hand_position_x,
                    hand_position_y
                ))
                
                self.db_connection.commit()
            
            # 更新实时缓存
            self.realtime_data_cache["product_recognitions"].append(asdict(recognition))
            
            # 保留最近50个识别结果
            if len(self.realtime_data_cache["product_recognitions"]) > 50:
                self.realtime_data_cache["product_recognitions"] = self.realtime_data_cache["product_recognitions"][-50:]
            
            logger.debug(f"商品识别已保存: {recognition.product_name}")
            
        except Exception as e:
            logger.error(f"保存商品识别失败: {e}")
    
    def save_system_status(self, module_name: str, status_data: Dict):
        """保存系统状态"""
        try:
            # 更新实时缓存
            self.realtime_data_cache["system_status"][module_name] = {
                "timestamp": datetime.now().isoformat(),
                "data": status_data
            }
            
            # 保存到数据库
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    INSERT INTO system_status 
                    (timestamp, module_name, status_data)
                    VALUES (?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    module_name,
                    json.dumps(status_data)
                ))
                
                self.db_connection.commit()
            
            logger.debug(f"系统状态已保存: {module_name}")
            
        except Exception as e:
            logger.error(f"保存系统状态失败: {e}")
    
    def _determine_inventory_status(self, slot_detail: Dict) -> str:
        """确定库存状态"""
        if slot_detail.get("is_empty", False):
            return "empty"
        elif slot_detail.get("is_low_stock", False):
            return "low_stock"
        elif slot_detail.get("has_anomaly", False):
            return "anomaly"
        else:
            return "normal"
    
    def send_alert(self, alert_data: Dict, severity: str = "medium"):
        """发送警报"""
        try:
            # 这里可以集成邮件、短信、Webhook等通知方式
            # 目前先记录到日志和数据库
            
            alert_id = f"alert_{int(time.time() * 1000)}"
            alert_type = alert_data.get("type", "general")
            description = alert_data.get("description", "未知警报")
            
            alert_record = {
                "alert_id": alert_id,
                "type": alert_type,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "data": alert_data
            }
            
            # 记录到日志
            if severity == "high":
                logger.error(f"高级警报: {description}")
            elif severity == "medium":
                logger.warning(f"中级警报: {description}")
            else:
                logger.info(f"低级警报: {description}")
            
            # 可以在这里添加其他通知方式，例如：
            # - 发送邮件
            # - 发送短信
            # - 调用Webhook
            # - 推送通知到手机App
            
            logger.info(f"警报已发送: {alert_type} - {severity}")
            
        except Exception as e:
            logger.error(f"发送警报失败: {e}")
    
    def get_real_time_data(self) -> Dict:
        """获取实时数据"""
        try:
            # 从数据库获取最新数据
            realtime_data = {
                "timestamp": datetime.now().isoformat(),
                "behavior_events": self._get_recent_behavior_events(limit=20),
                "abnormal_events": self._get_recent_abnormal_events(limit=10),
                "inventory_summary": self._get_inventory_summary(),
                "product_recognitions": self._get_recent_product_recognitions(limit=10),
                "system_status": self.realtime_data_cache["system_status"]
            }
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            return {}
    
    def _get_recent_behavior_events(self, limit: int = 20) -> List[Dict]:
        """获取最近的行为事件"""
        events = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM behavior_events 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    event = dict(row)
                    # 转换时间戳格式
                    if "timestamp" in event:
                        event["timestamp"] = datetime.fromisoformat(event["timestamp"])
                    events.append(event)
            
        except Exception as e:
            logger.error(f"获取行为事件失败: {e}")
        
        return events
    
    def _get_recent_abnormal_events(self, limit: int = 10) -> List[Dict]:
        """获取最近的异常事件"""
        events = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM abnormal_events 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    event = dict(row)
                    # 转换时间戳格式
                    if "timestamp" in event:
                        event["timestamp"] = datetime.fromisoformat(event["timestamp"])
                    events.append(event)
            
        except Exception as e:
            logger.error(f"获取异常事件失败: {e}")
        
        return events
    
    def _get_inventory_summary(self) -> Dict:
        """获取库存摘要"""
        summary = {
            "total_slots": 0,
            "occupied_slots": 0,
            "empty_slots": 0,
            "low_stock_slots": 0,
            "anomaly_slots": 0,
            "total_stock": 0,
            "total_capacity": 0
        }
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                # 获取最新的库存状态
                cursor.execute('''
                    SELECT * FROM inventory_status 
                    WHERE timestamp = (
                        SELECT MAX(timestamp) FROM inventory_status
                    )
                ''')
                
                rows = cursor.fetchall()
                
                for row in rows:
                    status = dict(row)
                    summary["total_slots"] += 1
                    summary["total_stock"] += status.get("current_stock", 0)
                    summary["total_capacity"] += status.get("capacity", 0)
                    
                    status_type = status.get("status", "normal")
                    if status_type == "empty":
                        summary["empty_slots"] += 1
                    elif status_type == "low_stock":
                        summary["low_stock_slots"] += 1
                    elif status_type == "anomaly":
                        summary["anomaly_slots"] += 1
                    else:
                        summary["occupied_slots"] += 1
            
        except Exception as e:
            logger.error(f"获取库存摘要失败: {e}")
        
        return summary
    
    def _get_recent_product_recognitions(self, limit: int = 10) -> List[Dict]:
        """获取最近的商品识别结果"""
        recognitions = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM product_recognitions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    recognition = dict(row)
                    # 转换时间戳格式
                    if "timestamp" in recognition:
                        recognition["timestamp"] = datetime.fromisoformat(recognition["timestamp"])
                    recognitions.append(recognition)
            
        except Exception as e:
            logger.error(f"获取商品识别结果失败: {e}")
        
        return recognitions
    
    def update_real_time_dashboard(self, realtime_data: Dict):
        """更新实时仪表板"""
        try:
            # 这里可以集成WebSocket、API推送等方式
            # 目前先记录到日志
            
            logger.debug(f"实时数据更新: {len(realtime_data.get('behavior_events', []))} 个行为事件")
            
            # 可以在这里添加实时数据推送，例如：
            # - WebSocket广播
            # - REST API更新
            # - MQTT发布
            
        except Exception as e:
            logger.error(f"更新实时仪表板失败: {e}")
    
    def generate_daily_report(self) -> Dict:
        """生成每日报告"""
        try:
            report_date = datetime.now().date()
            report_date_str = report_date.isoformat()
            
            # 获取当天的数据
            start_time = datetime.combine(report_date, datetime.min.time())
            end_time = datetime.combine(report_date + timedelta(days=1), datetime.min.time())
            
            report_data = {
                "report_date": report_date_str,
                "summary": self._generate_daily_summary(start_time, end_time),
                "behavior_analysis": self._generate_behavior_analysis(start_time, end_time),
                "abnormal_events": self._get_abnormal_events_by_date(start_time, end_time),
                "inventory_changes": self._get_inventory_changes_by_date(start_time, end_time),
                "sales_estimation": self._estimate_sales(start_time, end_time),
                "recommendations": self._generate_recommendations(start_time, end_time)
            }
            
            # 保存报告
            self._save_report("daily", report_date_str, report_data)
            
            logger.info(f"每日报告已生成: {report_date_str}")
            
            return report_data
            
        except Exception as e:
            logger.error(f"生成每日报告失败: {e}")
            return {"error": str(e)}
    
    def _generate_daily_summary(self, start_time: datetime, end_time: datetime) -> Dict:
        """生成每日摘要"""
        summary = {
            "total_customers": 0,
            "total_purchases": 0,
            "total_abnormal_events": 0,
            "inventory_changes": 0,
            "peak_hours": [],
            "most_popular_products": []
        }
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                # 统计客户数量
                cursor.execute('''
                    SELECT COUNT(DISTINCT user_id) as customer_count
                    FROM behavior_events
                    WHERE timestamp >= ? AND timestamp < ?
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                row = cursor.fetchone()
                if row:
                    summary["total_customers"] = row["customer_count"]
                
                # 统计购买次数
                cursor.execute('''
                    SELECT COUNT(*) as purchase_count
                    FROM behavior_events
                    WHERE action LIKE '%purchase%' 
                    AND timestamp >= ? AND timestamp < ?
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                row = cursor.fetchone()
                if row:
                    summary["total_purchases"] = row["purchase_count"]
                
                # 统计异常事件
                cursor.execute('''
                    SELECT COUNT(*) as abnormal_count
                    FROM abnormal_events
                    WHERE timestamp >= ? AND timestamp < ?
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                row = cursor.fetchone()
                if row:
                    summary["total_abnormal_events"] = row["abnormal_count"]
                
                # 统计高峰时段
                cursor.execute('''
                    SELECT strftime('%H', timestamp) as hour, COUNT(*) as event_count
                    FROM behavior_events
                    WHERE timestamp >= ? AND timestamp < ?
                    GROUP BY hour
                    ORDER BY event_count DESC
                    LIMIT 3
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                summary["peak_hours"] = [{"hour": row["hour"], "count": row["event_count"]} for row in rows]
                
                # 统计热门商品
                cursor.execute('''
                    SELECT product_name, COUNT(*) as recognition_count
                    FROM product_recognitions
                    WHERE timestamp >= ? AND timestamp < ?
                    AND product_name IS NOT NULL
                    GROUP BY product_name
                    ORDER BY recognition_count DESC
                    LIMIT 5
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                summary["most_popular_products"] = [
                    {"product": row["product_name"], "count": row["recognition_count"]} 
                    for row in rows
                ]
            
        except Exception as e:
            logger.error(f"生成每日摘要失败: {e}")
        
        return summary
    
    def _generate_behavior_analysis(self, start_time: datetime, end_time: datetime) -> Dict:
        """生成行为分析"""
        analysis = {
            "behavior_distribution": {},
            "average_purchase_time": 0,
            "success_rate": 0
        }
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                # 行为分布
                cursor.execute('''
                    SELECT action, COUNT(*) as count
                    FROM behavior_events
                    WHERE timestamp >= ? AND timestamp < ?
                    GROUP BY action
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                analysis["behavior_distribution"] = {row["action"]: row["count"] for row in rows}
                
                # 平均购买时间
                cursor.execute('''
                    SELECT AVG(duration) as avg_duration
                    FROM behavior_events
                    WHERE action LIKE '%purchase%' 
                    AND duration IS NOT NULL
                    AND timestamp >= ? AND timestamp < ?
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                row = cursor.fetchone()
                if row and row["avg_duration"]:
                    analysis["average_purchase_time"] = row["avg_duration"]
                
                # 成功率（完成购买的行为比例）
                total_behavior = sum(analysis["behavior_distribution"].values())
                purchase_behavior = sum(
                    count for action, count in analysis["behavior_distribution"].items() 
                    if "purchase" in action.lower()
                )
                
                if total_behavior > 0:
                    analysis["success_rate"] = purchase_behavior / total_behavior * 100
            
        except Exception as e:
            logger.error(f"生成行为分析失败: {e}")
        
        return analysis
    
    def _get_abnormal_events_by_date(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取指定日期的异常事件"""
        events = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    SELECT event_type, severity, description, timestamp
                    FROM abnormal_events
                    WHERE timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp DESC
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                events = [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"获取异常事件失败: {e}")
        
        return events
    
    def _get_inventory_changes_by_date(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取指定日期的库存变化"""
        changes = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    SELECT slot_id, product_name, current_stock, status, timestamp
                    FROM inventory_status
                    WHERE timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp DESC
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                changes = [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"获取库存变化失败: {e}")
        
        return changes
    
    def _estimate_sales(self, start_time: datetime, end_time: datetime) -> Dict:
        """估算销售额"""
        sales_estimation = {
            "estimated_sales_count": 0,
            "estimated_revenue": 0,
            "popular_products": []
        }
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                # 估算销售数量（基于购买行为）
                cursor.execute('''
                    SELECT COUNT(*) as purchase_count
                    FROM behavior_events
                    WHERE action LIKE '%purchase%' 
                    AND timestamp >= ? AND timestamp < ?
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                row = cursor.fetchone()
                if row:
                    sales_estimation["estimated_sales_count"] = row["purchase_count"]
                
                # 估算收入（基于商品识别和价格）
                # 这里需要商品价格数据，简化处理
                average_price = 3.5  # 平均价格
                sales_estimation["estimated_revenue"] = sales_estimation["estimated_sales_count"] * average_price
                
                # 热门商品
                cursor.execute('''
                    SELECT product_name, COUNT(*) as count
                    FROM product_recognitions
                    WHERE timestamp >= ? AND timestamp < ?
                    AND product_name IS NOT NULL
                    GROUP BY product_name
                    ORDER BY count DESC
                    LIMIT 5
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                sales_estimation["popular_products"] = [
                    {"product": row["product_name"], "sales_estimate": row["count"]} 
                    for row in rows
                ]
            
        except Exception as e:
            logger.error(f"估算销售额失败: {e}")
        
        return sales_estimation
    
    def _generate_recommendations(self, start_time: datetime, end_time: datetime) -> List[str]:
        """生成推荐建议"""
        recommendations = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                # 检查低库存
                cursor.execute('''
                    SELECT slot_id, product_name, current_stock
                    FROM inventory_status
                    WHERE status = 'low_stock' 
                    OR status = 'empty'
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''')
                
                rows = cursor.fetchall()
                for row in rows:
                    recommendations.append(f"槽位 {row['slot_id']} ({row['product_name']}) 库存低: {row['current_stock']}个")
                
                # 检查异常事件
                cursor.execute('''
                    SELECT COUNT(*) as abnormal_count
                    FROM abnormal_events
                    WHERE severity = 'high' 
                    AND timestamp >= ? AND timestamp < ?
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                row = cursor.fetchone()
                if row and row["abnormal_count"] > 0:
                    recommendations.append(f"发现 {row['abnormal_count']} 个高级异常事件，需要立即检查")
                
                # 检查设备状态
                if len(recommendations) == 0:
                    recommendations.append("系统运行正常，无紧急问题")
            
        except Exception as e:
            logger.error(f"生成推荐建议失败: {e}")
            recommendations.append("生成推荐时发生错误")
        
        return recommendations
    
    def _save_report(self, report_type: str, report_date: str, report_data: Dict):
        """保存报告"""
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    INSERT INTO report_history 
                    (report_type, report_date, report_data)
                    VALUES (?, ?, ?)
                ''', (
                    report_type,
                    report_date,
                    json.dumps(report_data, ensure_ascii=False, indent=2)
                ))
                
                self.db_connection.commit()
            
            # 更新报告历史缓存
            self.report_history.append({
                "type": report_type,
                "date": report_date,
                "data": report_data
            })
            
            # 保留最近100个报告
            if len(self.report_history) > 100:
                self.report_history = self.report_history[-100:]
            
            logger.debug(f"报告已保存: {report_type} - {report_date}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    def send_daily_report(self, report_data: Dict):
        """发送每日报告"""
        try:
            # 这里可以集成邮件发送、API推送等
            # 目前先记录到日志
            
            report_date = report_data.get("report_date", "未知日期")
            summary = report_data.get("summary", {})
            
            logger.info(f"=== 每日报告 {report_date} ===")
            logger.info(f"总客户数: {summary.get('total_customers', 0)}")
            logger.info(f"总购买次数: {summary.get('total_purchases', 0)}")
            logger.info(f"异常事件: {summary.get('total_abnormal_events', 0)}")
            logger.info(f"热门商品: {', '.join([p['product'] for p in summary.get('most_popular_products', [])])}")
            
            # 可以在这里添加报告发送功能，例如：
            # - 发送邮件给管理员
            # - 上传到云存储
            # - 推送到管理平台
            
            logger.info(f"每日报告已发送: {report_date}")
            
        except Exception as e:
            logger.error(f"发送每日报告失败: {e}")
    
    def export_data(self, data_type: str, start_time: datetime, end_time: datetime, format: str = "json") -> str:
        """导出数据"""
        try:
            export_data = {}
            
            if data_type == "behavior":
                export_data = self._get_behavior_data(start_time, end_time)
            elif data_type == "abnormal":
                export_data = self._get_abnormal_data(start_time, end_time)
            elif data_type == "inventory":
                export_data = self._get_inventory_data(start_time, end_time)
            elif data_type == "all":
                export_data = {
                    "behavior": self._get_behavior_data(start_time, end_time),
                    "abnormal": self._get_abnormal_data(start_time, end_time),
                    "inventory": self._get_inventory_data(start_time, end_time)
                }
            
            if format == "json":
                return json.dumps(export_data, ensure_ascii=False, indent=2, default=str)
            elif format == "csv":
                return self._convert_to_csv(export_data)
            else:
                return str(export_data)
            
        except Exception as e:
            logger.error(f"导出数据失败: {e}")
            return f"导出失败: {str(e)}"
    
    def _get_behavior_data(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取行为数据"""
        data = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM behavior_events
                    WHERE timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"获取行为数据失败: {e}")
        
        return data
    
    def _get_abnormal_data(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取异常数据"""
        data = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM abnormal_events
                    WHERE timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"获取异常数据失败: {e}")
        
        return data
    
    def _get_inventory_data(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取库存数据"""
        data = []
        
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM inventory_status
                    WHERE timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp
                ''', (start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"获取库存数据失败: {e}")
        
        return data
    
    def _convert_to_csv(self, data: Dict) -> str:
        """将数据转换为CSV格式"""
        try:
            import io
            
            output = io.StringIO()
            
            if isinstance(data, dict):
                # 处理嵌套字典
                for data_type, records in data.items():
                    if records:
                        output.write(f"=== {data_type.upper()} ===\n")
                        
                        # 获取字段名
                        fieldnames = list(records[0].keys())
                        
                        # 写入CSV
                        writer = csv.DictWriter(output, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(records)
                        output.write("\n\n")
            elif isinstance(data, list):
                # 处理列表
                if data:
                    fieldnames = list(data[0].keys())
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"转换为CSV失败: {e}")
            return f"CSV转换失败: {str(e)}"
    
    def get_status(self) -> Dict:
        """获取报告器状态"""
        try:
            # 获取数据库统计
            db_stats = {}
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                tables = ["behavior_events", "abnormal_events", "inventory_status", "product_recognitions"]
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    row = cursor.fetchone()
                    db_stats[table] = row["count"] if row else 0
            
            status = {
                "database_path": self.db_path,
                "database_stats": db_stats,
                "realtime_cache": {
                    "behavior_events": len(self.realtime_data_cache["behavior_events"]),
                    "abnormal_events": len(self.realtime_data_cache["abnormal_events"]),
                    "inventory_status": len(self.realtime_data_cache["inventory_status"]),
                    "product_recognitions": len(self.realtime_data_cache["product_recognitions"]),
                    "system_status_modules": len(self.realtime_data_cache["system_status"])
                },
                "report_history_count": len(self.report_history),
                "last_report_date": self.report_history[-1]["date"] if self.report_history else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取报告器状态失败: {e}")
            return {"error": str(e)}
    
    def close(self):
        """关闭数据库连接"""
        try:
            if self.db_connection:
                self.db_connection.close()
                logger.info("数据库连接已关闭")
            
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def __del__(self):
        """析构函数"""
        self.close()
