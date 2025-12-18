#!/usr/bin/env python3
"""
系统配置文件
定义售货机视觉分析系统的配置参数
"""

import json
import os
from typing import Dict, Any

class SystemConfig:
    """系统配置类"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        初始化系统配置
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = self._get_default_config()
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并配置，用户配置覆盖默认配置
                    default_config.update(user_config)
                    return default_config
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
                return default_config
        else:
            print(f"配置文件不存在: {self.config_file}，使用默认配置")
            return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "system": {
                "name": "RLL-LabelAnyThing",
                "version": "1.0.0",
                "mode": "development",  # development, production
                "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
                "data_dir": "data",
                "recordings_dir": "recordings",
                "models_dir": "models"
            },
            "video": {
                "camera_index": 0,
                "resolution": [1920, 1080],
                "fps": 30,
                "recording_duration": 10,
                "record_video": False,
                "save_frames": False,
                "frame_save_interval": 10
            },
            "analysis": {
                "min_confidence": 0.7,
                "detection_interval": 1.0,  # 检测间隔（秒）
                "enable_behavior_analysis": True,
                "enable_abnormal_detection": True,
                "enable_product_recognition": True,
                "enable_inventory_management": True
            },
            "behavior": {
                "detection_threshold": 0.6,
                "tracking_history_size": 50,
                "min_motion_area": 500,
                "max_motion_area": 50000,
                "user_timeout": 30.0,  # 用户超时时间（秒）
                "purchase_detection_confidence": 0.8,
                "theft_detection_confidence": 0.85
            },
            "abnormal": {
                "detection_threshold": 0.7,
                "check_interval": 2.0,
                "enable_vandalism_detection": True,
                "enable_theft_detection": True,
                "enable_malfunction_detection": True,
                "alert_cooldown": 60.0  # 警报冷却时间（秒）
            },
            "product": {
                "min_confidence": 0.7,
                "database_path": "data/products_database.json",
                "enable_slot_detection": True,
                "enable_hand_detection": True,
                "slot_detection_confidence": 0.6,
                "hand_detection_confidence": 0.7
            },
            "inventory": {
                "low_stock_threshold": 5,
                "empty_slot_threshold": 0,
                "check_interval": 5.0,
                "enable_anomaly_detection": True,
                "enable_stock_estimation": True,
                "save_inventory_state": True,
                "inventory_state_path": "data/inventory_state.json"
            },
            "reporting": {
                "enable_real_time_dashboard": True,
                "enable_daily_reports": True,
                "report_generation_time": "23:59",  # 每日报告生成时间
                "database_path": "data/system_data.db",
                "enable_alerts": True,
                "alert_channels": ["log"],  # log, email, webhook
                "email_settings": {
                    "enabled": False,
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "webhook_settings": {
                    "enabled": False,
                    "url": "",
                    "headers": {}
                }
            },
            "visualization": {
                "enable_display": True,
                "display_resolution": [1280, 720],
                "show_fps": True,
                "show_timestamp": True,
                "show_detections": True,
                "show_inventory": True,
                "show_alerts": True,
                "save_visualizations": False,
                "visualization_save_path": "visualizations"
            },
            "performance": {
                "max_workers": 4,
                "queue_size": 100,
                "enable_profiling": False,
                "profiling_interval": 60.0,
                "memory_limit_mb": 1024,
                "cpu_usage_limit": 80.0
            },
            "hardware": {
                "gpu_enabled": False,
                "gpu_device_id": 0,
                "enable_cuda": False,
                "enable_opencl": False,
                "memory_optimization": True
            },
            "network": {
                "enable_api": False,
                "api_host": "0.0.0.0",
                "api_port": 8080,
                "enable_websocket": False,
                "websocket_port": 8081,
                "enable_remote_access": False,
                "authentication_required": True,
                "api_keys": []
            },
            "maintenance": {
                "auto_cleanup": True,
                "cleanup_interval_days": 7,
                "max_log_files": 10,
                "max_recordings_days": 30,
                "max_data_days": 90,
                "backup_enabled": True,
                "backup_interval_days": 1,
                "backup_path": "backups"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔符，如 "video.camera_index"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持点分隔符
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        # 遍历到最后一个键的父级
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def save(self, config_file: str = None):
        """
        保存配置到文件
        
        Args:
            config_file: 配置文件路径，如果为None则使用初始化时的路径
        """
        if config_file is None:
            config_file = self.config_file
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            print(f"配置已保存到: {config_file}")
            
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        获取模块配置
        
        Args:
            module_name: 模块名称
            
        Returns:
            模块配置字典
        """
        return self.config.get(module_name, {})
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        从字典更新配置
        
        Args:
            config_dict: 配置字典
        """
        self.config.update(config_dict)
    
    def validate(self) -> Dict[str, Any]:
        """
        验证配置
        
        Returns:
            验证结果，包含错误和警告
        """
        result = {
            "errors": [],
            "warnings": [],
            "valid": True
        }
        
        # 验证视频配置
        video_config = self.config.get("video", {})
        if not isinstance(video_config.get("camera_index"), int):
            result["errors"].append("video.camera_index 必须是整数")
        
        if not isinstance(video_config.get("fps"), (int, float)) or video_config.get("fps") <= 0:
            result["errors"].append("video.fps 必须是正数")
        
        # 验证分析配置
        analysis_config = self.config.get("analysis", {})
        confidence = analysis_config.get("min_confidence", 0.7)
        if not 0 <= confidence <= 1:
            result["errors"].append("analysis.min_confidence 必须在0到1之间")
        
        # 验证库存配置
        inventory_config = self.config.get("inventory", {})
        low_stock = inventory_config.get("low_stock_threshold", 5)
        if not isinstance(low_stock, int) or low_stock < 0:
            result["errors"].append("inventory.low_stock_threshold 必须是非负整数")
        
        # 检查目录是否存在
        data_dir = self.config.get("system", {}).get("data_dir", "data")
        if not os.path.exists(data_dir):
            result["warnings"].append(f"数据目录不存在: {data_dir}")
        
        # 更新验证状态
        result["valid"] = len(result["errors"]) == 0
        
        return result
    
    def print_summary(self):
        """打印配置摘要"""
        print("=== 系统配置摘要 ===")
        
        # 系统信息
        system = self.config.get("system", {})
        print(f"系统名称: {system.get('name')}")
        print(f"版本: {system.get('version')}")
        print(f"运行模式: {system.get('mode')}")
        print(f"日志级别: {system.get('log_level')}")
        
        # 视频配置
        video = self.config.get("video", {})
        print(f"\n视频配置:")
        print(f"  摄像头索引: {video.get('camera_index')}")
        print(f"  分辨率: {video.get('resolution')}")
        print(f"  FPS: {video.get('fps')}")
        
        # 分析配置
        analysis = self.config.get("analysis", {})
        print(f"\n分析配置:")
        print(f"  最小置信度: {analysis.get('min_confidence')}")
        print(f"  行为分析: {'启用' if analysis.get('enable_behavior_analysis') else '禁用'}")
        print(f"  异常检测: {'启用' if analysis.get('enable_abnormal_detection') else '禁用'}")
        print(f"  商品识别: {'启用' if analysis.get('enable_product_recognition') else '禁用'}")
        print(f"  库存管理: {'启用' if analysis.get('enable_inventory_management') else '禁用'}")
        
        # 库存配置
        inventory = self.config.get("inventory", {})
        print(f"\n库存配置:")
        print(f"  低库存阈值: {inventory.get('low_stock_threshold')}")
        print(f"  空槽位阈值: {inventory.get('empty_slot_threshold')}")
        
        # 验证配置
        validation = self.validate()
        if validation["valid"]:
            print(f"\n配置验证: 通过")
        else:
            print(f"\n配置验证: 失败")
            for error in validation["errors"]:
                print(f"  错误: {error}")
        
        if validation["warnings"]:
            print(f"\n警告:")
            for warning in validation["warnings"]:
                print(f"  {warning}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.copy()


# 全局配置实例
_config_instance = None

def get_config(config_file: str = "config.json") -> SystemConfig:
    """
    获取全局配置实例
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        系统配置实例
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = SystemConfig(config_file)
    
    return _config_instance


if __name__ == "__main__":
    # 测试配置
    config = get_config()
    config.print_summary()
    
    # 验证配置
    validation = config.validate()
    if not validation["valid"]:
        print("\n配置验证失败:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    # 保存默认配置
    config.save("config.default.json")
    print(f"\n默认配置已保存到: config.default.json")
