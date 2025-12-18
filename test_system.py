#!/usr/bin/env python3
"""
系统功能测试脚本
测试售货机视觉分析系统的基本功能
"""

import os
import sys
import json

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_module():
    """测试配置模块"""
    print("测试配置模块...")
    try:
        from config import SystemConfig
        
        # 创建配置实例
        config = SystemConfig()
        
        # 测试配置获取
        system_name = config.get("system.name")
        camera_index = config.get("video.camera_index")
        
        print(f"✓ 配置模块测试通过")
        print(f"  系统名称: {system_name}")
        print(f"  摄像头索引: {camera_index}")
        
        # 测试配置验证
        validation = config.validate()
        if validation["valid"]:
            print(f"  配置验证: 通过")
        else:
            print(f"  配置验证: 失败")
            for error in validation["errors"]:
                print(f"    错误: {error}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置模块测试失败: {e}")
        return False

def test_data_reporter_module():
    """测试数据报告模块"""
    print("\n测试数据报告模块...")
    try:
        from Web.data_reporter import DataReporter
        
        # 创建测试配置
        test_config = {
            "reporting": {
                "database_path": "data/test_system_data.db"
            }
        }
        
        # 创建数据报告器
        reporter = DataReporter(test_config)
        
        # 测试行为事件保存
        behavior_event = {
            "user_id": "test_user_001",
            "action": "approach",
            "confidence": 0.85,
            "position": (100, 200),
            "product": {
                "product_id": "prod_001",
                "product_name": "测试商品"
            }
        }
        
        reporter.save_behavior_event(behavior_event)
        
        # 测试异常事件保存
        abnormal_event = {
            "type": "vandalism",
            "severity": "medium",
            "description": "测试异常事件",
            "location": (150, 250),
            "confidence": 0.75
        }
        
        reporter.save_abnormal_event(abnormal_event)
        
        # 测试库存状态保存
        inventory_data = {
            "slot_details": [
                {
                    "slot_id": "slot_001",
                    "product_id": "prod_001",
                    "product_name": "测试商品",
                    "estimated_stock": 8,
                    "capacity": 15,
                    "is_empty": False,
                    "is_low_stock": False,
                    "has_anomaly": False
                }
            ]
        }
        
        reporter.save_inventory_status(inventory_data)
        
        # 测试实时数据获取
        realtime_data = reporter.get_real_time_data()
        
        print(f"✓ 数据报告模块测试通过")
        print(f"  数据库路径: {test_config['reporting']['database_path']}")
        print(f"  实时数据包含: {len(realtime_data.get('behavior_events', []))} 个行为事件")
        
        # 清理
        reporter.close()
        
        # 删除测试数据库
        if os.path.exists("data/test_system_data.db"):
            os.remove("data/test_system_data.db")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据报告模块测试失败: {e}")
        return False

def test_main_system():
    """测试主系统"""
    print("\n测试主系统...")
    try:
        from main import VendingMachineVisionSystem
        
        # 创建测试配置
        test_config = {
            "system": {
                "name": "Test System",
                "version": "1.0.0",
                "mode": "test",
                "log_level": "INFO"
            },
            "video": {
                "camera_index": 0,
                "resolution": [640, 480],
                "fps": 10
            },
            "analysis": {
                "enable_behavior_analysis": False,
                "enable_abnormal_detection": False,
                "enable_product_recognition": False,
                "enable_inventory_management": False
            },
            "visualization": {
                "enable_display": False
            }
        }
        
        # 保存测试配置
        with open("test_config.json", "w", encoding="utf-8") as f:
            json.dump(test_config, f, indent=2)
        
        # 创建系统实例
        system = VendingMachineVisionSystem("test_config.json")
        
        # 测试初始化
        system._initialize_modules()
        
        # 测试状态获取
        status = system.get_status()
        
        print(f"✓ 主系统测试通过")
        print(f"  系统状态: {'运行中' if status['is_running'] else '已停止'}")
        print(f"  模块状态:")
        for module, enabled in status['modules'].items():
            print(f"    {module}: {'启用' if enabled else '禁用'}")
        
        # 清理
        system.stop()
        
        # 删除测试配置文件
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")
        
        return True
        
    except Exception as e:
        print(f"✗ 主系统测试失败: {e}")
        return False

def test_directory_structure():
    """测试目录结构"""
    print("\n测试目录结构...")
    
    required_dirs = [
        "src",
        "src/Video",
        "src/predict",
        "src/Web"
    ]
    
    required_files = [
        "src/config.py",
        "src/main.py",
        "src/Video/video_capture.py",
        "src/predict/behavior_analyzer.py",
        "src/predict/abnormal_detector.py",
        "src/predict/product_recognizer.py",
        "src/predict/inventory_manager.py",
        "src/Web/data_reporter.py",
        "run.py",
        "README.md",
        "requirements.txt"
    ]
    
    all_passed = True
    
    # 检查目录
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ 目录存在: {directory}")
        else:
            print(f"✗ 目录不存在: {directory}")
            all_passed = False
    
    # 检查文件
    for file in required_files:
        if os.path.exists(file) and os.path.isfile(file):
            print(f"✓ 文件存在: {file}")
        else:
            print(f"✗ 文件不存在: {file}")
            all_passed = False
    
    return all_passed

def main():
    """主测试函数"""
    print("=== 售货机视觉分析系统功能测试 ===\n")
    
    # 创建必要的测试目录
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    test_results = []
    
    # 运行测试
    test_results.append(("目录结构", test_directory_structure()))
    test_results.append(("配置模块", test_config_module()))
    test_results.append(("数据报告模块", test_data_reporter_module()))
    test_results.append(("主系统", test_main_system()))
    
    # 打印测试结果摘要
    print("\n=== 测试结果摘要 ===")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, passed in test_results if passed)
    
    for test_name, passed in test_results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name:15} {status}")
    
    print(f"\n总计: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        print("\n✅ 所有测试通过！系统可以正常运行。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
