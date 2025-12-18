#!/usr/bin/env python3
"""
售货机视觉分析系统演示脚本
展示系统的主要功能和使用方法
"""

import os
import sys
import json
import time

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_configuration():
    """演示配置功能"""
    print("=== 配置功能演示 ===")
    
    try:
        from config import SystemConfig
        
        # 创建配置实例
        config = SystemConfig()
        
        # 显示默认配置
        print("1. 默认配置:")
        print(f"   系统名称: {config.get('system.name')}")
        print(f"   版本: {config.get('system.version')}")
        print(f"   摄像头索引: {config.get('video.camera_index')}")
        print(f"   分辨率: {config.get('video.resolution')}")
        print(f"   帧率: {config.get('video.fps')}")
        
        # 修改配置
        config.set("system.name", "演示系统")
        config.set("video.resolution", [1280, 720])
        
        print("\n2. 修改后的配置:")
        print(f"   系统名称: {config.get('system.name')}")
        print(f"   分辨率: {config.get('video.resolution')}")
        
        # 保存配置
        config.save("demo_config.json")
        print(f"\n3. 配置已保存到: demo_config.json")
        
        # 验证配置
        validation = config.validate()
        print(f"4. 配置验证: {'通过' if validation['valid'] else '失败'}")
        
        return True
        
    except Exception as e:
        print(f"配置演示失败: {e}")
        return False

def demo_data_reporting():
    """演示数据报告功能"""
    print("\n=== 数据报告功能演示 ===")
    
    try:
        from Web.data_reporter import DataReporter
        
        # 创建测试配置
        test_config = {
            "reporting": {
                "database_path": "data/demo_system_data.db"
            }
        }
        
        # 创建数据报告器
        reporter = DataReporter(test_config)
        
        # 演示行为事件记录
        print("1. 记录行为事件:")
        behavior_events = [
            {
                "user_id": "user_001",
                "action": "approach",
                "confidence": 0.92,
                "position": (150, 200),
                "product": {
                    "product_id": "cola_001",
                    "product_name": "可口可乐"
                }
            },
            {
                "user_id": "user_001",
                "action": "select",
                "confidence": 0.85,
                "position": (160, 210),
                "product": {
                    "product_id": "cola_001",
                    "product_name": "可口可乐"
                }
            },
            {
                "user_id": "user_001",
                "action": "purchase",
                "confidence": 0.88,
                "position": (155, 205),
                "product": {
                    "product_id": "cola_001",
                    "product_name": "可口可乐"
                }
            }
        ]
        
        for event in behavior_events:
            reporter.save_behavior_event(event)
            print(f"   - {event['action']}: {event['product']['product_name']} (置信度: {event['confidence']})")
        
        # 演示异常事件记录
        print("\n2. 记录异常事件:")
        abnormal_events = [
            {
                "type": "vandalism",
                "severity": "low",
                "description": "轻微敲击",
                "location": (300, 400),
                "confidence": 0.75
            },
            {
                "type": "theft",
                "severity": "high",
                "description": "疑似盗窃行为",
                "location": (280, 380),
                "confidence": 0.82
            }
        ]
        
        for event in abnormal_events:
            reporter.save_abnormal_event(event)
            print(f"   - {event['type']}: {event['description']} (严重程度: {event['severity']})")
        
        # 演示库存管理
        print("\n3. 记录库存状态:")
        inventory_data = {
            "slot_details": [
                {
                    "slot_id": "slot_001",
                    "product_id": "cola_001",
                    "product_name": "可口可乐",
                    "estimated_stock": 5,
                    "capacity": 15,
                    "is_empty": False,
                    "is_low_stock": True,
                    "has_anomaly": False
                },
                {
                    "slot_id": "slot_002",
                    "product_id": "sprite_001",
                    "product_name": "雪碧",
                    "estimated_stock": 12,
                    "capacity": 15,
                    "is_empty": False,
                    "is_low_stock": False,
                    "has_anomaly": False
                },
                {
                    "slot_id": "slot_003",
                    "product_id": "water_001",
                    "product_name": "矿泉水",
                    "estimated_stock": 0,
                    "capacity": 15,
                    "is_empty": True,
                    "is_low_stock": True,
                    "has_anomaly": False
                }
            ]
        }
        
        reporter.save_inventory_status(inventory_data)
        
        for slot in inventory_data["slot_details"]:
            status = "空" if slot["is_empty"] else ("低库存" if slot["is_low_stock"] else "正常")
            print(f"   - {slot['product_name']}: {slot['estimated_stock']}/{slot['capacity']} ({status})")
        
        # 获取实时数据
        print("\n4. 获取实时数据:")
        realtime_data = reporter.get_real_time_data()
        
        print(f"   行为事件数量: {len(realtime_data.get('behavior_events', []))}")
        print(f"   异常事件数量: {len(realtime_data.get('abnormal_events', []))}")
        print(f"   库存槽位数量: {len(realtime_data.get('inventory_status', {}).get('slot_details', []))}")
        
        # 生成报告
        print("\n5. 生成每日报告:")
        daily_report = reporter.generate_daily_report()
        
        print(f"   报告日期: {daily_report.get('report_date', 'N/A')}")
        print(f"   总用户数: {daily_report.get('total_users', 0)}")
        print(f"   总交易数: {daily_report.get('total_transactions', 0)}")
        print(f"   异常事件数: {daily_report.get('abnormal_events_count', 0)}")
        
        # 清理
        reporter.close()
        
        # 删除演示数据库
        if os.path.exists("data/demo_system_data.db"):
            os.remove("data/demo_system_data.db")
        
        print("\n✅ 数据报告演示完成")
        return True
        
    except Exception as e:
        print(f"数据报告演示失败: {e}")
        return False

def demo_system_integration():
    """演示系统集成"""
    print("\n=== 系统集成演示 ===")
    
    try:
        from main import VendingMachineVisionSystem
        
        # 创建演示配置
        demo_config = {
            "system": {
                "name": "演示系统",
                "version": "1.0.0",
                "mode": "demo",
                "log_level": "INFO"
            },
            "video": {
                "camera_index": 0,
                "resolution": [640, 480],
                "fps": 10
            },
            "analysis": {
                "enable_behavior_analysis": True,
                "enable_abnormal_detection": True,
                "enable_product_recognition": True,
                "enable_inventory_management": True
            },
            "visualization": {
                "enable_display": False
            }
        }
        
        # 保存演示配置
        with open("demo_system_config.json", "w", encoding="utf-8") as f:
            json.dump(demo_config, f, indent=2)
        
        print("1. 创建系统实例...")
        system = VendingMachineVisionSystem("demo_system_config.json")
        
        print("2. 初始化系统模块...")
        system._initialize_modules()
        
        print("3. 获取系统状态...")
        status = system.get_status()
        
        print(f"   系统状态: {'运行中' if status['is_running'] else '已停止'}")
        print(f"   模块状态:")
        for module, enabled in status['modules'].items():
            print(f"     - {module}: {'启用' if enabled else '禁用'}")
        
        print("4. 模拟系统运行...")
        print("   (模拟处理中...)")
        time.sleep(2)
        
        print("5. 停止系统...")
        system.stop()
        
        # 清理
        if os.path.exists("demo_system_config.json"):
            os.remove("demo_system_config.json")
        
        print("\n✅ 系统集成演示完成")
        return True
        
    except Exception as e:
        print(f"系统集成演示失败: {e}")
        return False

def demo_usage_scenarios():
    """演示使用场景"""
    print("\n=== 使用场景演示 ===")
    
    scenarios = [
        {
            "name": "场景1: 用户购买流程",
            "description": "用户接近售货机 -> 选择商品 -> 完成购买",
            "steps": [
                "检测到用户接近售货机",
                "识别用户选择的商品",
                "确认购买行为",
                "更新库存状态",
                "记录交易数据"
            ]
        },
        {
            "name": "场景2: 异常检测",
            "description": "检测破坏行为或盗窃行为",
            "steps": [
                "监控售货机周围活动",
                "检测异常行为模式",
                "评估异常严重程度",
                "记录异常证据",
                "生成警报通知"
            ]
        },
        {
            "name": "场景3: 库存管理",
            "description": "监控库存状态和自动补货",
            "steps": [
                "实时监控各槽位库存",
                "检测低库存或空槽位",
                "生成补货建议",
                "记录库存变化",
                "分析销售趋势"
            ]
        },
        {
            "name": "场景4: 数据分析",
            "description": "生成业务报告和洞察",
            "steps": [
                "收集各类事件数据",
                "分析用户行为模式",
                "识别热门商品",
                "检测异常趋势",
                "生成可视化报告"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   描述: {scenario['description']}")
        print(f"   步骤:")
        for step in scenario['steps']:
            print(f"     - {step}")
    
    print("\n✅ 使用场景演示完成")
    return True

def main():
    """主演示函数"""
    print("=" * 60)
    print("售货机视觉分析系统演示")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    demo_results = []
    
    # 运行演示
    demo_results.append(("配置功能", demo_configuration()))
    demo_results.append(("数据报告", demo_data_reporting()))
    demo_results.append(("系统集成", demo_system_integration()))
    demo_results.append(("使用场景", demo_usage_scenarios()))
    
    # 打印演示结果摘要
    print("\n" + "=" * 60)
    print("演示结果摘要")
    print("=" * 60)
    
    total_demos = len(demo_results)
    successful_demos = sum(1 for _, success in demo_results if success)
    
    for demo_name, success in demo_results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{demo_name:15} {status}")
    
    print(f"\n总计: {successful_demos}/{total_demos} 个演示成功")
    
    if successful_demos == total_demos:
        print("\n🎉 所有演示成功完成！")
        print("\n系统功能完整，可以正常运行。")
        print("下一步:")
        print("1. 运行 'python run.py --setup' 进行系统设置")
        print("2. 运行 'python run.py --test' 进行系统测试")
        print("3. 运行 'python run.py' 启动系统")
    else:
        print("\n⚠️  部分演示失败，请检查错误信息。")
    
    print("\n" + "=" * 60)
    
    return 0 if successful_demos == total_demos else 1

if __name__ == "__main__":
    sys.exit(main())
