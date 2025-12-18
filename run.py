#!/usr/bin/env python3
"""
售货机视觉分析系统启动脚本
简化系统启动和配置
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_default_config():
    """创建默认配置文件"""
    config_path = "config.json"
    
    if os.path.exists(config_path):
        print(f"配置文件已存在: {config_path}")
        return config_path
    
    # 导入配置类
    from config import SystemConfig
    
    # 创建默认配置
    config = SystemConfig()
    config.save(config_path)
    
    print(f"默认配置文件已创建: {config_path}")
    return config_path

def setup_directories():
    """设置必要的目录"""
    directories = [
        "data",
        "logs",
        "recordings",
        "models",
        "screenshots",
        "visualizations",
        "backups",
        "test_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"目录已创建/确认: {directory}")
    
    print("所有必要目录已设置完成")

def install_dependencies():
    """安装依赖"""
    print("正在检查依赖...")
    
    # 检查OpenCV
    try:
        import cv2
        print(f"✓ OpenCV 已安装 (版本: {cv2.__version__})")
    except ImportError:
        print("✗ OpenCV 未安装，请运行: pip install opencv-python")
    
    # 检查NumPy
    try:
        import numpy as np
        print(f"✓ NumPy 已安装 (版本: {np.__version__})")
    except ImportError:
        print("✗ NumPy 未安装，请运行: pip install numpy")
    
    # 检查其他依赖
    dependencies = [
        ("PIL", "Pillow"),
        ("sqlite3", "sqlite3 (内置)"),
        ("json", "json (内置)"),
        ("logging", "logging (内置)"),
        ("threading", "threading (内置)"),
        ("datetime", "datetime (内置)"),
        ("argparse", "argparse (内置)")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} 可用")
        except ImportError:
            print(f"✗ {name} 不可用")
    
    print("\n如果需要安装缺失的依赖，请运行:")
    print("pip install opencv-python numpy Pillow")

def test_system():
    """测试系统"""
    print("正在测试系统...")
    
    try:
        # 导入主系统
        from src.main import VendingMachineVisionSystem
        
        # 创建系统实例
        system = VendingMachineVisionSystem("config.json")
        
        # 测试初始化
        system._initialize_modules()
        
        # 测试状态获取
        status = system.get_status()
        
        print("✓ 系统测试通过")
        print(f"  模块状态:")
        for module, enabled in status['modules'].items():
            print(f"    {module}: {'✓ 启用' if enabled else '✗ 禁用'}")
        
        # 清理
        system.stop()
        
    except Exception as e:
        print(f"✗ 系统测试失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="售货机视觉分析系统启动脚本")
    parser.add_argument("--setup", action="store_true", help="设置系统（创建配置文件和目录）")
    parser.add_argument("--test", action="store_true", help="测试系统")
    parser.add_argument("--install-check", action="store_true", help="检查依赖安装")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--no-display", action="store_true", help="不显示视频窗口")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置模式
    if args.setup:
        print("=== 系统设置 ===")
        create_default_config()
        setup_directories()
        install_dependencies()
        print("\n设置完成！")
        return
    
    # 安装检查模式
    if args.install_check:
        print("=== 依赖检查 ===")
        install_dependencies()
        return
    
    # 测试模式
    if args.test:
        print("=== 系统测试 ===")
        success = test_system()
        if success:
            print("\n所有测试通过！系统可以正常运行。")
        else:
            print("\n测试失败，请检查错误信息。")
        return
    
    # 正常启动模式
    print("=== 启动售货机视觉分析系统 ===")
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        create = input("是否创建默认配置文件？(y/n): ")
        if create.lower() == 'y':
            create_default_config()
        else:
            print("启动中止")
            return
    
    # 检查必要目录
    setup_directories()
    
    # 启动系统
    try:
        # 导入主系统
        from src.main import main as system_main
        
        # 准备命令行参数
        sys.argv = ["main.py"]
        if args.no_display:
            sys.argv.append("--no-display")
        sys.argv.extend(["--log-level", args.log_level])
        sys.argv.extend(["--config", args.config])
        
        print(f"使用配置文件: {args.config}")
        print(f"日志级别: {args.log_level}")
        print(f"显示窗口: {'否' if args.no_display else '是'}")
        print("\n启动系统... (按 Ctrl+C 停止)")
        print("控制命令:")
        print("  q - 退出系统")
        print("  p - 暂停/恢复")
        print("  s - 保存截图")
        
        # 启动系统
        system_main()
        
    except KeyboardInterrupt:
        print("\n系统已停止")
    except Exception as e:
        print(f"系统启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
