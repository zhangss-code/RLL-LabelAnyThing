#!/usr/bin/env python3
"""
售货机视觉分析系统主程序
整合所有模块，实现完整的售货机监控和分析功能
"""

import cv2
import time
import logging
import threading
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional

# 导入自定义模块
from config import get_config
from Video.video_capture import VideoCaptureManager
from predict.behavior_analyzer import BehaviorAnalyzer
from predict.abnormal_detector import AbnormalDetector
from predict.product_recognizer import ProductRecognizer
from predict.inventory_manager import InventoryManager
from Web.data_reporter import DataReporter

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VendingMachineVisionSystem:
    """售货机视觉分析系统主类"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        初始化系统
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = get_config(config_file)
        
        # 系统状态
        self.is_running = False
        self.is_paused = False
        self.start_time = None
        self.frame_count = 0
        
        # 模块实例
        self.video_capture = None
        self.behavior_analyzer = None
        self.abnormal_detector = None
        self.product_recognizer = None
        self.inventory_manager = None
        self.data_reporter = None
        
        # 线程和锁
        self.processing_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # 性能统计
        self.processing_times = []
        self.average_fps = 0.0
        
        # 初始化模块
        self._initialize_modules()
        
        logger.info("售货机视觉分析系统初始化完成")
    
    def _initialize_modules(self):
        """初始化所有模块"""
        try:
            # 1. 视频捕获模块
            logger.info("初始化视频捕获模块...")
            self.video_capture = VideoCaptureManager(self.config.config)
            
            # 2. 行为分析模块
            if self.config.get("analysis.enable_behavior_analysis", True):
                logger.info("初始化行为分析模块...")
                self.behavior_analyzer = BehaviorAnalyzer(self.config.config)
            
            # 3. 异常检测模块
            if self.config.get("analysis.enable_abnormal_detection", True):
                logger.info("初始化异常检测模块...")
                self.abnormal_detector = AbnormalDetector(self.config.config)
            
            # 4. 商品识别模块
            if self.config.get("analysis.enable_product_recognition", True):
                logger.info("初始化商品识别模块...")
                self.product_recognizer = ProductRecognizer(self.config.config)
            
            # 5. 库存管理模块
            if self.config.get("analysis.enable_inventory_management", True):
                logger.info("初始化库存管理模块...")
                self.inventory_manager = InventoryManager(self.config.config)
            
            # 6. 数据报告模块
            logger.info("初始化数据报告模块...")
            self.data_reporter = DataReporter(self.config.config)
            
            logger.info("所有模块初始化完成")
            
        except Exception as e:
            logger.error(f"初始化模块失败: {e}")
            raise
    
    def start(self):
        """启动系统"""
        if self.is_running:
            logger.warning("系统已经在运行中")
            return
        
        logger.info("启动售货机视觉分析系统...")
        self.is_running = True
        self.start_time = time.time()
        self.stop_event.clear()
        
        # 启动处理线程
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        
        # 启动监控线程
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("系统启动完成")
        
        # 等待停止信号
        try:
            while self.is_running and not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止系统...")
            self.stop()
    
    def _processing_loop(self):
        """处理循环"""
        logger.info("开始处理循环")
        
        last_behavior_time = 0
        last_abnormal_time = 0
        last_product_time = 0
        last_inventory_time = 0
        
        # 获取配置间隔
        behavior_interval = self.config.get("analysis.detection_interval", 1.0)
        abnormal_interval = self.config.get("abnormal.check_interval", 2.0)
        product_interval = self.config.get("analysis.detection_interval", 1.0)
        inventory_interval = self.config.get("inventory.check_interval", 5.0)
        
        while self.is_running and not self.stop_event.is_set():
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                start_time = time.time()
                
                # 捕获视频帧
                frame, timestamp = self.video_capture.capture_frame()
                
                if frame is None:
                    logger.warning("无法捕获视频帧")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # 行为分析
                if self.behavior_analyzer and (current_time - last_behavior_time >= behavior_interval):
                    try:
                        behavior_results = self.behavior_analyzer.analyze(frame)
                        if behavior_results:
                            # 保存行为事件
                            for event in behavior_results:
                                self.data_reporter.save_behavior_event(event)
                            
                            # 发送警报（如果需要）
                            for event in behavior_results:
                                if event.get("requires_alert", False):
                                    self.data_reporter.send_alert({
                                        "type": "behavior_alert",
                                        "description": f"检测到异常行为: {event.get('action')}",
                                        "event": event
                                    }, severity="medium")
                    except Exception as e:
                        logger.error(f"行为分析失败: {e}")
                    
                    last_behavior_time = current_time
                
                # 异常检测
                if self.abnormal_detector and (current_time - last_abnormal_time >= abnormal_interval):
                    try:
                        abnormal_results = self.abnormal_detector.detect(frame)
                        if abnormal_results:
                            # 保存异常事件
                            for event in abnormal_results:
                                self.data_reporter.save_abnormal_event(event)
                            
                            # 发送警报
                            for event in abnormal_results:
                                severity = event.get("severity", "medium")
                                self.data_reporter.send_alert({
                                    "type": "abnormal_event",
                                    "description": event.get("description", "未知异常"),
                                    "event": event
                                }, severity=severity)
                    except Exception as e:
                        logger.error(f"异常检测失败: {e}")
                    
                    last_abnormal_time = current_time
                
                # 商品识别
                if self.product_recognizer and (current_time - last_product_time >= product_interval):
                    try:
                        product_results = self.product_recognizer.recognize(frame)
                        if product_results:
                            # 保存商品识别结果
                            for result in product_results:
                                self.data_reporter.save_product_recognition(result)
                    except Exception as e:
                        logger.error(f"商品识别失败: {e}")
                    
                    last_product_time = current_time
                
                # 库存管理
                if self.inventory_manager and (current_time - last_inventory_time >= inventory_interval):
                    try:
                        inventory_report = self.inventory_manager.check_inventory(frame)
                        if inventory_report:
                            # 保存库存状态
                            self.data_reporter.save_inventory_status(inventory_report)
                            
                            # 检查警报
                            for alert in inventory_report.get("alerts", []):
                                self.data_reporter.send_alert({
                                    "type": alert.get("type", "inventory_alert"),
                                    "description": alert.get("description", "库存警报"),
                                    "alert": alert
                                }, severity=alert.get("severity", "low"))
                    except Exception as e:
                        logger.error(f"库存检查失败: {e}")
                    
                    last_inventory_time = current_time
                
                # 可视化
                if self.config.get("visualization.enable_display", True):
                    self._visualize_frame(frame, {
                        "behavior": behavior_results if 'behavior_results' in locals() else None,
                        "abnormal": abnormal_results if 'abnormal_results' in locals() else None,
                        "product": product_results if 'product_results' in locals() else None,
                        "inventory": inventory_report if 'inventory_report' in locals() else None
                    })
                
                # 性能统计
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # 保留最近100个处理时间
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]
                
                # 计算平均FPS
                if len(self.processing_times) > 0:
                    avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                    self.average_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
                
                # 控制处理频率
                target_fps = self.config.get("video.fps", 30)
                target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.033
                
                if processing_time < target_frame_time:
                    time.sleep(target_frame_time - processing_time)
                
            except Exception as e:
                logger.error(f"处理循环错误: {e}")
                time.sleep(0.1)
        
        logger.info("处理循环结束")
    
    def _visualize_frame(self, frame, analysis_results: Dict):
        """可视化帧和分析结果"""
        try:
            vis_frame = frame.copy()
            
            # 添加基本信息
            if self.config.get("visualization.show_timestamp", True):
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(vis_frame, timestamp_str, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.config.get("visualization.show_fps", True):
                fps_str = f"FPS: {self.average_fps:.1f}"
                cv2.putText(vis_frame, fps_str, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 添加行为分析结果
            if self.config.get("visualization.show_detections", True) and analysis_results.get("behavior"):
                for event in analysis_results["behavior"]:
                    if "position" in event:
                        x, y = event["position"]
                        cv2.circle(vis_frame, (x, y), 10, (0, 255, 0), 2)
                        
                        label = f"{event.get('action', 'unknown')}"
                        cv2.putText(vis_frame, label, (x + 15, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 添加商品识别结果
            if self.config.get("visualization.show_detections", True) and analysis_results.get("product"):
                for result in analysis_results["product"]:
                    if "position" in result:
                        x, y = result["position"]
                        color = (255, 0, 0) if result.get("in_hand", False) else (0, 0, 255)
                        cv2.rectangle(vis_frame, (x - 20, y - 20), (x + 20, y + 20), color, 2)
                        
                        label = f"{result.get('product_name', 'unknown')}"
                        cv2.putText(vis_frame, label, (x - 50, y - 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 添加库存信息
            if self.config.get("visualization.show_inventory", True) and analysis_results.get("inventory"):
                vis_frame = self.inventory_manager.visualize_inventory(vis_frame, analysis_results["inventory"])
            
            # 显示帧
            if self.config.get("visualization.enable_display", True):
                display_resolution = self.config.get("visualization.display_resolution", [1280, 720])
                display_frame = cv2.resize(vis_frame, (display_resolution[0], display_resolution[1]))
                cv2.imshow("Vending Machine Vision System", display_frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                elif key == ord('p'):
                    self.toggle_pause()
                elif key == ord('s'):
                    self._save_screenshot(vis_frame)
            
            # 保存可视化结果
            if self.config.get("visualization.save_visualizations", False):
                self._save_visualization(vis_frame)
                
        except Exception as e:
            logger.error(f"可视化失败: {e}")
    
    def _save_screenshot(self, frame):
        """保存截图"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/screenshot_{timestamp}.jpg"
            
            import os
            os.makedirs("screenshots", exist_ok=True)
            
            cv2.imwrite(filename, frame)
            logger.info(f"截图已保存: {filename}")
            
        except Exception as e:
            logger.error(f"保存截图失败: {e}")
    
    def _save_visualization(self, frame):
        """保存可视化结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualizations/visualization_{timestamp}.jpg"
            
            import os
            os.makedirs("visualizations", exist_ok=True)
            
            cv2.imwrite(filename, frame)
            
        except Exception as e:
            logger.error(f"保存可视化结果失败: {e}")
    
    def _monitoring_loop(self):
        """监控循环"""
        logger.info("开始监控循环")
        
        last_status_time = 0
        last_report_time = 0
        
        while self.is_running and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # 更新系统状态
                if current_time - last_status_time >= 10.0:  # 每10秒更新一次状态
                    self._update_system_status()
                    last_status_time = current_time
                
                # 生成每日报告
                if self.config.get("reporting.enable_daily_reports", True):
                    report_time = self.config.get("reporting.report_generation_time", "23:59")
                    current_hour_minute = datetime.now().strftime("%H:%M")
                    
                    if current_hour_minute == report_time and current_time - last_report_time >= 86400:  # 每天一次
                        self._generate_daily_report()
                        last_report_time = current_time
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(1.0)
        
        logger.info("监控循环结束")
    
    def _update_system_status(self):
        """更新系统状态"""
        try:
            # 收集各模块状态
            status_data = {
                "system": {
                    "running_time": time.time() - self.start_time if self.start_time else 0,
                    "frame_count": self.frame_count,
                    "average_fps": self.average_fps,
                    "is_paused": self.is_paused,
                    "processing_times": {
                        "count": len(self.processing_times),
                        "average": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                        "min": min(self.processing_times) if self.processing_times else 0,
                        "max": max(self.processing_times) if self.processing_times else 0
                    }
                },
                "video_capture": self.video_capture.get_status() if self.video_capture else None,
                "behavior_analyzer": self.behavior_analyzer.get_status() if self.behavior_analyzer else None,
                "abnormal_detector": self.abnormal_detector.get_status() if self.abnormal_detector else None,
                "product_recognizer": self.product_recognizer.get_status() if self.product_recognizer else None,
                "inventory_manager": self.inventory_manager.get_status() if self.inventory_manager else None,
                "data_reporter": self.data_reporter.get_status() if self.data_reporter else None
            }
            
            # 保存系统状态
            if self.data_reporter:
                self.data_reporter.save_system_status("main_system", status_data)
            
            logger.debug(f"系统状态已更新: {status_data['system']['average_fps']:.1f} FPS")
            
        except Exception as e:
            logger.error(f"更新系统状态失败: {e}")
    
    def _generate_daily_report(self):
        """生成每日报告"""
        try:
            if self.data_reporter:
                report_data = self.data_reporter.generate_daily_report()
                self.data_reporter.send_daily_report(report_data)
                logger.info("每日报告已生成并发送")
                
        except Exception as e:
            logger.error(f"生成每日报告失败: {e}")
    
    def toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        status = "已暂停" if self.is_paused else "已恢复"
        logger.info(f"系统{status}")
    
    def stop(self):
        """停止系统"""
        if not self.is_running:
            logger.warning("系统未在运行中")
            return
        
        logger.info("正在停止系统...")
        self.is_running = False
        self.stop_event.set()
        
        # 等待处理线程结束
        time.sleep(0.5)
        
        # 释放资源
        self._cleanup()
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        
        logger.info("系统已停止")
    
    def _cleanup(self):
        """清理资源"""
        try:
            # 释放视频捕获
            if self.video_capture:
                self.video_capture.release()
            
            # 关闭数据报告器
            if self.data_reporter:
                self.data_reporter.close()
            
            logger.info("资源已清理")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        status = {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "running_time": time.time() - self.start_time if self.start_time else 0,
            "frame_count": self.frame_count,
            "average_fps": self.average_fps,
            "processing_times": {
                "count": len(self.processing_times),
                "average": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                "min": min(self.processing_times) if self.processing_times else 0,
                "max": max(self.processing_times) if self.processing_times else 0
            },
            "modules": {
                "video_capture": self.video_capture is not None,
                "behavior_analyzer": self.behavior_analyzer is not None,
                "abnormal_detector": self.abnormal_detector is not None,
                "product_recognizer": self.product_recognizer is not None,
                "inventory_manager": self.inventory_manager is not None,
                "data_reporter": self.data_reporter is not None
            }
        }
        
        return status
    
    def print_status(self):
        """打印系统状态"""
        status = self.get_status()
        
        print("=== 系统状态 ===")
        print(f"运行状态: {'运行中' if status['is_running'] else '已停止'}")
        print(f"暂停状态: {'已暂停' if status['is_paused'] else '运行中'}")
        print(f"运行时间: {status['running_time']:.1f} 秒")
        print(f"处理帧数: {status['frame_count']}")
        print(f"平均FPS: {status['average_fps']:.1f}")
        
        if status['processing_times']['count'] > 0:
            print(f"处理时间: {status['processing_times']['average']*1000:.1f}ms "
                  f"(最小: {status['processing_times']['min']*1000:.1f}ms, "
                  f"最大: {status['processing_times']['max']*1000:.1f}ms)")
        
        print("\n模块状态:")
        for module, enabled in status['modules'].items():
            print(f"  {module}: {'启用' if enabled else '禁用'}")


def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info(f"收到信号 {signum}，正在停止系统...")
    sys.exit(0)


def main():
    """主函数"""
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="售货机视觉分析系统")
    parser.add_argument("--config", "-c", default="config.json", help="配置文件路径")
    parser.add_argument("--test", "-t", action="store_true", help="测试模式")
    parser.add_argument("--no-display", action="store_true", help="不显示视频窗口")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # 创建系统实例
        system = VendingMachineVisionSystem(args.config)
        
        # 测试模式
        if args.test:
            logger.info("运行测试模式...")
            system._initialize_modules()
            
            # 测试各模块
            test_results = []
            
            # 测试视频捕获
            try:
                frame, timestamp = system.video_capture.capture_frame()
                if frame is not None:
                    test_results.append(("视频捕获", "通过", f"分辨率: {frame.shape[1]}x{frame.shape[0]}"))
                else:
                    test_results.append(("视频捕获", "失败", "无法捕获帧"))
            except Exception as e:
                test_results.append(("视频捕获", "失败", str(e)))
            
            # 测试行为分析
            if system.behavior_analyzer:
                try:
                    test_frame = cv2.imread("test_data/test_frame.jpg") if os.path.exists("test_data/test_frame.jpg") else frame
                    if test_frame is not None:
                        results = system.behavior_analyzer.analyze(test_frame)
                        test_results.append(("行为分析", "通过", f"检测到 {len(results) if results else 0} 个行为"))
                    else:
                        test_results.append(("行为分析", "跳过", "无测试帧"))
                except Exception as e:
                    test_results.append(("行为分析", "失败", str(e)))
            
            # 打印测试结果
            print("\n=== 测试结果 ===")
            for module, status, message in test_results:
                print(f"{module:15} [{status:4}] {message}")
            
            system.stop()
            return
        
        # 正常启动
        if args.no_display:
            system.config.set("visualization.enable_display", False)
        
        # 启动系统
        system.start()
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
