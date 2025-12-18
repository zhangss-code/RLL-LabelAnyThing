#!/usr/bin/env python3
"""
视频捕获模块
实现摄像头视频流的捕获和管理
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import threading

logger = logging.getLogger(__name__)

class VideoCaptureManager:
    """视频捕获管理器类"""
    
    def __init__(self, config: Dict):
        """
        初始化视频捕获管理器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.video_config = config.get("video", {})
        
        # 摄像头配置
        self.camera_index = self.video_config.get("camera_index", 0)
        self.resolution = self.video_config.get("resolution", [1920, 1080])
        self.fps = self.video_config.get("fps", 30)
        self.recording_duration = self.video_config.get("recording_duration", 10)
        
        # 摄像头状态
        self.camera = None
        self.is_camera_open = False
        self.last_frame = None
        self.last_timestamp = None
        self.frame_count = 0
        
        # 录制状态
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.recorded_videos = []
        
        # 性能统计
        self.frame_times = []
        self.average_fps = 0.0
        
        # 初始化摄像头
        self._initialize_camera()
        
        logger.info(f"视频捕获管理器初始化完成，摄像头: {self.camera_index}, 分辨率: {self.resolution}, FPS: {self.fps}")
    
    def _initialize_camera(self):
        """初始化摄像头"""
        try:
            # 尝试打开摄像头
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self.camera.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_index}")
                return
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 验证设置
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"摄像头实际参数: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            self.is_camera_open = True
            
            # 预热：读取几帧
            for _ in range(5):
                ret, _ = self.camera.read()
                if not ret:
                    logger.warning("摄像头预热失败")
                    break
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            self.is_camera_open = False
    
    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        捕获一帧视频
        
        Returns:
            (帧图像, 时间戳) 或 (None, None) 如果失败
        """
        if not self.is_camera_open or self.camera is None:
            logger.warning("摄像头未打开，无法捕获帧")
            return None, None
        
        try:
            start_time = time.time()
            
            # 读取帧
            ret, frame = self.camera.read()
            
            if not ret:
                logger.error("读取摄像头帧失败")
                return None, None
            
            # 更新时间戳
            timestamp = datetime.now()
            
            # 更新帧统计
            self.frame_count += 1
            self.last_frame = frame.copy()
            self.last_timestamp = timestamp
            
            # 计算帧时间
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            
            # 保留最近100个帧时间
            if len(self.frame_times) > 100:
                self.frame_times = self.frame_times[-100:]
            
            # 计算平均FPS
            if len(self.frame_times) > 0:
                avg_frame_time = np.mean(self.frame_times)
                self.average_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            
            # 检查是否需要开始录制
            if self.video_config.get("record_video", False) and not self.is_recording:
                self._start_recording(frame)
            
            # 如果正在录制，写入帧
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)
                
                # 检查录制时长
                if time.time() - self.recording_start_time >= self.recording_duration:
                    self._stop_recording()
            
            return frame, timestamp
            
        except Exception as e:
            logger.error(f"捕获帧时发生错误: {e}")
            return None, None
    
    def _start_recording(self, first_frame: np.ndarray):
        """开始录制视频"""
        try:
            # 创建文件名
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"recordings/video_{timestamp_str}.avi"
            
            # 确保目录存在
            import os
            os.makedirs("recordings", exist_ok=True)
            
            # 获取视频编码器和参数
            frame_height, frame_width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            # 创建VideoWriter
            self.video_writer = cv2.VideoWriter(
                video_filename,
                fourcc,
                self.fps,
                (frame_width, frame_height)
            )
            
            if not self.video_writer.isOpened():
                logger.error(f"无法创建视频文件: {video_filename}")
                self.video_writer = None
                return
            
            self.is_recording = True
            self.recording_start_time = time.time()
            
            logger.info(f"开始录制视频: {video_filename}")
            
        except Exception as e:
            logger.error(f"开始录制失败: {e}")
            self.is_recording = False
            self.video_writer = None
    
    def _stop_recording(self):
        """停止录制视频"""
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                
                # 记录视频信息
                video_info = {
                    "filename": "recordings/video_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi",
                    "start_time": self.recording_start_time,
                    "duration": time.time() - self.recording_start_time,
                    "frame_count": self.frame_count
                }
                self.recorded_videos.append(video_info)
                
                logger.info(f"停止录制视频，时长: {video_info['duration']:.2f}秒")
                
            except Exception as e:
                logger.error(f"停止录制时发生错误: {e}")
            
            finally:
                self.video_writer = None
                self.is_recording = False
                self.recording_start_time = None
    
    def record_frame(self, frame: np.ndarray, timestamp: datetime):
        """记录单帧到视频文件"""
        if not self.is_recording:
            self._start_recording(frame)
        
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
    
    def get_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """获取最新的帧和时间戳"""
        return self.last_frame, self.last_timestamp
    
    def capture_burst(self, num_frames: int = 10, interval: float = 0.1) -> List[Tuple[np.ndarray, datetime]]:
        """
        捕获连续多帧
        
        Args:
            num_frames: 帧数
            interval: 帧间隔（秒）
            
        Returns:
            帧和时间戳列表
        """
        frames = []
        
        for i in range(num_frames):
            frame, timestamp = self.capture_frame()
            
            if frame is not None:
                frames.append((frame, timestamp))
            
            # 等待间隔
            time.sleep(interval)
        
        return frames
    
    def capture_with_retry(self, max_retries: int = 3) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        带重试的帧捕获
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            (帧图像, 时间戳)
        """
        for attempt in range(max_retries):
            frame, timestamp = self.capture_frame()
            
            if frame is not None:
                return frame, timestamp
            
            logger.warning(f"帧捕获失败，重试 {attempt + 1}/{max_retries}")
            time.sleep(0.1)
        
        logger.error(f"帧捕获失败，已达到最大重试次数 {max_retries}")
        return None, None
    
    def adjust_exposure(self, exposure_value: float = 0.5):
        """调整摄像头曝光"""
        if self.camera is not None and self.is_camera_open:
            try:
                # 设置曝光（值范围通常为0-1）
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
                logger.info(f"调整曝光为: {exposure_value}")
            except Exception as e:
                logger.error(f"调整曝光失败: {e}")
    
    def adjust_brightness(self, brightness_value: float = 0.5):
        """调整摄像头亮度"""
        if self.camera is not None and self.is_camera_open:
            try:
                # 设置亮度（值范围通常为0-1）
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
                logger.info(f"调整亮度为: {brightness_value}")
            except Exception as e:
                logger.error(f"调整亮度失败: {e}")
    
    def get_camera_info(self) -> Dict:
        """获取摄像头信息"""
        if self.camera is None or not self.is_camera_open:
            return {"status": "camera_not_available"}
        
        try:
            info = {
                "status": "active",
                "camera_index": self.camera_index,
                "resolution": {
                    "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                },
                "fps": self.camera.get(cv2.CAP_PROP_FPS),
                "frame_count": self.frame_count,
                "average_fps": self.average_fps,
                "is_recording": self.is_recording,
                "recorded_videos_count": len(self.recorded_videos),
                "brightness": self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.camera.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self.camera.get(cv2.CAP_PROP_SATURATION),
                "exposure": self.camera.get(cv2.CAP_PROP_EXPOSURE)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取摄像头信息失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_status(self) -> Dict:
        """获取捕获器状态"""
        camera_info = self.get_camera_info()
        
        status = {
            "camera_info": camera_info,
            "last_frame_time": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "frame_times_stats": {
                "count": len(self.frame_times),
                "average": np.mean(self.frame_times) if self.frame_times else 0,
                "min": np.min(self.frame_times) if self.frame_times else 0,
                "max": np.max(self.frame_times) if self.frame_times else 0
            },
            "recording": {
                "is_recording": self.is_recording,
                "start_time": self.recording_start_time,
                "duration": time.time() - self.recording_start_time if self.recording_start_time else 0
            }
        }
        
        return status
    
    def release(self):
        """释放摄像头资源"""
        try:
            # 停止录制
            if self.is_recording:
                self._stop_recording()
            
            # 释放摄像头
            if self.camera is not None:
                self.camera.release()
                self.is_camera_open = False
                logger.info("摄像头已释放")
            
            # 释放VideoWriter
            if self.video_writer is not None:
                self.video_writer.release()
            
        except Exception as e:
            logger.error(f"释放资源时发生错误: {e}")
        
        finally:
            self.camera = None
            self.video_writer = None
            self.is_camera_open = False
            self.is_recording = False
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.release()
    
    def visualize_frame_info(self, frame: np.ndarray) -> np.ndarray:
        """在帧上可视化捕获信息"""
        if frame is None:
            return None
        
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # 添加时间戳
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(vis_frame, timestamp_str, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加帧计数
        frame_count_str = f"Frame: {self.frame_count}"
        cv2.putText(vis_frame, frame_count_str, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加FPS信息
        fps_str = f"FPS: {self.average_fps:.1f}"
        cv2.putText(vis_frame, fps_str, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加录制状态
        if self.is_recording:
            record_str = "REC"
            cv2.putText(vis_frame, record_str, (width - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # 添加录制时间
            if self.recording_start_time:
                record_time = time.time() - self.recording_start_time
                time_str = f"{record_time:.1f}s"
                cv2.putText(vis_frame, time_str, (width - 100, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 添加分辨率信息
        res_str = f"{width}x{height}"
        cv2.putText(vis_frame, res_str, (width - 150, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return vis_frame
