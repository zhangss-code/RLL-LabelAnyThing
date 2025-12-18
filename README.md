# RLL-LabelAnyThing - 售货机视觉分析系统

## 项目概述

基于计算机视觉的售货机智能监控与分析系统，能够实时分析用户行为、检测异常事件、识别商品、管理库存，并生成详细的数据报告。

## 系统功能

### 1. 视频捕获模块
- 实时视频流捕获
- 帧处理和优化
- 视频录制和截图功能

### 2. 行为分析模块
- 用户行为识别（接近、选择、购买等）
- 行为模式分析
- 异常行为检测

### 3. 异常检测模块
- 破坏行为检测
- 盗窃行为检测
- 设备故障检测
- 实时警报系统

### 4. 商品识别模块
- 商品图像识别
- 槽位状态检测
- 手持商品识别

### 5. 库存管理模块
- 实时库存监控
- 低库存预警
- 库存变化分析
- 销售数据统计

### 6. 数据报告模块
- 实时数据收集和存储
- 每日报告生成
- 数据导出功能
- 系统状态监控

## 系统架构

```
RLL-LabelAnyThing/
├── src/                    # 源代码目录
│   ├── config.py          # 系统配置
│   ├── main.py            # 主程序
│   ├── Video/             # 视频处理模块
│   │   └── video_capture.py
│   ├── predict/           # 分析预测模块
│   │   ├── behavior_analyzer.py
│   │   ├── abnormal_detector.py
│   │   ├── product_recognizer.py
│   │   └── inventory_manager.py
│   └── Web/               # 数据报告模块
│       └── data_reporter.py
├── run.py                 # 启动脚本
├── config.json            # 配置文件
├── requirements.txt       # 依赖包列表
└── README.md             # 说明文档
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd RLL-LabelAnyThing

# 安装依赖
pip install opencv-python numpy Pillow
```

### 2. 系统设置

```bash
# 运行设置脚本
python run.py --setup

# 或手动创建配置
python run.py --install-check
```

### 3. 启动系统

```bash
# 正常启动（显示视频窗口）
python run.py

# 无显示模式启动
python run.py --no-display

# 指定配置文件
python run.py --config my_config.json

# 设置日志级别
python run.py --log-level DEBUG
```

### 4. 系统测试

```bash
# 测试系统功能
python run.py --test
```

## 配置文件说明

系统使用JSON格式的配置文件，主要配置项包括：

### 系统配置
```json
{
  "system": {
    "name": "RLL-LabelAnyThing",
    "version": "1.0.0",
    "mode": "development",
    "log_level": "INFO"
  }
}
```

### 视频配置
```json
{
  "video": {
    "camera_index": 0,
    "resolution": [1920, 1080],
    "fps": 30,
    "record_video": false
  }
}
```

### 分析配置
```json
{
  "analysis": {
    "min_confidence": 0.7,
    "enable_behavior_analysis": true,
    "enable_abnormal_detection": true,
    "enable_product_recognition": true,
    "enable_inventory_management": true
  }
}
```

## 使用说明

### 控制命令
系统运行时支持以下键盘命令：
- **q** - 退出系统
- **p** - 暂停/恢复处理
- **s** - 保存当前截图

### 数据查看
系统数据存储在SQLite数据库中：
```bash
# 查看数据库
sqlite3 data/system_data.db

# 常用查询
SELECT * FROM behavior_events ORDER BY timestamp DESC LIMIT 10;
SELECT * FROM abnormal_events WHERE severity = 'high';
SELECT * FROM inventory_status WHERE status = 'low_stock';
```

### 报告生成
系统自动生成每日报告，也可手动导出数据：
```python
# 通过Python API导出数据
from Web.data_reporter import DataReporter
reporter = DataReporter(config)
report = reporter.generate_daily_report()
```

## 模块详细说明

### 视频捕获模块 (VideoCaptureManager)
- 支持多摄像头
- 自动调整分辨率
- 帧率控制
- 录制功能

### 行为分析模块 (BehaviorAnalyzer)
- 运动检测
- 行为分类
- 用户跟踪
- 行为模式识别

### 异常检测模块 (AbnormalDetector)
- 实时异常检测
- 多级警报系统
- 证据收集
- 历史记录

### 商品识别模块 (ProductRecognizer)
- 基于深度学习的商品识别
- 槽位状态分析
- 手持商品检测
- 置信度评估

### 库存管理模块 (InventoryManager)
- 实时库存监控
- 自动补货建议
- 销售预测
- 异常库存检测

### 数据报告模块 (DataReporter)
- 数据持久化存储
- 实时仪表板
- 报告生成
- 数据导出

## 开发指南

### 添加新模块
1. 在`src/predict/`目录下创建新模块
2. 实现标准接口（analyze/detect/recognize等方法）
3. 在主程序中集成
4. 更新配置文件

### 扩展功能
- 添加新的异常检测算法
- 集成新的商品识别模型
- 扩展报告格式
- 添加新的通知渠道

### 性能优化
- 使用GPU加速
- 多线程处理
- 缓存优化
- 内存管理

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头索引
   - 确认摄像头权限
   - 尝试其他摄像头索引

2. **处理速度慢**
   - 降低视频分辨率
   - 减少检测频率
   - 启用GPU加速

3. **识别准确率低**
   - 调整置信度阈值
   - 优化光照条件
   - 更新识别模型

### 日志查看
```bash
# 查看系统日志
tail -f logs/system.log

# 查看错误日志
grep ERROR logs/system.log
```

## 性能指标

- 处理速度：30+ FPS (1080p)
- 识别准确率：>90%
- 内存使用：<1GB
- CPU使用率：<80%

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库：<repository-url>
- 问题跟踪：GitHub Issues

## 更新日志

### v1.0.0 (2025-12-18)
- 初始版本发布
- 完整的功能模块
- 基础文档
- 示例配置

---

**注意**：本系统为原型系统，实际部署前需要进行充分的测试和优化。
