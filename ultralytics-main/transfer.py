from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO('best (4).pt')

# 导出为 ONNX 格式
model.export(format='onnx')