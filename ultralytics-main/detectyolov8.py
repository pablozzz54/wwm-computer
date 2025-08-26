import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

# 加载模型 (使用更小的模型可以提高速度)
model = YOLO(r"D:\yolov8\ultralytics-main\best (4).onnx")  # 确保这是适合你需求的最小模型

# 深度相机配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# 禁用不需要的流以提升性能
config.disable_stream(rs.stream.infrared)  # 如果不需要红外流

pipe_profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# 预热摄像头
for _ in range(10):
    pipeline.wait_for_frames()


def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # 直接使用numpy数组，减少转换
    color_image = np.asanyarray(color_frame.get_data())
    return depth_frame, color_image  # 只返回必要的数据


if __name__ == '__main__':
    try:
        prev_time = time.time()
        fps_counter = 0
        fps = 0
        fps_update_interval = 1  # 每秒更新一次FPS显示

        while True:
            start_time = time.time()

            # 获取帧数据
            depth_frame, color_image = get_aligned_images()

            # YOLO推理优化
            results = model.predict(
                color_image,
                imgsz=640,  # 使用更小的推理尺寸
                conf=0.5,  # 提高置信度阈值减少检测数量
                device='CPU',  # 明确使用GPU
                half=False,  # 使用半精度浮点(如果GPU支持)
                verbose=False,  # 禁用详细输出
                stream=True  # 使用流模式
            )

            # 处理结果
            for result in results:
                im_array = result.plot()  # 获取带标注的图像

                # 只处理mouth_close和mouth_open
                boxes = result.boxes
                for box, cls in zip(boxes.xywh, boxes.cls):
                    class_name = model.names[int(cls)]
                    if class_name in ["mouth_close", "mouth_open"]:
                        ux, uy = int(box[0]), int(box[1])
                        dis = depth_frame.get_distance(ux, uy)
                        if dis > 0:  # 只处理有效深度
                            camera_xyz = rs.rs2_deproject_pixel_to_point(
                                depth_frame.profile.as_video_stream_profile().intrinsics,
                                (ux, uy), dis)
                            camera_xyz = np.round(np.array(camera_xyz) * 1000, 3)
                            print(f"Detected {class_name} at coordinates: {camera_xyz}")

                            # 标注
                            cv2.circle(im_array, (ux, uy), 4, (255, 255, 255), 2)
                            cv2.putText(im_array, str(camera_xyz), (ux + 10, uy + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (225, 255, 255), 1)

            # FPS计算
            fps_counter += 1
            if (time.time() - prev_time) > fps_update_interval:
                fps = fps_counter / (time.time() - prev_time)
                fps_counter = 0
                prev_time = time.time()

            # 显示优化 - 只在图像上绘制必要的元素
            if fps > 0:
                cv2.putText(im_array, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示
            cv2.imshow('RealSense', im_array)

            # 处理按键 - 使用更高效的检查方式
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()