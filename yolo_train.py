from ultralytics import YOLO


"""
yolov8n.pt -> the smallest training model
yolov8s.pt
yolov8m.pt
"""
model = YOLO('yolov8s.pt')

model.train(
    data = 'object_classes.yaml',
    epochs = 30, # ??
    imgsz = (1280, 720), # (width, height)
    batch = 16
)