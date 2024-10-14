from ultralytics import YOLO
"""
yolov8n-pose.pt 
yolov8s-pose.pt 
yolov8m-pose.pt 
yolov8l-pose.pt 
yolov8x-pose.pt 
yolov8x-pose-p6.pt

tensorboard --logdir runs/pose/train6

"""
model = YOLO('yolov8x-pose.pt')

results = model.train(data='./test_dataset/data.yaml', epochs=100, imgsz=640, device='mps', batch=2, mosaic=0.0, plots=True)