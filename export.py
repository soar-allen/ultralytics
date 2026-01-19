from ultralytics import YOLO

model = YOLO("./runs/obb/train2/weights/best.pt", task="obb")
path = model.export(format="onnx", simplify=True, device=0, opset=16, dynamic=False, imgsz=640)