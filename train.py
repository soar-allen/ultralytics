from ultralytics import YOLO

# Load a model
# model = YOLO("yolo26n-obb.yaml")  # build a new model from YAML
model = YOLO("yolo26s-obb.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo26n-obb.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/home/cotek/datasets/yolo_obb_dataset/dataset.yaml", epochs=100, imgsz=640)