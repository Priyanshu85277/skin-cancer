import torch
from ultralytics import YOLO
model = YOLO('best.pt')  # load a pretrained model (recommended for training)


# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model.predict(source='test/images/-4-R-_jpg.rf.d67bb5087e6c291ed7239e9cf182608a.jpg',conf=0.70,save=True)
print(results)
