from ultralytics import YOLO
from IPython.display import Image

def train_model(epochs,image_size,seed,device):
    print("-----Start training...")

    # 載入YOLO模型
    model = YOLO('yolov8x.pt')

    # 開始訓練，依需求調整 epochs、影像尺寸
    model.train(data='./dataset/data.yaml', epochs=epochs, imgsz=image_size, seed=seed, device=device)

    # Image(filename='runs/detect/train/results.png')

    print(f"-----Training completed, epochs={epochs}, image_size={image_size}")