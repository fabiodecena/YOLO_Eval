import torch
from ultralytics import YOLO

def run_transfer_learning():
    # 1. Load the model
    model = YOLO('yolo11n.pt')

    # 2. Automatically select device
    # Uses GPU (0) if available, otherwise falls back to CPU
    current_device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {current_device}")

    # 3. Run Training
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16 if current_device == 0 else 4, # Smaller batch for CPU
        name='flir_thermal_yolo11',
        device=current_device,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        augment=True
    )

if __name__ == "__main__":
    run_transfer_learning()