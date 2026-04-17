from ultralytics import YOLO


def resume_training():
    # Point DIRECTLY to the 'last.pt'
    # This file contains the architecture, weights, and the epoch counter
    model_path = r'/runs/detect/flir_thermal_yolo11/weights/last.pt'

    model = YOLO(model_path)

    # Resume the training
    model.train(resume=True)


if __name__ == "__main__":
    resume_training()