import os
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Configuration & Hyperparameters ---
MODELS_TO_TEST = [
    {'path': 'runs/detect/flir_thermal_yolo11/weights/best.pt', 'name': 'YOLO11n'},
    # {'path': 'runs/detect/yolo12/weights/best.pt', 'name': 'YOLO12n'},
    # {'path': 'runs/detect/yolo26/weights/best.pt', 'name': 'YOLO26n'}
]

# Degradation Grids
NOISE_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  # Sigma values for Gaussian Noise
RES_LEVELS = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]  # Scale factors (1.0 = Original)

IMAGES_DIR = 'datasets/FLIR/images/test'
LABELS_DIR = 'datasets/FLIR/labels/test'
OUTPUT_CSV = 'LME_Ready_Data_Degraded.csv'


# --- Utility Functions ---

def apply_gaussian_noise(image, sigma):
    """Adds synthetic thermal sensor noise."""
    if sigma == 0:
        return image
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy


def apply_resolution_reduction(image, scale):
    """Simulates spatial information loss (e.g., distance or low-res sensor)."""
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    low_res = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # Upscale back to original size so the model receives expected dimensions.3
    return cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)


def get_ground_truth_count(label_path):
    if not os.path.exists(label_path):
        return 0
    with open(label_path, 'r') as f:
        return len(f.readlines())


# --- Main Pipeline ---

def run_stress_test_pipeline():
    data_log = []

    for model_cfg in MODELS_TO_TEST:
        print(f"Loading Model: {model_cfg['name']}")
        model = YOLO(model_cfg['path'])

        # Warm-up (Ensures consistent latency measurements)
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            _ = model.predict(dummy_img, verbose=False)

        image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

        for deg_type in ['noise', 'resolution']:
            levels = NOISE_LEVELS if deg_type == 'noise' else RES_LEVELS

            for level in levels:
                print(f"Processing {model_cfg['name']} | {deg_type} Level: {level}")

                for img_file in image_files:
                    img_path = os.path.join(IMAGES_DIR, img_file)
                    lbl_path = os.path.join(LABELS_DIR, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                    # 1. Load and Degrade Image
                    raw_img = cv2.imread(img_path)
                    if deg_type == 'noise':
                        proc_img = apply_gaussian_noise(raw_img, level)
                    else:
                        proc_img = apply_resolution_reduction(raw_img, level)

                    # 2. Measure Latency (Inference Only)
                    start_time = time.perf_counter()
                    results = model.predict(source=proc_img, conf=0.45, verbose=False)
                    end_time = time.perf_counter()
                    inf_ms = (end_time - start_time) * 1000

                    # 3. Extract Metadata & Metrics
                    seq_id = img_file.split('-')[1] if '-' in img_file else "unknown"
                    gt_count = get_ground_truth_count(lbl_path)
                    pred_count = len(results[0].boxes)

                    # Binary Success: Did the model correctly identify presence/absence?
                    success = 1 if (gt_count > 0 and pred_count > 0) or (gt_count == 0 and pred_count == 0) else 0
                    conf_avg = float(results[0].boxes.conf.mean()) if pred_count > 0 else 0

                    # 4. Log Entry
                    data_log.append({
                        'Model': model_cfg['name'],
                        'Sequence': seq_id,
                        'Image': img_file,
                        'Degradation_Type': deg_type,
                        'Degradation_Level': level,
                        'Inference_ms': inf_ms,
                        'GT_Count': gt_count,
                        'Pred_Count': pred_count,
                        'Success': success,
                        'Confidence': conf_avg
                    })

    # Save Results
    df = pd.DataFrame(data_log)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Pipeline complete. Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_stress_test_pipeline()