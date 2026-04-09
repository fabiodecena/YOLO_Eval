import os
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Configuration ---
MODELS_TO_TEST = [
    {'path': 'runs/detect/flir_thermal_yolo11/weights/best.pt', 'name': 'YOLO11n'},
    {'path': 'runs/detect/flir_thermal_yolo12/weights/best.pt', 'name': 'YOLO12n'},
    {'path': 'runs/detect/flir_thermal_yolo26/weights/best.pt', 'name': 'YOLO26n'}
]

# Degradation Grids
NOISE_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
RES_LEVELS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

IMAGES_DIR = 'datasets/FLIR/images/test'
LABELS_DIR = 'datasets/FLIR/labels/test'
OUTPUT_CSV = 'LME_Ready_Data_Degraded.csv'


def get_ground_truth_count(label_path):
    if not os.path.exists(label_path):
        return 0
    with open(label_path, 'r') as f:
        return len(f.readlines())


def apply_gaussian_noise(image, sigma):
    if sigma == 0:
        return image
    noise = np.random.normal(0, sigma, image.shape).astype('float32')
    noisy_image = image.astype('float32') + noise
    return np.clip(noisy_image, 0, 255).astype('uint8')


def apply_res_degradation(image, scale):
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    low_res = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)


def run_unified_pipeline():
    data_log = []
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

    for model_cfg in MODELS_TO_TEST:
        print(f"--- Processing Model: {model_cfg['name']} ---")
        model = YOLO(model_cfg['path'])

        for img_file in image_files:
            img_path = os.path.join(IMAGES_DIR, img_file)
            lbl_path = os.path.join(LABELS_DIR, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

            # Read image once per image loop to save I/O time
            original_img = cv2.imread(img_path)
            if original_img is None: continue

            gt_count = get_ground_truth_count(lbl_path)
            seq_id = img_file.split('-')[1] if '-' in img_file else "unknown"

            # Test both degradation types
            for deg_type in ['noise', 'resolution']:
                levels = NOISE_LEVELS if deg_type == 'noise' else RES_LEVELS

                for level in levels:
                    # Apply specific degradation
                    if deg_type == 'noise':
                        processed_img = apply_gaussian_noise(original_img, level)
                        # Map 0-95 to 0.0-1.0
                        # Normalize: 0 is clean, 1.0 is max noise (95)
                        norm_stress = level / 95.0
                    else:
                        processed_img = apply_res_degradation(original_img, level)
                        # Map 1.0-0.05 to 0.0-1.0
                        # Normalize: 0 is clean (1.0), 1.0 is max degradation (0.05)
                        norm_stress = 1.0 - ((level - 0.05) / 0.95)

                    # Inference
                    start_time = time.perf_counter()
                    results = model(processed_img, verbose=False)
                    inf_ms = (time.perf_counter() - start_time) * 1000

                    pred_count = len(results[0].boxes)
                    conf_avg = float(results[0].boxes.conf.mean()) if pred_count > 0 else 0.0

                    # Stricter Success Metric: Sensitivity > 50%
                    if gt_count > 0:
                        sensitivity = pred_count / gt_count
                        success = 1 if sensitivity >= 0.5 else 0
                    else:
                        # For empty frames, success is not having false positives
                        success = 1 if pred_count == 0 else 0

                    data_log.append({
                        'Model': model_cfg['name'],
                        'Sequence': seq_id,
                        'Image': img_file,
                        'Degradation_Type': deg_type,
                        'Raw_Level': level,
                        'Normalized_Stress': norm_stress,
                        'Inference_ms': inf_ms,
                        'GT_Count': gt_count,
                        'Pred_Count': pred_count,
                        'Success': success,
                        'Confidence': conf_avg
                    })

    # Save to CSV
    df = pd.DataFrame(data_log)
    df.to_csv(OUTPUT_CSV)
    print(f"✅ Success! Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_unified_pipeline()