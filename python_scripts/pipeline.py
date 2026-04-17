import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# --- Configuration ---
MODELS_TO_TEST = [
    {'path': 'runs/detect/flir_thermal_yolo11/weights/best.pt', 'name': 'YOLO11n'},
    {'path': 'runs/detect/flir_thermal_yolo12/weights/best.pt', 'name': 'YOLO12n'},
    {'path': 'runs/detect/flir_thermal_yolo26/weights/best.pt', 'name': 'YOLO26n'}
]

NOISE_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
RES_LEVELS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15,
              0.10, 0.05]

IMAGES_DIR = '../datasets/FLIR/images/test'
LABELS_DIR = '../datasets/FLIR/labels/test'
OUTPUT_CSV = 'LME_Ready_Data_Degraded.csv'


def load_gt_for_torchmetrics(label_path, img_w, img_h, device):
    """Loads YOLO labels and converts to absolute xyxy tensors."""
    if not os.path.exists(label_path):
        return torch.empty((0, 4), device=device), torch.empty(0, dtype=torch.int64, device=device)

    boxes, labels = [], []
    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.split())
            boxes.append([x, y, w, h])
            labels.append(int(cls))

    if not boxes:
        return torch.empty((0, 4), device=device), torch.empty(0, dtype=torch.int64, device=device)

    boxes_tensor = torch.tensor(boxes, device=device)
    boxes_abs = ops.xywhn2xyxy(boxes_tensor, img_w, img_h)
    return boxes_abs, torch.tensor(labels, dtype=torch.int64, device=device)


def apply_gaussian_noise(image, sigma):
    if sigma == 0: return image
    noise = np.random.normal(0, sigma, image.shape).astype('float32')
    return np.clip(image.astype('float32') + noise, 0, 255).astype('uint8')


def apply_res_degradation(image, scale):
    if scale == 1.0: return image
    h, w = image.shape[:2]
    low_res = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)


def run_unified_pipeline():
    data_log = []
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize metric with extended summary for precision/recall extraction
    metric = MeanAveragePrecision(iou_type="bbox").to(device)

    for model_cfg in MODELS_TO_TEST:
        print(f"--- Processing Model: {model_cfg['name']} ---")
        model = YOLO(model_cfg['path']).to(device)

        for img_file in image_files:
            img_path = os.path.join(IMAGES_DIR, img_file)
            lbl_path = os.path.join(LABELS_DIR, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

            original_img = cv2.imread(img_path)
            if original_img is None: continue
            h, w, _ = original_img.shape

            gt_boxes, gt_labels = load_gt_for_torchmetrics(lbl_path, w, h, device)
            # --- THE SPEED OPTIMIZATION ---
            # If the label file is empty or doesn't exist, gt_labels will have length 0.
            # Skip everything below (Noise, Resolution, Inference) for this image.
            if len(gt_labels) == 0:
                continue
            seq_id = img_file.split('-')[1] if '-' in img_file else "unknown"

            for deg_type in ['noise', 'resolution']:
                levels = NOISE_LEVELS if deg_type == 'noise' else RES_LEVELS

                for level in levels:
                    if deg_type == 'noise':
                        processed_img = apply_gaussian_noise(original_img, level)
                        norm_stress = level / 95.0
                    else:
                        processed_img = apply_res_degradation(original_img, level)
                        norm_stress = 1.0 - ((level - 0.05) / 0.95)

                    # Inference
                    start_time = time.perf_counter()
                    results = model(processed_img, verbose=False)
                    inf_ms = (time.perf_counter() - start_time) * 1000

                    pred_boxes = results[0].boxes.xyxy
                    pred_scores = results[0].boxes.conf
                    pred_labels = results[0].boxes.cls.to(torch.int64)

                    # Calculate Metrics
                    metric.update(
                        [dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels)],
                        [dict(boxes=gt_boxes, labels=gt_labels)]
                    )
                    res = metric.compute()
                    metric.reset()

                    # Extracting Core Metrics
                    # mAP@50 is the standard proxy for Precision in many detection papers
                    m_precision = float(res['map_50'].cpu())
                    # mar_100 is the Max Recall given 100 detections
                    m_recall = float(res['mar_100'].cpu())

                    # F1 Score calculation
                    if (m_precision + m_recall) > 0:
                        f1 = 2 * (m_precision * m_recall) / (m_precision + m_recall)
                    else:
                        f1 = 0.0

                    # Calculate Absolute Percentage Error (APE) for specific image
                    gt_count = len(gt_labels)
                    pred_count = len(pred_labels)
                    count_error = abs(gt_count - pred_count) / gt_count if gt_count > 0 else 0.0

                    data_log.append({
                        'Model': model_cfg['name'],
                        'Image': img_file,
                        'Sequence': seq_id,
                        'Degradation_Type': deg_type,
                        'Raw_Value': level,
                        'Normalized_Stress': norm_stress,
                        'Inference_ms': inf_ms,
                        'mAP_50': m_precision,
                        'Recall': m_recall,
                        'F1_Score': f1,
                        'APE': count_error,
                        'mAP_COCO': float(res['map'].cpu()),
                        'Avg_Confidence': float(pred_scores.mean().cpu()) if len(pred_scores) > 0 else 0.0
                    })

    df = pd.DataFrame(data_log)
    df.index.name = 'Test_ID'
    df.to_csv(OUTPUT_CSV, index=True)
    print(f"✅ Success! Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_unified_pipeline()