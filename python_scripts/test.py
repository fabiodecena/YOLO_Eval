from ultralytics import YOLO
import pandas as pd

def run_test():
    # Load the model (Fusion happens here)
    model = YOLO(r'/runs/detect/flir_thermal_yolo11/weights/best.pt')

    # Run Validation on the TEST split
    # This will generate the visual charts and the stats
    metrics = model.val(
        data='data.yaml',
        split='test',
        imgsz=640,
        device=0,
        name='flir_final_test_report_yolo11',
        save=True         # Saves sample prediction images
    )

    results_data = {
        'Model': ['YOLO11n'],
        'mAP50': [metrics.results_dict['metrics/mAP50(B)']],
        'Recall': [metrics.results_dict['metrics/recall(B)']],
        'Precision': [metrics.results_dict['metrics/precision(B)']],
        'Inference_ms': [metrics.speed['inference']]
    }

    # Create CSV
    df = pd.DataFrame(results_data)
    df.to_csv(r'C:\Users\fabio\PycharmProjects\YOLO_Eval\runs\detect\flir_final_test_report\test_summary.csv',
              index=False)
    print("Manual Test CSV created!")

if __name__ == '__main__':
    # This block is MANDATORY on Windows to prevent the RuntimeError
    run_test()