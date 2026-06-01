# YOLO_Eval — B.Sc. Thesis Project

**A Regression-Based Analysis of Performance in Degraded Thermal Imagery for Anti-Poaching: Identifying Failure Thresholds in YOLOv11, v12, and 26 Architectures under Controlled Stress Conditions**

**Author:** Dr. Fabio De Cena 

**Institution:** International University of Applied Sciences (IU) 

**Degree:** Bachelor of Science in Software Development

---

## Project Overview

This repository contains the code and materials for a B.Sc. thesis project evaluating the operational limits of YOLO (You Only Look Once) object detection models in the context of nocturnal wildlife conservation. The deployment of automated surveillance systems, particularly unmanned aerial vehicles (UAVs) equipped with thermal infrared (TIR) computer vision, is a fundamental tool for protecting endangered species. However, conservation drones rely on resource-constrained edge devices and uncooled microbolometer cameras that are vulnerable to heat buildup and changing atmospheric conditions. 

This project investigates the architectural resilience of three nano-scale YOLO variants (YOLO11n, YOLO12n, YOLO26n) when subjected to simulated environmental stress. The experimental framework uses a Python-controlled pipeline to inject synthetic spatial downscaling (simulating altitude blur) and Additive White Gaussian Noise (AWGN, simulating thermal sensor drift) into the Teledyne FLIR Thermal dataset. Through comprehensive hardware profiling and statistical modeling (including Linear Mixed-Effects models and Dose-Response ED50 extraction), the research maps the degradation curves for Mean Average Precision (mAP), F1-score, and computational latency (ms) to provide some deployment hypotheses for edge computing systems.

---

## ⚠️ Data & Model Weights Notice

Due to standard version control size constraints, the raw **Teledyne FLIR ADAS Thermal Dataset**, the heavy database files, and the trained PyTorch model weights (`.pt` files) are **not included** in this repository. 

* **Dataset:** To fully replicate the data curation pipeline, please download the raw thermal dataset directly from the [Teledyne FLIR website](https://oem.flir.com/solutions/automotive/adas-dataset-form/).
* **Database Schema:** To illustrate the structure of the data logged by the Python pipeline and processed by the R scripts, below is a structural extract of the `LME_Ready_Data_Degraded.csv` master file:

| Test_ID | Model | Image | Sequence | Degradation_Type | Raw_Value | Normalized_Stress | Inference_ms | mAP_50 | Recall | F1_Score | APE | mAP_COCO | Avg_Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 187363 | YOLO26n | `video-4FRn...2DNSq.jpg` | 4FRnNpm... | noise | 15 | 0.157895 | 7.6459 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.251122 |
| 39708 | YOLO11n | `video-vbrS...HL5TW.jpg` | vbrSzr4... | resolution | 0.6 | 0.421053 | 7.5652 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.254594 |
| 222723 | YOLO26n | `video-vbrS...iCeza.jpg` | vbrSzr4... | noise | 15 | 0.157895 | 7.4443 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.263524 |

---

## Repository Structure

The project maintains the following directory structure to separate data, execution scripts, and statistical analysis outputs:

*   `database/` — Database schema for statistics and outputs (not committed)
*   `datasets/` — Curated FLIR thermal datasets and annotations (not committed)
*   `python_scripts/` — Runnable Python scripts for model training, synthetic distortion pipeline (AWGN and downscaling), and inference evaluation
*   `R_files/` — Runnable R scripts for hierarchical statistical modeling, regression analysis, and ED50 threshold extraction
*   `results/` — Results obtained from regression analysis, including output graphs (e.g., Pareto efficiency distribution) and tables
*   `runs/` — Results and metrics generated directly from YOLO inference runs

---

## Suggested Requirements for Replication

To successfully replicate the performance profiling and statistical analyses presented in the thesis, the following environment configuration is suggested based on the original experimental setup:

### Hardware Specifications
*   **OS:** Windows 11 Home 64-bit
*   **Processor:** Intel(R) Core(TM) i9-14900HX (2.20 GHz)
*   **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB dedicated VRAM)

### Python Environment (Computer Vision & Profiling)
* **Deep Learning Framework:** PyTorch version 2.5.1
* **CUDA Toolkit:** version 12.1
* **Required Python Packages:**
    * `ultralytics` (for YOLO architecture fine-tuning and inference)
    * `opencv-python` (cv2, for the synthetic degradation pipeline)
    * `pandas` and `numpy` (for data logging and manipulation)
    * `torchmetrics` (for COCO mAP evaluation)

### R Environment (Statistical Analysis)
* **Core:** R computing environment (v4.2+)
* **Required Packages:**
    * `lme4` and `lmerTest` (for fitting Linear Mixed-Effects models and extracting p-values)
    * `sjPlot` (for exporting publication-ready HTML regression tables)
    * `drc` (for dose-response modeling and ED50 extraction)
    * `tidyverse` (includes `dplyr` and `ggplot2` for data manipulation and visualization)
    * `scales` (for percentage formatting in plots)

---

## Evaluation Methodology Highlights

The evaluation framework independently tracks visual accuracy and hardware efficiency:
*   **Metrics:** Localization accuracy is measured using mAP@0.50:0.95, while classification reliability is tracked via the F1-score. Raw inference latency is measured in milliseconds (ms).
*   **Stress Normalization:** Degradation variables are mapped onto a unified 0.0 to 1.0 normalized stress spectrum for comparative analysis.
*   **Statistical Validation:** Models are cleaned of hardware anomalies using the Tukey Interquartile Range (IQR) method and evaluated using Akaike (AIC) and Bayesian (BIC) Information Criteria.

---

## Script Execution Workflow

The project is split into discrete execution scripts. To replicate the study, execute them in the following order:

### 1. Model Training
* **`train.py`**: Executes the transfer learning protocol (100 epochs, SGD optimizer, batch size) for the YOLO architectures.
* **`resume.py`**: Utility script to resume an interrupted training session directly from the `last.pt` checkpoint.

### 2. Inference and Profiling
* **`test.py`**: Runs a baseline validation on the clean test split to generate standard Ultralytics performance charts and a summary CSV.
* **`pipeline.py`**: The core synthetic stress pipeline. Iterates through predefined noise ($\sigma$ 0-95) and resolution (1.0-0.05) levels, applies them via OpenCV, tracks inference latency, and logs all metrics to a master CSV using `torchmetrics`.

### 3. Statistical Analysis (R)
Execute the R scripts to process the metrics generated by the Python pipeline. Each script handles a specific stage of the statistical workflow and exports its own tables or figures:

* **`AIC.R`**: Validates the mathematical shape of the degradation curves. It fits Linear, Quadratic, and Cubic mixed-effects models (using `REML = FALSE`) and compares them using Akaike (AIC) and Bayesian (BIC) Information Criteria to justify the polynomial degrees used in the final analysis.
* **`lme_latency.R`**: Handles hardware profiling. It applies Tukey's Interquartile Range (IQR) fence to remove non-parametric latency outliers, fits linear mixed-effects models for inference speed, and exports the results to `Latency_Comparison.html`.
* **`accuracy_analysis.R`**: Handles the core accuracy regressions and dose-response modeling. It fits the justified polynomial models (Quadratic for resolution, Cubic for noise) for mAP and F1-scores, exports the HTML regression tables, and uses the `drc` package to calculate the **ED50 Critical Failure Thresholds**.
* **`generate_plots.R`**: Consolidates the cleaned data to produce the final thesis visuals. It calculates the overall mean performance (Robustness AUC) and exports the final `png` graphics, including the Pareto Efficiency distribution and the comparative robustness bar charts.

---

## Citation

If you use this repository or its methodology in academic work, please cite the original thesis:

```bibtex
@bachelorthesis{DeCena2026,
  author  = {De Cena, Fabio},
  title   = {A Regression-Based Analysis of Performance in Degraded Thermal Imagery for Anti-Poaching: Identifying Failure Thresholds in YOLOv11, v12, and 26 Architectures under Controlled Stress Conditions},
  school  = {International University of Applied Sciences (IU)},
  year    = {2026},
  type    = {Bachelor's Thesis}
}
```

## Contact
For questions related to the thesis project or the codebase:

* **GitHub**: @fabiodecena

* **email**: fabiodecena@gmail.com