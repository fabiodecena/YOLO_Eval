library(tidyverse)
library(lme4)
library(lmerTest)
library(sjPlot)

# 1. Load Data
data <- read.csv("LME_Ready_Data_Degraded.csv", sep = ";")

# Clean pandas index column if present
if ("Unnamed..0" %in% colnames(data)) {
  data <- data %>% select(-Unnamed..0)
}

# Focus only on frames with actual targets to measure true NMS latency
data_objects_only <- data %>%
  filter(mAP_50 >= 0)

# ==========================================
# 2. ADVANCED OUTLIER REMOVAL: Tukey's IQR Method
# (This is suitable as non-parametric approach!!!)
# ==========================================
data_clean <- data_objects_only %>%
  group_by(Model, Degradation_Type, Normalized_Stress) %>%
  mutate(
    Q1 = quantile(Inference_ms, 0.25, na.rm = TRUE),
    Q3 = quantile(Inference_ms, 0.75, na.rm = TRUE),
    IQR_value = Q3 - Q1,
    Upper_Fence = Q3 + (1.5 * IQR_value),
    Lower_Fence = Q1 - (1.5 * IQR_value)
  ) %>%
  filter(Inference_ms >= Lower_Fence & Inference_ms <= Upper_Fence) %>%
  ungroup() %>%
  select(-Q1, -Q3, -IQR_value, -Upper_Fence, -Lower_Fence)

# Set YOLO11n as the Baseline
data_clean$Model <- factor(data_clean$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))

# 3. SEPARATE the data
data_noise <- data_clean %>% filter(Degradation_Type == "noise")
data_res <- data_clean %>% filter(Degradation_Type == "resolution")

# 4. Run the two separated models for Inference Time
reg_latency_noise <- lmer(Inference_ms ~ Model * Normalized_Stress + (1 | Image), data = data_noise)
reg_latency_res <- lmer(Inference_ms ~ Model * Normalized_Stress + (1 | Image), data = data_res)

# 5. Generate the side-by-side Thesis Table
tab_model(reg_latency_noise, reg_latency_res,
          dv.labels = c("Latency: Noise Degradation", "Latency: Resolution Degradation"),
          title = "Mixed-Effects Regression on Inference Time (ms) - IQR Cleaned",
          file = "Latency_Separated_Results.html")