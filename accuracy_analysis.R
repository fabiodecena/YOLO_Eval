library(tidyverse)
library(sjPlot)
library(drc)

# 1. LOAD AND PREP
data <- read.csv("LME_Ready_Data_Degraded.csv", sep = ";")
data_valid <- data %>% filter(mAP_50 >= 0)
data_valid$Model <- factor(data_valid$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))

# --- A. RESOLUTION ANALYSIS (POLYNOMIAL) ---
res_data <- data_valid %>% filter(Degradation_Type == "resolution")

# Model for mAP
reg_res_map <- lm(mAP_COCO ~ Model * poly(Normalized_Stress, 2), data = res_data)
# Model for F1
reg_res_f1 <- lm(F1_Score ~ Model * poly(Normalized_Stress, 2), data = res_data)

tab_model(reg_res_map, reg_res_f1,
          dv.labels = c("mAP_COCO (Resolution)", "F1-Score (Resolution)"),
          title = "Table: Resolution Decay Comparison",
          file = "Resolution_mAP_vs_F1.html")


# --- B. NOISE ANALYSIS (CUBIC) ---
noise_data <- data_valid %>% filter(Degradation_Type == "noise")

# Model for mAP
reg_noise_map <- lm(mAP_COCO ~ Model * poly(Normalized_Stress, 3), data = noise_data)
# Model for F1
reg_noise_f1 <- lm(F1_Score ~ Model * poly(Normalized_Stress, 3), data = noise_data)

tab_model(reg_noise_map, reg_noise_f1,
          dv.labels = c("mAP_COCO (Noise)", "F1-Score (Noise)"),
          title = "Table: Noise Decay Comparison (Cubic)",
          file = "Noise_mAP_vs_F1.html")


# --- C. NOISE FAILURE THRESHOLDS (ED50) ---
# Function to get ED50 for a specific metric
get_ed50 <- function(metric_name) {
  df_agg <- noise_data %>%
    group_by(Model, Normalized_Stress) %>%
    summarize(m = mean(get(metric_name)), .groups = 'drop')

  fit <- drm(m ~ Normalized_Stress, curveid = Model, data = df_agg, fct = L.4())
  res <- as.data.frame(ED(fit, 50, interval = "delta"))
  res$Metric <- metric_name
  res$Model <- rownames(res)
  return(res)
}

ed50_combined <- rbind(get_ed50("mAP_COCO"), get_ed50("F1_Score"))

tab_df(ed50_combined, title = "Table: Failure Thresholds (mAP vs F1)", file = "Noise_Failure_Thresholds_Comparison.html")