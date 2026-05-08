library(tidyverse)
library(sjPlot)
library(drc)
library(lmerTest)

# 1. LOAD AND PREP
data <- read.csv("../database/LME_Ready_Data_Degraded.csv", sep = ";")
data_valid <- data %>% filter(mAP_50 >= 0)
data_valid$Model <- factor(data_valid$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))

# --- A. RESOLUTION ANALYSIS (POLYNOMIAL) ---
res_data <- data_valid %>% filter(Degradation_Type == "resolution")

# Model for mAP
reg_res_map <- lmer(mAP_COCO ~ Model * poly(Normalized_Stress, 2) + (1 | Sequence/Image), data = res_data)
# Model for F1
reg_res_f1 <- lmer(F1_Score ~ Model * poly(Normalized_Stress, 2) + (1 | Sequence/Image), data = res_data)

tab_model(reg_res_map, reg_res_f1,
          dv.labels = c("mAP_COCO (Resolution)", "F1-Score (Resolution)"),
          title = "Table: Resolution Decay Comparison",
          file = "Resolution_mAP_vs_F1.html")


# --- B. NOISE ANALYSIS (CUBIC) ---
noise_data <- data_valid %>% filter(Degradation_Type == "noise")

# Model for mAP
reg_noise_map <- lmer(mAP_COCO ~ Model * poly(Normalized_Stress, 3) + (1 | Sequence/Image), data = noise_data)
# Model for F1
reg_noise_f1 <- lmer(F1_Score ~ Model * poly(Normalized_Stress, 3) + (1 | Sequence/Image), data = noise_data)

tab_model(reg_noise_map, reg_noise_f1,
          dv.labels = c("mAP_COCO (Noise)", "F1-Score (Noise)"),
          title = "Table: Noise Decay Comparison (Cubic)",
          file = "Noise_mAP_vs_F1.html")


# --- C. FAILURE THRESHOLDS (ED50) FOR BOTH DEGRADATIONS ---

# Function to get ED50 for a specific metric and specific degradation type
get_ed50 <- function(dataset, metric_name, deg_type) {
  df_agg <- dataset %>%
    filter(Degradation_Type == deg_type) %>%
    group_by(Model, Normalized_Stress) %>%
    summarize(m = mean(get(metric_name), na.rm = TRUE), .groups = 'drop')

  # Fit the dose-response model
  fit <- drm(m ~ Normalized_Stress, curveid = Model, data = df_agg, fct = L.4())
  res <- as.data.frame(ED(fit, 50, interval = "delta", display = FALSE))

  # Clean up the output table structure
  res$Metric <- metric_name
  res$Degradation <- deg_type

  # Reorder columns
  res <- res[, c("Degradation", "Model", "Metric", "Estimate", "Std. Error", "Lower", "Upper")]
  return(res)
}

# Calculate all four combinations
ed50_noise_map <- get_ed50(data_objects_only, "mAP_COCO", "noise")
ed50_noise_f1  <- get_ed50(data_objects_only, "F1_Score", "noise")

ed50_res_map <- get_ed50(data_objects_only, "mAP_COCO", "resolution")
ed50_res_f1  <- get_ed50(data_objects_only, "F1_Score", "resolution")

# Bind them all into one master table
ed50_combined <- rbind(ed50_noise_map, ed50_noise_f1, ed50_res_map, ed50_res_f1)

# Generate the HTML Table
tab_df(ed50_combined,
       title = "Table 2: Effective Dose 50% (ED50) Failure Thresholds",
       file = "Master_Failure_Thresholds.html")