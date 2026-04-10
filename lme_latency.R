library(tidyverse)
library(lme4)
library(lmerTest)
library(sjPlot)

# 1. Load Data
data <- read.csv("LME_Ready_Data_Degraded.csv")

# Clean up warm-up spikes 
data_clean <- data %>% filter(Inference_ms < 200)

# Set YOLO11n as the Baseline
data_clean$Model <- factor(data_clean$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))

# 2. SEPARATE the data
data_noise <- data_clean %>% filter(Degradation_Type == "noise")
data_res <- data_clean %>% filter(Degradation_Type == "resolution")

# 3. Run the two separated models for Inference Time
reg_latency_noise <- lmer(Inference_ms ~ Model * Normalized_Stress + (1 | Image), data = data_noise)
reg_latency_res <- lmer(Inference_ms ~ Model * Normalized_Stress + (1 | Image), data = data_res)

# 4. Generate the side-by-side Thesis Table
tab_model(reg_latency_noise, reg_latency_res,
          dv.labels = c("Latency: Noise Degradation", "Latency: Resolution Degradation"),
          title = "Mixed-Effects Regression on Inference Time (ms)",
          file = "Latency_Separated_Results.html")