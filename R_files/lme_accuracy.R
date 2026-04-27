# Install packages
install.packages(c("lme4", "lmerTest", "sjPlot", "tidyverse"))

library(tidyverse)
library(lme4)
library(lmerTest) # Adds p-values to lme4 models
library(sjPlot)   # Creates publication-ready tables

# 1. Load and prep data
data <- read.csv("../database/LME_Ready_Data_Degraded.csv")

# Ensure we are only looking at valid frames
data <- data %>% filter(mAP_50 >= 0)

# Set YOLO11n as the "Baseline" factor.
# This means all regression coefficients will compare YOLO12 and YOLO26 directly against YOLO11.
data$Model <- factor(data$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))

# Split the data because Noise and Resolution behave very differently
data_noise <- data %>% filter(Degradation_Type == "noise")
data_res <- data %>% filter(Degradation_Type == "resolution")

# 2. Run the Mixed-Effects Regressions
# Formula: mAP_COCO depends on Model + Stress + (Model interacting with Stress),
# while accounting for the baseline difficulty of the individual Image.
reg_noise <- lmer(mAP_COCO ~ Model * Normalized_Stress + (1 | Sequence/Image), data = data_noise)
reg_res <- lmer(mAP_COCO ~ Model * Normalized_Stress + (1 | Sequence/Image), data = data_res)

# 3. View the raw mathematical output
summary(reg_noise)

# 4. Generate a Thesis-Ready HTML Table
# This will create a file called "lme_Results.html".
tab_model(reg_noise, reg_res,
          dv.labels = c("Model 1: Noise Degradation", "Model 2: Resolution Degradation"),
          title = "Mixed-Effects Regression on mAP_COCO",
          file = "lme_results.html")