# --- Load Libraries ---
library(tidyverse)
library(sjPlot)
library(drc) # Now this will work!

# 1. LOAD DATA (Semicolon separator)
data <- read.csv("LME_Ready_Data_Degraded.csv", sep = ";")

# 2. PREP DATA
# Use only frames with objects and set the baseline model
data_valid <- data %>% filter(mAP_50 >= 0)
data_valid$Model <- factor(data_valid$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))

# =======================================================
# MODEL A: RESOLUTION ACCURACY (Polynomial Curve)
# =======================================================
# We use a 2nd-order polynomial to catch the accelerating decay waterfall
res_data <- data_valid %>% filter(Degradation_Type == "resolution")
reg_res_poly <- lm(mAP_COCO ~ Model * poly(Normalized_Stress, 2), data = res_data)

# This will create a professional table with the quadratic (poly2) coefficients
tab_model(reg_res_poly,
          title = "Table 2: Resolution Accuracy Decay - Polynomial Analysis",
          file = "Resolution_Performance_Results.html")


# =======================================================
# MODEL B: NOISE ACCURACY (Cubic Polynomial + Sigmoidal Failure Thresholds)
# =======================================================
# --- NOISE ACCURACY (Cubic Polynomial) ---
# We use poly(x, 3) to capture the "S-Curve" cliff of Gaussian Noise
noise_data <- data_valid %>% filter(Degradation_Type == "noise")

reg_noise_cubic <- lm(mAP_COCO ~ Model * poly(Normalized_Stress, 3), data = noise_data)

# Generate the formal table with p-values
tab_model(reg_noise_cubic,
          title = "Table 3: Noise Accuracy Decay - Cubic Polynomial Analysis",
          file = "Noise_Cubic_Results.html")

# 1. Group the data to get the average 'Cliff' path for each model
noise_data_agg <- data_valid %>%
  filter(Degradation_Type == "noise") %>%
  group_by(Model, Normalized_Stress) %>%
  summarize(Mean_mAP = mean(mAP_COCO), .groups = 'drop')

# 2. Fit the Sigmoidal (4-Parameter Logistic) model
# This mathematically fits the 'S-Curve' you saw in the graphs
fit_noise_sigmoid <- drm(Mean_mAP ~ Normalized_Stress, curveid = Model,
                         data = noise_data_agg, fct = L.4())

# 3. Calculate the "ED50" (Effective Degradation 50%)
# This tells the EXACT stress level where each model collapsed by 50%
thresholds <- as.data.frame(ED(fit_noise_sigmoid, 50, interval = "delta"))
thresholds$Model <- rownames(thresholds)

# Rename columns for the thesis table
colnames(thresholds) <- c("Failure_Threshold_ED50", "Std_Error", "Lower_CI", "Upper_CI", "Model_ID")

# 4. Generate the HTML table for your Noise results
tab_df(thresholds,
       title = "Table 4: Noise Robustness - Sigmoidal Failure Thresholds (ED50)",
       file = "Noise_Failure_Thresholds.html")

cat("Success! Check your folder for Resolution_Performance_Results.html and Noise_Failure_Thresholds.html")