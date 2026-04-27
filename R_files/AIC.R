library(tidyverse)
library(sjPlot)
library(lme4)



# 1. LOAD AND PREP
data <- read.csv("../database/LME_Ready_Data_Degraded.csv", sep = ";")
data_valid <- data %>% filter(mAP_50 >= 0)
data_valid$Model <- factor(data_valid$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))

# Resolution data
res_data <- data_valid %>% filter(Degradation_Type == "resolution")
# Noise data
noise_data <- data_valid %>% filter(Degradation_Type == "noise")


# 2. FIT MODELS (Using REML = FALSE for correct AIC comparison of fixed effects)
# Resolution Models
mod_res_linear <- lmer(mAP_COCO ~ Normalized_Stress + (1 | Sequence/Image), data = res_data, REML = FALSE)
mod_res_quad <- lmer(mAP_COCO ~ poly(Normalized_Stress, 2) + (1 | Sequence/Image), data = res_data, REML = FALSE)

# Noise Models
mod_noise_linear <- lmer(mAP_COCO ~ Normalized_Stress + (1 | Sequence/Image), data = noise_data, REML = FALSE)
mod_noise_quad <- lmer(mAP_COCO ~ poly(Normalized_Stress, 2) + (1 | Sequence/Image), data = noise_data, REML = FALSE)
mod_noise_cubic <- lmer(mAP_COCO ~ poly(Normalized_Stress, 3) + (1 | Sequence/Image), data = noise_data, REML = FALSE)


# 3. STATISTICAL ANOVA COMPARISON
anova_res <- anova(mod_res_linear, mod_res_quad)
anova_noise <- anova(mod_noise_linear, mod_noise_quad, mod_noise_cubic)

# Print to console just so you can see it while running
print(anova_res)
print(anova_noise)


# 4. BUILD THE SUMMARY DATA FRAME
# Extracting the metrics into a clean table structure
aic_summary <- data.frame(
  Degradation = c("Resolution", "Resolution", "Noise", "Noise", "Noise"),
  Polynomial_Fit = c("Linear (1st Order)", "Quadratic (2nd Order)", "Linear (1st Order)", "Quadratic (2nd Order)", "Cubic (3rd Order)"),
  AIC = c(AIC(mod_res_linear), AIC(mod_res_quad), AIC(mod_noise_linear), AIC(mod_noise_quad), AIC(mod_noise_cubic)),
  BIC = c(BIC(mod_res_linear), BIC(mod_res_quad), BIC(mod_noise_linear), BIC(mod_noise_quad), BIC(mod_noise_cubic)),
  p_value = c(NA, anova_res$`Pr(>Chisq)`[2], NA, anova_noise$`Pr(>Chisq)`[2], anova_noise$`Pr(>Chisq)`[3])
)

# Clean up the numbers for presentation
aic_summary$AIC <- round(aic_summary$AIC, 1)
aic_summary$BIC <- round(aic_summary$BIC, 1)
aic_summary$p_value <- ifelse(is.na(aic_summary$p_value), "-",
                              ifelse(aic_summary$p_value < 0.001, "< 0.001",
                                     round(aic_summary$p_value, 4)))


# 5. EXPORT TO HTML (Using sjPlot)
tab_df(aic_summary,
       title = "Table: Akaike Information Criterion (AIC) Polynomial Model Comparisons",
       file = "AIC_Model_Comparisons.html")

print("HTML Table successfully generated: AIC_Model_Comparisons.html")
