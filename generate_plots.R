library(tidyverse)
library(scales)
library(ggplot2)

# ==========================================
# 1. LOAD DATA
# ==========================================
data <- read.csv("LME_Ready_Data_Degraded.csv", sep = ";")

# Fix 1: Use dplyr:: explicitly to avoid namespace conflicts
if ("Unnamed..0" %in% colnames(data)) {
  data <- data %>% dplyr::select(-Unnamed..0)
}

data_objects_only <- data %>% dplyr::filter(mAP_50 >= 0)

data_objects_only$Model <- factor(data_objects_only$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))
data_objects_only$Degradation_Type <- as.factor(data_objects_only$Degradation_Type)

# ==========================================
# 2. APPLY TUKEY'S FENCE (IQR) FOR LATENCY
# ==========================================
data_clean_latency <- data_objects_only %>%
  group_by(Model, Degradation_Type, Normalized_Stress) %>%
  mutate(
    Q1 = quantile(Inference_ms, 0.25, na.rm = TRUE),
    Q3 = quantile(Inference_ms, 0.75, na.rm = TRUE),
    IQR_val = Q3 - Q1,
    Upper = Q3 + (1.5 * IQR_val),
    Lower = Q1 - (1.5 * IQR_val)
  ) %>%
  # Fix 2: Use dplyr::filter and dplyr::select here as well!
  dplyr::filter(Inference_ms >= Lower & Inference_ms <= Upper) %>%
  ungroup() %>%
  dplyr::select(-Q1, -Q3, -IQR_val, -Upper, -Lower)

# ==========================================
# PLOT 1: F1-SCORE DECAY
# ==========================================
p1 <- ggplot(data_objects_only, aes(x = Normalized_Stress, y = F1_Score, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  stat_summary(fun.data = mean_se, geom = "ribbon", aes(fill = Model), alpha = 0.1, color = NA) +
  facet_wrap(~ Degradation_Type, scales = "free_x") +
  theme_minimal(base_size = 14) +
  labs(title = "Classification Reliability: F1 Score vs. Environmental Stress",
       x = "Normalized Stress", y = "Mean F1 Score") +
  theme(legend.position = "bottom")

ggsave("f1_decay.png", plot = p1, width = 12, height = 6, dpi = 300)

# ==========================================
# PLOT 2: mAP_COCO DECAY
# ==========================================
p_map <- ggplot(data_objects_only, aes(x = Normalized_Stress, y = mAP_COCO, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  stat_summary(fun.data = mean_se, geom = "ribbon", aes(fill = Model), alpha = 0.1, color = NA) +
  facet_wrap(~ Degradation_Type, scales = "free_x") +
  theme_minimal(base_size = 14) +
  labs(title = "Localization Accuracy: mAP vs. Environmental Stress",
       x = "Normalized Stress", y = "Mean mAP@[0.50:0.95]") +
  theme(legend.position = "bottom")

ggsave("map_decay.png", plot = p_map, width = 12, height = 6, dpi = 300)

# ==========================================
# PLOT 3: LATENCY TRENDS (LINE + RIBBON)
# ==========================================
p3 <- ggplot(data_clean_latency, aes(x = Normalized_Stress, y = Inference_ms, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  stat_summary(fun.data = mean_se, geom = "ribbon", aes(fill = Model), alpha = 0.2, color = NA) +
  facet_wrap(~ Degradation_Type, scales = "free_x") +
  theme_minimal(base_size = 14) +
  labs(title = "Latency Trends vs. Stress",
       subtitle = "Cleaned using Tukey's Fence (IQR)",
       x = "Normalized Stress", y = "Inference Time (ms)") +
  theme(legend.position = "bottom")

ggsave("latency_line_ribbon.png", plot = p3, width = 12, height = 6, dpi = 300)

# ==========================================
# PLOT 4: PARETO EFFICIENCY
# ==========================================
pareto_data <- data_clean_latency %>%
  group_by(Model, Degradation_Type) %>%
  summarize(Mean_mAP = mean(mAP_COCO, na.rm = TRUE),
            Mean_Latency = mean(Inference_ms, na.rm = TRUE), .groups = 'drop')

p4 <- ggplot(pareto_data, aes(x = Mean_Latency, y = Mean_mAP, color = Model)) +
  geom_point(size = 6, alpha = 0.8) +
  geom_text(aes(label = Model), vjust = -2, fontface = "bold", show.legend = FALSE) +
  facet_wrap(~ Degradation_Type, scales = "free") +
  theme_minimal(base_size = 14) +
  scale_y_continuous(labels = scales::percent_format(), expand = expansion(mult = c(0.2, 0.2))) +
  scale_x_continuous(expand = expansion(mult = c(0.2, 0.2))) +
  labs(title = "Pareto Efficiency: Speed vs. Quality", x = "Mean Latency (ms)", y = "Mean mAP")

ggsave("pareto_efficiency.png", plot = p4, width = 12, height = 6, dpi = 300)

# ==========================================
# PLOT 5: OVERALL LATENCY COMPARISON (FACETED BOXPLOT)
# ==========================================
p5 <- ggplot(data_clean_latency, aes(x = Model, y = Inference_ms, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.05) +
  facet_wrap(~ Degradation_Type) +
  theme_minimal(base_size = 14) +
  labs(title = "Architectural Latency Stability Across Stressors",
       subtitle = "Distribution of pure inference times (IQR Cleaned)",
       x = "YOLO Architecture",
       y = "Inference Time (ms)") +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold", size = 14)) +
  scale_fill_brewer(palette = "Set1")

ggsave("latency_comparison_boxplot_split.png", plot = p5, width = 10, height = 6, dpi = 300)

# ==========================================
# PLOT 6: OVERALL ROBUSTNESS (AUC / MEAN PERFORMANCE) - mAP
# ==========================================
# Calculate the overall mean performance across all stress levels
robustness_data <- data_objects_only %>%
  group_by(Model, Degradation_Type) %>%
  summarize(
    Mean_mAP = mean(mAP_COCO, na.rm = TRUE),
    Mean_F1 = mean(F1_Score, na.rm = TRUE),
    .groups = 'drop'
  )

p6 <- ggplot(robustness_data, aes(x = Degradation_Type, y = Mean_mAP, fill = Model)) +
  # Use geom_col for a clean Bar Chart, grouped side-by-side
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black", alpha = 0.85) +
  # Add the exact percentages on top of the bars!
  geom_text(aes(label = scales::percent(Mean_mAP, accuracy = 0.1)),
            position = position_dodge(width = 0.8), vjust = -0.5, fontface = "bold") +
  theme_minimal(base_size = 14) +
  labs(title = "Overall Localization Robustness (mAP Area Under Curve)",
       subtitle = "Calculated as the total mean accuracy across the entire 0.0 to 1.0 stress spectrum",
       x = "Environmental Stressor",
       y = "Mean mAP@[0.50:0.95] (Higher is more robust)") +
  # Expand the Y-axis slightly so the text labels don't get cut off
  scale_y_continuous(labels = scales::percent_format(), expand = expansion(mult = c(0, 0.15))) +
  scale_fill_brewer(palette = "Set1") +
  theme(legend.position = "bottom",
        axis.text.x = element_text(face = "bold", size = 12))

ggsave("robustness_map_bar.png", plot = p6, width = 10, height = 6, dpi = 300)

# ==========================================
# PLOT 7: OVERALL ROBUSTNESS (AUC / MEAN PERFORMANCE) - F1 SCORE
# ==========================================
p7 <- ggplot(robustness_data, aes(x = Degradation_Type, y = Mean_F1, fill = Model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black", alpha = 0.85) +
  geom_text(aes(label = scales::percent(Mean_F1, accuracy = 0.1)),
            position = position_dodge(width = 0.8), vjust = -0.5, fontface = "bold") +
  theme_minimal(base_size = 14) +
  labs(title = "Overall Detection Reliability (F1 Area Under Curve)",
       subtitle = "Calculated as the total mean reliability across the entire 0.0 to 1.0 stress spectrum",
       x = "Environmental Stressor",
       y = "Mean F1-Score (Higher is more robust)") +
  scale_y_continuous(labels = scales::percent_format(), expand = expansion(mult = c(0, 0.15))) +
  scale_fill_brewer(palette = "Set1") +
  theme(legend.position = "bottom",
        axis.text.x = element_text(face = "bold", size = 12))

ggsave("robustness_f1_bar.png", plot = p7, width = 10, height = 6, dpi = 300)

cat("Successfully generated Accuracy Boxplots!\n")