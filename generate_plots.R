library(tidyverse)
library(scales)
library(ggplot2)

# ==========================================
# 1. LOAD DATA
# ==========================================
# Read the semicolon-separated CSV
data <- read.csv("LME_Ready_Data_Degraded.csv", sep = ";")

# Clean pandas index column if present
if ("Unnamed..0" %in% colnames(data)) {
  data <- data %>% select(-Unnamed..0)
}

# Focus only on frames with Objects (mAP >= 0)
data_objects_only <- data %>%
  filter(mAP_50 >= 0)

# Factorize models to keep them in a logical order
data_objects_only$Model <- factor(data_objects_only$Model, levels = c("YOLO11n", "YOLO12n", "YOLO26n"))
data_objects_only$Degradation_Type <- as.factor(data_objects_only$Degradation_Type)

# ==========================================
# 2. APPLY TUKEY'S FENCE (IQR) FOR LATENCY
# ==========================================
# We group by Model, Type, and Stress to remove jitter without losing architectural differences
data_clean_latency <- data_objects_only %>%
  group_by(Model, Degradation_Type, Normalized_Stress) %>%
  mutate(
    Q1 = quantile(Inference_ms, 0.25, na.rm = TRUE),
    Q3 = quantile(Inference_ms, 0.75, na.rm = TRUE),
    IQR_val = Q3 - Q1,
    Upper = Q3 + (1.5 * IQR_val),
    Lower = Q1 - (1.5 * IQR_val)
  ) %>%
  # Filter only the latency metrics; we keep all accuracy data
  filter(Inference_ms >= Lower & Inference_ms <= Upper) %>%
  ungroup() %>%
  select(-Q1, -Q3, -IQR_val, -Upper, -Lower)

# ==========================================
# PLOT 1: F1-SCORE DECAY
# ==========================================
p1 <- ggplot(data_objects_only, aes(x = Normalized_Stress, y = F1_Score, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  stat_summary(fun.data = mean_se, geom = "ribbon", aes(fill = Model), alpha = 0.1, color = NA) +
  facet_wrap(~ Degradation_Type, scales = "free_x") +
  theme_minimal(base_size = 14) +
  labs(title = "Classification Reliability: F1 Score vs. Environmental Stress",
       x = "Normalized Stress (0 = Clean, 1 = Max Degradation)",
       y = "Mean F1 Score") +
  theme(legend.position = "bottom")

ggsave("f1_decay.png", plot = p1, width = 12, height = 6, dpi = 300)

# ==========================================
# PLOT2: mAP_COCO DECAY
# ==========================================
p_map <- ggplot(data_objects_only, aes(x = Normalized_Stress, y = mAP_COCO, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  stat_summary(fun.data = mean_se, geom = "ribbon", aes(fill = Model), alpha = 0.1, color = NA) +
  facet_wrap(~ Degradation_Type, scales = "free_x") +
  theme_minimal(base_size = 14) +
  labs(title = "Localization Accuracy: mAP vs. Environmental Stress",
       x = "Normalized Stress (0 = Clean, 1 = Max Degradation)",
       y = "Mean mAP@[0.50:0.95]") +
  theme(legend.position = "bottom")

ggsave("map_decay.png", plot = p_map, width = 12, height = 6, dpi = 300)


# ==========================================
# PLOT 3: LATENCY (CLEANED BY IQR)
# ==========================================
p3 <- ggplot(data_clean_latency, aes(x = Normalized_Stress, y = Inference_ms, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  stat_summary(fun.data = mean_se, geom = "ribbon", aes(fill = Model), alpha = 0.2, color = NA) +
  facet_wrap(~ Degradation_Type, scales = "free_x") +
  theme_minimal(base_size = 14) +
  labs(title = "Computational Cost: Inference Latency vs. Environmental Stress",
       subtitle = "Cleaned using Tukey's Fence (IQR) to remove hardware jitter",
       x = "Normalized Stress (0 = Clean, 1 = Max Degradation)",
       y = "Mean Inference Time (ms)") +
  theme(legend.position = "bottom")

ggsave("latency_line_ribbon.png", plot = p3, width = 12, height = 6, dpi = 300)

# ==========================================
# PLOT 4: PARETO EFFICIENCY (CLEANED)
# ==========================================
pareto_data <- data_clean_latency %>%
  group_by(Model, Degradation_Type) %>%
  summarize(Mean_mAP = mean(mAP_COCO, na.rm = TRUE),
            Mean_Latency = mean(Inference_ms, na.rm = TRUE),
            .groups = 'drop')

p4 <- ggplot(pareto_data, aes(x = Mean_Latency, y = Mean_mAP, color = Model)) +
  geom_point(size = 6, alpha = 0.8) +
  geom_text(aes(label = Model), vjust = -2, fontface = "bold", show.legend = FALSE) +
  facet_wrap(~ Degradation_Type, scales = "free") +
  theme_minimal(base_size = 14) +
  # Ensure labels are visible
  scale_y_continuous(labels = scales::percent_format(), expand = expansion(mult = c(0.2, 0.2))) +
  scale_x_continuous(expand = expansion(mult = c(0.2, 0.2))) +
  labs(title = "Pareto Efficiency: Speed vs. Quality",
       subtitle = "Calculated using IQR-cleaned latency for precise hardware profiling",
       x = "Mean Inference Time (ms)",
       y = "Mean mAP@[0.50:0.95]") +
  theme(legend.position = "none")

ggsave("pareto_efficiency.png", plot = p4, width = 12, height = 6, dpi = 300)

cat("Successfully generated all cleaned thesis plots!\n")
