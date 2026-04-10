library(tidyverse)
library(scales)

# 1. Load Data
data <- read.csv("LME_Ready_Data_Degraded.csv")

# Clean pandas index column if present
if ("Unnamed..0" %in% colnames(data)) {
  data <- data %>% select(-Unnamed..0)
}

# --- CRITICAL FIX: Focus only on frames with Objects ---
# mAP is -1.0 when there is no ground truth.
# We filter these out to see how models actually perform on targets.
data_objects_only <- data %>%
  filter(mAP_50 >= 0)

# 2. Pre-processing
data_objects_only$Model <- as.factor(data_objects_only$Model)
data_objects_only$Degradation_Type <- as.factor(data_objects_only$Degradation_Type)

# Clean up latency: Remove warm-up spikes
data_clean <- data_objects_only %>%
  filter(Inference_ms < 200)

# --- PLOT 1: F1-Score Decay ---
# Using stat_summary for efficient plotting of large datasets
p1 <- ggplot(data_objects_only, aes(x = Normalized_Stress, y = F1_Score, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  stat_summary(fun.data = mean_se, geom = "ribbon", aes(fill = Model), alpha = 0.1, color = NA) +
  facet_wrap(~Degradation_Type) +
  theme_minimal() +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) + # Removed fixed 0-1 limits to see decay clearly
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", alpha = 0.5) +
  labs(title = "YOLO Architecture Robustness: F1-Score Decay",
       subtitle = "Analysis of frames containing objects; Shaded area = Standard Error",
       x = "Normalized Stress Level",
       y = "F1-Score") +
  theme(legend.position = "bottom")

ggsave("f1_decay.png", plot = p1, width = 10, height = 6)

# --- PLOT 2: Precision vs. Recall Divergence ---
metrics_summary <- data_objects_only %>%
  group_by(Model, Degradation_Type, Normalized_Stress) %>%
  summarise(
    Precision = mean(mAP_50),
    Recall = mean(Recall),
    .groups = 'drop'
  ) %>%
  pivot_longer(cols = c(Precision, Recall), names_to = "Metric", values_to = "Value")

p2 <- ggplot(metrics_summary, aes(x = Normalized_Stress, y = Value, color = Metric)) +
  geom_line(linewidth = 1) +
  facet_grid(Degradation_Type ~ Model) +
  theme_light() +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  scale_color_manual(values = c("Precision" = "#1b9e77", "Recall" = "#d95f02")) +
  labs(title = "Failure Mode Analysis: Precision vs. Recall",
       subtitle = "Visualizing the cross-over point where thermal noise breaks detection",
       x = "Normalized Stress Intensity",
       y = "Mean Score") +
  theme(legend.position = "bottom")

ggsave("precision_recall_divergence.png", plot = p2, width = 12, height = 6)

# --- PLOT 3: Latency Distribution ---
p3 <- ggplot(data_clean, aes(x = Model, y = Inference_ms, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  coord_cartesian(ylim = c(0, 30)) +
  theme_bw() +
  labs(title = "Inference Latency Distribution (v11 vs v12 vs v26)",
       x = "YOLO Architecture",
       y = "Inference Time (ms)") +
  theme(legend.position = "none")

ggsave("latency_comparison.png", plot = p3, width = 8, height = 6)

# --- PLOT 4: Pareto Efficiency (Zoomed) ---
# We calculate averages per model/degradation type
pareto_data <- data_clean %>%
  group_by(Model, Degradation_Type) %>%
  summarize(
    mean_latency = mean(Inference_ms, na.rm = TRUE),
    mean_map_coco = mean(mAP_COCO, na.rm = TRUE),
    .groups = 'drop'
  )

p4 <- ggplot(pareto_data, aes(x = mean_latency, y = mean_map_coco, color = Model, shape = Degradation_Type)) +
  geom_point(size = 6, alpha = 0.8) +
  # Using ggrepel-style logic to separate labels if needed
  geom_text(aes(label = Model), vjust = -1.5, fontface = "bold", show.legend = FALSE) +
  theme_minimal() +
  # CRITICAL: No forced limits here so the points separate vertically
  scale_y_continuous(labels = percent) +
  labs(title = "Pareto Efficiency: Speed vs. Quality (mAP_COCO)",
       subtitle = "Zoomed view: Visualizing small performance gaps in thermal sensing",
       x = "Mean Inference Time (ms)",
       y = "Mean mAP (IoU 0.50:0.95)") +
  theme(legend.position = "right")

ggsave("pareto_efficiency.png", plot = p4, width = 9, height = 6)

# --- FINAL SUMMARY PRINT ---
cat("\n--- Object-Focused Performance Table ---\n")
print(as.data.frame(pareto_data %>% arrange(desc(mean_map_coco))))

