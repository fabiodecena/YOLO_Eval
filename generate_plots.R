# generate_plots.R
library(tidyverse)

# 1. Load Data
data <- read.csv("LME_Ready_Data_Degraded.csv")

# 2. Pre-processing
# Convert to factors for correct grouping in plots
data$Model <- as.factor(data$Model)
data$Degradation_Type <- as.factor(data$Degradation_Type)

# --- PLOT 1: Confidence Decay ---
# Identifies how model reliability drops as stress increases
p1 <- ggplot(data, aes(x = Degradation_Level, y = Confidence, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.05) +
  facet_wrap(~Degradation_Type, scales = "free_x") +
  theme_minimal() +
  labs(title = "Detection Confidence Decay",
       subtitle = "Grouped by Noise and Resolution levels",
       x = "Intensity", y = "Mean Confidence")

ggsave("confidence_decay.png", plot = p1, width = 10, height = 6)

# --- PLOT 2: Latency Efficiency ---
# Visualizes the stability and speed of the different YOLO versions
p2 <- ggplot(data, aes(x = Model, y = Inference_ms, fill = Model)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~Degradation_Type) +
  theme_bw() +
  labs(title = "Inference Latency by Architecture",
       x = "Model Version", y = "Latency (ms)")

ggsave("latency_comparison.png", plot = p2, width = 8, height = 6)

# --- PLOT 3: The Failure Threshold (Logistic Curve) ---
# Shows the probability of the model actually seeing an object
p3 <- ggplot(data, aes(x = Degradation_Level, y = Success, color = Model)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = TRUE) +
  facet_wrap(~Degradation_Type, scales = "free_x") +
  scale_y_continuous(labels = scales::percent) +
  theme_light() +
  labs(title = "Detection Success Probability",
       subtitle = "S-curve identifying the tipping point for each model",
       x = "Degradation Level", y = "Success Rate (%)")

ggsave("failure_threshold.png", plot = p3, width = 10, height = 6)

# --- PLOT 4: Accuracy-Speed Trade-off ---
# Summarize data first
pareto_data <- data %>%
  group_by(Model, Degradation_Type) %>%
  summarize(
    mean_latency = mean(Inference_ms),
    mean_conf = mean(Confidence)
  )

p4 <- ggplot(pareto_data, aes(x = mean_latency, y = mean_conf, color = Model, shape = Degradation_Type)) +
  geom_point(size = 4) +
  geom_text(aes(label = Model), vjust = -1, size = 3) +
  theme_minimal() +
  labs(title = "Hardware Efficiency vs. Model Reliability",
       x = "Mean Latency (ms) - Lower is Better",
       y = "Mean Confidence - Higher is Better")

ggsave("efficiency_tradeoff.png", plot = p4, width = 8, height = 6)

message("Analysis complete. Check the directory for .png files.")

