library(ggplot2); library(ggdist); library(tidyr); library(yaml); library(dplyr)
config <- read_yaml("config.yaml")
source("src/mytheme.R")

# Load data
data_behavior <- read.csv("data/processed/data_behavior.csv")

# value difference vs Response Time ----------------------------

# Calculate empirical median RT for observed data
empirical_data_rt <- data_behavior %>%
  mutate(
    value_diff = abs(gain - loss),
    condition = ifelse(SalL, "Loss Salient", "Gain Salient"),
    choice = ifelse(choice, "Accept", "Reject"),
    tft = tft / 1000
  ) %>%
  group_by(value_diff, condition, choice) %>%
  summarise(
    median_rt = median(tft, na.rm = TRUE),
    q25_rt = quantile(tft, 0.25, na.rm = TRUE),
    q75_rt = quantile(tft, 0.75, na.rm = TRUE),
    n_trials = n(),
    se_rt = sd(tft, na.rm = TRUE) / sqrt(n_trials),  # Standard error for ribbon
    .groups = 'drop'
  ) %>%
  filter(n_trials >= 5)  # Only include bins with at least 5 trials

p_rt_value_diff <- ggplot() +
  # Empirical data ribbons (confidence bands using standard error)
  geom_ribbon(data = empirical_data_rt,
              aes(x = value_diff, 
                  ymin = pmax(0, median_rt - se_rt), 
                  ymax = median_rt + se_rt,
                  fill = condition),
              alpha = 0.3, color = NA) +
  
  # Empirical data points
  geom_point(data = empirical_data_rt,
             aes(x = value_diff, y = median_rt, color = condition),
             size = 4, alpha = 0.8, shape = 16) +
  
  # Reference line at 0 value difference
  geom_vline(xintercept = 0, linewidth = 0.3, linetype = 2) +
  
  # Visual settings
  scale_color_manual(values = c("Loss Salient" = config$colors$`loss-salient`, 
                               "Gain Salient" = config$colors$`gain-salient`)) +
  scale_fill_manual(values = c("Loss Salient" = config$colors$`loss-salient`, 
                              "Gain Salient" = config$colors$`gain-salient`)) +
  mytheme() + 
  labs(
    x = "Module Value Difference\n|Gain - Loss|",
    y = "Median Decision Time\n(seconds)",
    color = "",
    fill = ""
  ) +
  scale_x_continuous(guide = "prism_offset") + 
  scale_y_continuous(guide = "prism_offset") +
  theme(legend.position = "none") +
  facet_grid(~choice)


# Save the plot
path2save <- file.path("figures/response-time")
if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)

ggsave("figures/response-time/rt_vs_value_difference.png",
       plot = p_rt_value_diff,
       width = 6, height = 5, dpi = 300)

ggsave("figures/response-time/rt_vs_value_difference.svg",
       plot = p_rt_value_diff,
       width = 6, height = 5)

# heat map  --------------------------------------------------------------------
heatmap_plot <- data_behavior %>% 
  group_by(gain, loss) %>% 
  summarise(rt = mean(rt), .groups = 'drop') %>% 
  ggplot(aes(x = gain, y = loss, fill = rt)) +
  geom_tile() +
  scale_fill_gradient2(
    midpoint = 1.65,
    low = config$colors$reject,
    high = config$colors$accept,
    mid = "white",
    # breaks = seq(0, 1, 0.5),
    # limits = c(0, 1)
  ) +
  labs(
    x = "Gain",
    y = "Loss",
    fill = "Response\ntime (s)\n"
  ) +
  mytheme() +
  scale_x_continuous(guide = "prism_offset", breaks = 3:9) +
  scale_y_continuous(guide = "prism_offset", breaks = 3:9) +
  theme(
    legend.position = "right"
  ); print(heatmap_plot)

# Save the plot
ggsave("figures/response-time/heatmap.png",
       plot = heatmap_plot,
       width = 7.5, height = 6, dpi = 300)

ggsave("figures/response-time/heatmap.svg",
       plot = heatmap_plot,
       width = 7.5, height = 6)
