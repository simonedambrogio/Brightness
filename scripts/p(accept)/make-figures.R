library(ggplot2); library(ggdist); library(tidyr); library(yaml)
library(grid); library(gtable); library(forcats)
config <- read_yaml("config.yaml")
source("src/mytheme.R")

# value difference -------------------------------------------------------------
data_behavior <- read.csv("data/processed/data_behavior.csv") 

# Calculate empirical p(accept) for observed data
empirical_data <- data_behavior %>%
  mutate(
    value_diff = gain - loss,
    condition = ifelse(SalL, "Loss Salient", "Gain Salient"),
    # Create value difference bins for aggregation
    value_diff_bin = round(value_diff * 4) / 4  # Bin to nearest 0.25
  ) %>%
  group_by(value_diff_bin, condition) %>%
  summarise(
    p_accept_emp = mean(choice, na.rm = TRUE),
    n_trials = n(),
    se = sqrt(p_accept_emp * (1 - p_accept_emp) / n_trials),
    .groups = 'drop'
  ) %>%
  filter(n_trials >= 5)  # Only include bins with at least 5 trials

p_value_diff <- ggplot() +
  # Empirical data ribbons (confidence bands around empirical data)
  geom_ribbon(data = empirical_data,
              aes(x = value_diff_bin, 
                  ymin = pmax(0, p_accept_emp - se), 
                  ymax = pmin(1, p_accept_emp + se),
                  fill = condition),
              alpha = 0.3, color = NA) +
  
  # Empirical data points
  geom_point(data = empirical_data,
             aes(x = value_diff_bin, y = p_accept_emp, color = condition),
             size = 4, alpha = 0.8, shape = 16) +
  # Cross 0 and 0.5
  geom_vline(xintercept = 0, linewidth = 0.3, linetype = 2) +
  geom_hline(yintercept = 0.5, linewidth = 0.3, linetype = 2) +
  
  # Visual settings
  scale_color_manual(values = c("Loss Salient" = config$colors$`loss-salient`, 
                               "Gain Salient" = config$colors$`gain-salient`)) +
  scale_fill_manual(values = c("Loss Salient" = config$colors$`loss-salient`, 
                              "Gain Salient" = config$colors$`gain-salient`)) +
  mytheme() + 
  labs(
    x = "Value Difference\n(Gain - Loss)",
    y="Probability of\naccepting the gamble",
    color = "",
    fill = ""
  ) +
  scale_x_continuous(guide = "prism_offset") + 
  scale_y_continuous(guide = "prism_offset", limits = c(-0.05, 1.05), breaks = seq(0, 1, 0.25)) +
  theme(legend.position = c(0.25, 0.85))

print(p_value_diff)

# Save the plot
ggsave("figures/p(accept)/value_difference.png",
       plot = p_value_diff,
       width = 6, height = 6, dpi = 300)

ggsave("figures/p(accept)/value_difference.svg",
       plot = p_value_diff,
       width = 6, height = 6)


# heat map  --------------------------------------------------------------------
heatmap_plot <- data_behavior %>% 
  group_by(gain, loss) %>% 
  summarise(p_accept = mean(choice), .groups = 'drop') %>% 
  ggplot(aes(x = gain, y = loss, fill = p_accept)) +
  geom_tile() +
  scale_fill_gradient2(
    midpoint = 0.5,
    low = config$colors$reject,
    high = config$colors$accept,
    mid = "white",
    breaks = seq(0, 1, 0.5),
    limits = c(0, 1)
  ) +
  labs(
    x = "Gain",
    y = "Loss",
    fill = "P(Accept)\n"
  ) +
  mytheme() +
  scale_x_continuous(guide = "prism_offset", breaks = 3:9) +
  scale_y_continuous(guide = "prism_offset", breaks = 3:9) +
  theme(
    legend.position = "right"
  )

# Save the plot
ggsave("figures/p(accept)/heatmap.png",
       plot = heatmap_plot,
       width = 7.5, height = 6, dpi = 300)

ggsave("figures/p(accept)/heatmap.svg",
       plot = heatmap_plot,
       width = 7.5, height = 6)



# coefficients -----------------------------------------------------------------
source("scripts/p(accept)/fit.R")

# Horizontal 
pp <- prepare_predictions(m)
fe <- as_tibble(pp$dpars$mu$fe$b) %>% 
  rename(
    Intercept = b_Intercept, 
    Gain = b_gainZ, 
    Loss = b_lossZ, 
    `Prop. Gaze\nGain` = b_gaze_gainZ,
    `First Gaze\nat Gain` = b_first_is_gain,
    `Loss in\nSalient` = b_SalL
  ) %>% 
  select(Intercept, Gain, Loss, `Prop. Gaze\nGain`, `First Gaze\nat Gain`, `Loss in\nSalient`)

# Create the main plot without arrow
plot_coef_base <- fe %>% 
  reshape2::melt() %>% as_tibble() %>%
  mutate(variable = factor(variable, levels = unique(.$variable))) %>%
  ggplot(aes(y = value, x = variable)) + 
  stat_gradientinterval(width=0.2) +
  geom_hline(yintercept = 0., linetype="dashed") +
  mytheme() + 
  theme(legend.position = "top") +
  labs(x = NULL, y = "Coefficients estimates", color="", fill="") +
  scale_y_continuous(guide = "prism_offset") + 
  scale_x_discrete(guide = guide_prism_offset(n.dodge = 2)); print(plot_coef_base) 

# Create arrow grob with two separate arrows
arrow_grob <- gTree(children = gList(
  # Upper arrow (Look at gain) - arrow head at top
  segmentsGrob(x0 = 0.5, y0 = 0.55, x1 = 0.5, y1 = 1,
               arrow = arrow(length = unit(0.08, "inches"), ends = "last", type = "closed"),
               gp = gpar(col = "black", fill = "black", lwd = 2)),
  
  # Lower arrow (Look at loss) - arrow head at bottom
  segmentsGrob(x0 = 0.5, y0 = 0.45, x1 = 0.5, y1 = 0,
               arrow = arrow(length = unit(0.08, "inches"), ends = "last", type = "closed"),
               gp = gpar(col = "black", fill = "black", lwd = 2))
))

# Convert plot to gtable and add arrow column
gt <- ggplotGrob(plot_coef_base)

# Find the panel position
panel_col <- which(gt$layout$name == "panel")
panel_row <- gt$layout$t[panel_col]

# Add a new column between y-axis title and panel
gt <- gtable_add_cols(gt, unit(1.2, "cm"), pos = panel_col - 1)

# Insert arrow grob in the new column
gt <- gtable_add_grob(gt, arrow_grob, 
                      t = panel_row, l = panel_col, 
                      name = "arrow")

# Draw the modified plot
grid.newpage()
grid.draw(gt)

plot_coef <- gt  # For saving purposes

# Save the plot
ggsave("figures/p(accept)/pars.png",
       plot = plot_coef,
       width = 10, height = 4.5, dpi = 300)

ggsave("figures/p(accept)/pars.svg",
       plot = plot_coef,
       width = 10, height = 4.5)

w = 0.7
ggsave("figures/p(accept)/pars.svg",
       plot = plot_coef,
       width = 10 * w, height = 4.5 * w)

# Vertical 
pp <- prepare_predictions(m)
fe <- as_tibble(pp$dpars$mu$fe$b) %>% 
  rename(
    Intercept = b_Intercept, 
    Gain = b_gainZ, 
    Loss = b_lossZ, 
    `Prop. Gaze\nGain` = b_gaze_gainZ,
    `First Gaze\nat Gain` = b_first_is_gain,
    `Loss in\nSalient` = b_SalL
  ) %>% 
  select(Intercept, Gain, Loss, `Prop. Gaze\nGain`, `First Gaze\nat Gain`, `Loss in\nSalient`)

# Extract random effects for each subject
library(brms)
ranef_data <- ranef(m)$subject[, 1, ]  # Get all subjects, first statistic (Estimate), all parameters

# Get fixed effects means
fe_means <- fe %>% 
  summarise(across(everything(), mean)) %>%
  as.list()

# Extract subject-specific coefficients (fixed effects + random effects)
subject_coefs <- tibble(
  subject = rownames(ranef_data),
  Intercept = ranef_data[, "Intercept"],
  Gain = ranef_data[, "gainZ"], 
  Loss = ranef_data[, "lossZ"],
  `Prop. Gaze\nGain` = ranef_data[, "gaze_gainZ"],
  `First Gaze\nat Gain` = ranef_data[, "first_is_gain"],
  `Loss in\nSalient` = ranef_data[, "SalL"]
) %>%
  # Add fixed effects to get total subject-specific effects
  mutate(
    Intercept = Intercept + fe_means$Intercept,
    Gain = Gain + fe_means$Gain,
    Loss = Loss + fe_means$Loss,
    `Prop. Gaze\nGain` = `Prop. Gaze\nGain` + fe_means$`Prop. Gaze\nGain`,
    `First Gaze\nat Gain` = `First Gaze\nat Gain` + fe_means$`First Gaze\nat Gain`,
    `Loss in\nSalient` = `Loss in\nSalient` + fe_means$`Loss in\nSalient`
  ) %>%
  # Reshape for plotting
  reshape2::melt(id.vars = "subject") %>%
  as_tibble() %>%
  mutate(variable = factor(variable, levels = names(fe))) %>%
  mutate(variable = fct_rev(variable))

# Create the main plot with jittered subject points
plot_flipped <- fe %>% 
  reshape2::melt() %>% as_tibble() %>%
  mutate(variable = factor(variable, levels = unique(.$variable))) %>%
  mutate(variable = fct_rev(variable)) %>%  # Add this line to reverse order
  ggplot(aes(y = value, x = variable)) + 
  # Add jittered points for individual subjects
  geom_jitter(data = subject_coefs, aes(y = value, x = variable), 
              width = 0.05, height = 0, alpha = 0.2, size = 3, color = "gray") +
  stat_gradientinterval(width=0.2) +
  geom_hline(yintercept = 0., linetype="dashed") +
  coord_flip() +  # Add this line to flip
  mytheme() + 
  theme(legend.position = "top") +
  labs(x = NULL, y = "\nCoefficients estimates", color="", fill="") +
  scale_y_continuous(guide = "prism_offset") + 
  scale_x_discrete(guide = "prism_offset"); print(plot_flipped)

# Save the flipped plot
ggsave("figures/p(accept)/pars_flipped.png",
       plot = plot_flipped,
       width = 5.5, height = 10, dpi = 300)

ggsave("figures/p(accept)/pars_flipped.svg",
       plot = plot_flipped,
       width = 5.5, height = 10)

