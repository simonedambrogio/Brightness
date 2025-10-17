library(tidyr); library(tidybayes); library(grid); library(gtable)
source("src/utils.R")
source("scripts/p(gaze)/fit.R")

m = readRDS("data/results/p(gaze)/m.rds")
data_behavior <- read.csv("data/processed/data_behavior.csv") 

# Plot probability of looking at gain ------------------------------------------
# Get random effects and fixed effects draws
re_draws <- m %>%
  spread_draws(r_subject[subject,term]) %>%
  select(subject, .draw, term, r_subject)

fe_draws <- m %>%
  spread_draws(
    b_Intercept,
    b_gain_is_salient
  )

# Create predictions for all conditions
ranef_df <- re_draws %>%
  pivot_wider(names_from = term, values_from = r_subject) %>%
  left_join(fe_draws, by = ".draw") %>%
  reframe(
    # Brightness, Gain is Salient
    prob_brightness_gain = plogis(b_Intercept + Intercept + gain_is_salient) |> mean(),
    # Brightness, Loss is Salient
    prob_brightness_loss = plogis(b_Intercept + Intercept) |> mean()
  ) %>%
  pivot_longer(
    cols = starts_with("prob"),
    names_to = "condition",
    values_to = "prob"
  ) %>%
  mutate(
    condition = case_when(
      condition == "prob_brightness_gain" ~ "Brightness_Gain",
      condition == "prob_brightness_loss" ~ "Brightness_Loss"
    )
  ) %>%
  separate(condition, into = c("condition", "salience"), sep = "_") %>% 
  mutate(x = ifelse(salience=="Gain", 1, 2))

# Compute mean and credible intervals
pp <- prepare_predictions(m)
fe <- as.data.frame(pp$dpars$mu$fe$b)
post_gainissalient = plogis(fe$b_Intercept + fe$b_gain_is_salient)
post_lossissalient = plogis(fe$b_Intercept)

fixef_df <- rbind(
  tibble(salience="Gain is\nSalient", post=post_gainissalient, x=1),
  tibble(salience="Loss is\nSalient", post=post_lossissalient, x=2)
) 



plot_gaze <- fixef_df %>% 
  ggplot(aes(x = salience, y = post)) +
  stat_gradientinterval(aes(, fill=salience, color=salience), width=0.2) +
  geom_line(
    data=fixef_df %>% group_by(salience) %>% summarise(post=mean(post)), 
    aes(, fill=salience, color=salience, group=salience), linetype = "dashed"
  ) +
  # geom_jitter(data = ranef_df, aes(x-0.1, prob),  alpha=0.1, width=0.1) +
  scale_color_manual(values = c(config$colors$`gain-salient`, config$colors$`loss-salient`)) +
  scale_fill_manual(values = c(blend_colors(config$colors$`gain-salient`, alpha = 0.4), blend_colors(config$colors$`loss-salient`, alpha = 0.4) )) +
  mytheme() + 
  labs(x="", y="Probability of looking at gain", color="", fill="") +
  scale_x_discrete(guide = "prism_offset") + 
  scale_y_continuous(guide = "prism_offset", limits = c(0.43, 0.65), breaks = seq(0.4, .65, .05)) +
  theme(
    legend.position="none",
    plot.margin = margin(t = 10, r = 5, b = 5, l = 5, unit = "pt")
  ); print(plot_gaze)

# Save the plot
ggsave("figures/p(gaze)/salience.png",
       plot = plot_gaze,
       width = 4.5, height = 5.5, dpi = 300)

ggsave("figures/p(gaze)/salience.svg",
       plot = plot_gaze,
       width = 4, height = 5.5)




# Plot coefficients estimates --------------------------------------------------
ranef_data <- ranef(m)$subject[, 1, ]  # Get all subjects, first statistic (Estimate), all parameters
pp <- prepare_predictions(m)
fe <- as.data.frame(pp$dpars$mu$fe$b) %>% 
  rename(
    Intercept = b_Intercept, 
    Gain = b_gainZ, 
    Loss = b_lossZ, 
    `Gain is\nSalient` = `b_gain_is_salient`
  ) %>% 
  select(Intercept, Gain, Loss, `Gain is\nSalient`)

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
  `Gain is\nSalient` = ranef_data[, "gain_is_salient"]
) %>%
  # Add fixed effects to get total subject-specific effects
  mutate(
    Intercept = Intercept + fe_means$Intercept,
    Gain = Gain + fe_means$Gain,
    Loss = Loss + fe_means$Loss,
    `Gain is\nSalient` = `Gain is\nSalient` + fe_means$`Gain is\nSalient`
  ) %>%
  # Reshape for plotting
  reshape2::melt(id.vars = "subject") %>%
  as_tibble() %>%
  mutate(variable = factor(variable, levels = names(fe)))


# Create the main plot without arrow
plot_coef_base <- fe %>% 
  reshape2::melt() %>% as_tibble() %>%
  mutate(variable = factor(variable, levels = unique(.$variable))) %>%
  ggplot(aes(y = value, x = variable)) + 
  geom_jitter(data = subject_coefs, aes(y = value, x = variable), 
              width = 0.05, height = 0, alpha = 0.2, size = 3, color = "gray") +
  stat_gradientinterval(width=0.2) +
  geom_hline(yintercept = 0., linetype="dashed") +
  mytheme() + 
  theme(
    legend.position = "top",
    plot.margin = margin(t = 20, r = 5, b = 15, l = 5, unit = "pt")
  ) +
  labs(x = NULL, y = "Coefficients estimates", color="", fill="") +
  scale_y_continuous(guide = "prism_offset") + 
  scale_x_discrete(guide = "prism_offset"); print(plot_coef_base) 

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
ggsave("figures/p(gaze)/pars-re.png",
       plot = plot_coef,
       width = 8, height = 4.5, dpi = 300)

ggsave("figures/p(gaze)/pars-re.svg",
       plot = plot_coef,
       width = 8, height = 4.5)



# value difference -------------------------------------------------------------
# Calculate empirical p(gaze) for observed data
empirical_data <- data_behavior %>%
  mutate(
    value_diff = gain - loss,
    condition = ifelse(SalL, "Loss Salient", "Gain Salient"),
    # Create value difference bins for aggregation
    value_diff_bin = round(value_diff * 4) / 4  # Bin to nearest 0.25
  ) %>%
  group_by(value_diff_bin, condition) %>%
  summarise(
    p_accept_emp = mean(gaze_gain, na.rm = TRUE),
    n_trials = n(),
    se = sqrt(p_accept_emp * (1 - p_accept_emp) / n_trials),
    .groups = 'drop'
  ) %>%
  filter(n_trials >= 5)  # Only include bins with at least 5 trials

p_value_diff <- ggplot() +
  
  # Empirical data ribbons (confidence bands around empirical data)
  geom_ribbon(data = empirical_data,
              aes(x = value_diff_bin, 
                  ymin = p_accept_emp - se, 
                  ymax = p_accept_emp + se,
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
    y="Proportion of Gaze on Gain",
    color = "",
    fill = ""
  ) +
  scale_x_continuous(guide = "prism_offset") + 
  scale_y_continuous(guide = "prism_offset", limits = c(-0.05, 1.05), breaks = seq(0, 1, 0.25)) +
  theme(
    legend.position = c(0.25, 0.85),
    plot.margin = margin(t = 20, unit = "pt")
  )

print(p_value_diff)

# Save the plot
ggsave("figures/p(gaze)/value_difference.png",
       plot = p_value_diff,
       width = 6, height = 6, dpi = 300)

ggsave("figures/p(gaze)/value_difference.svg",
       plot = p_value_diff,
       width = 6, height = 6)

