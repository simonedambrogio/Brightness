source("src/utils.R")
source("scripts/mediation/fit.R")
library(ggdist); library(yaml)
config <- yaml::read_yaml(file = "config.yaml")

# 1. Extract posterior samples
post_samples <- posterior_samples(m)

# 2. Calculate conditional effects for each posterior draw
# ACME (Indirect Effect)
# Effect of salience on gaze * effect of gaze on choice
post_samples$ACME <- post_samples$b_gazegainZ_salience * post_samples$b_choice_gaze_gainZ

# ADE (Direct Effect)
post_samples$ADE <- post_samples$b_choice_salience

# 3. Summarize the posterior distributions of the conditional effects

# ACME Summary
median_ACME <- median(post_samples$ACME)
ci_ACME <- quantile(post_samples$ACME, probs = c(0.025, 0.975))
sd_ACME <- sd(post_samples$ACME)

# ADE Summary
median_ADE <- median(post_samples$ADE)
ci_ADE <- quantile(post_samples$ADE, probs = c(0.025, 0.975))
sd_ADE <- sd(post_samples$ADE)

# Print the summaries
cat("Conditional Indirect Effects (ACME):\n")
cat("Estimate =", round(median_ACME, 4), " 95% CI [", round(ci_ACME[1], 4), ",", round(ci_ACME[2], 4), "]\n")
cat("$\\beta =", round(median_ACME, 3), "\\text{SD}=", round(sd_ACME, 3), "95\\%\\text{CrI}~[", round(ci_ACME[1], 3), ",", round(ci_ACME[2], 3), "]$")

cat("Conditional Direct Effects (ADE):\n")
cat(" Brightness: Estimate =", round(median_ADE, 4), " 95% CI [", round(ci_ADE[1], 4), ",", round(ci_ADE[2], 4), "]\n")
cat("$\\beta =", round(median_ADE, 3), "\\text{SD}=", round(sd_ADE, 3), "95\\%\\text{CrI}~[", round(ci_ADE[1], 3), ",", round(ci_ADE[2], 3), "]$")


color_densities <- config$colors$`gain-salient`

# Salience -> Gaze
data.frame(post=post_samples$b_gazegainZ_salience) %>% 
  ggplot(aes(post)) +
  stat_halfeye(fill=blend_colors(color_densities, alpha=0.4), color=color_densities, alpha=0.8) +
  geom_vline(xintercept=0, linetype=2) +
  mytheme() +
  theme(
    axis.title.y = element_blank(), # Remove y-axis label
    axis.text.y = element_blank(),  # Remove y-axis text (numbers)
    axis.title.x = element_blank(), # Remove x-axis label
    axis.ticks.y = element_blank(),  # Optionally remove y-axis ticks as well
    axis.ticks.x = element_blank(),  # Optionally remove y-axis ticks as well
    axis.text = element_text(size = 20),
    axis.line = element_blank()
  )

ggsave("figures/mediation/salience->gaze.svg", width = 4, height = 2.5)

# Gaze -> Accept
data.frame(post=post_samples$b_choice_gaze_gainZ) %>% 
  ggplot(aes(post)) +
  stat_halfeye(fill=blend_colors(color_densities, alpha=0.4), color=color_densities, alpha=0.8) +
  geom_vline(xintercept=0, linetype=2) +
  mytheme() +
  theme(
    axis.title.y = element_blank(), # Remove y-axis label
    axis.text.y = element_blank(),  # Remove y-axis text (numbers)
    axis.title.x = element_blank(), # Remove x-axis label
    axis.ticks.y = element_blank(),  # Optionally remove y-axis ticks as well
    axis.ticks.x = element_blank(),  # Optionally remove y-axis ticks as well
    axis.text = element_text(size = 20),
    axis.line = element_blank()
  )

ggsave("figures/mediation/gaze->accept.svg", width = 4, height = 2.5)

# Salience -> Gaze -> Accept
data.frame(post=post_samples$ACME) %>% 
  ggplot(aes(post)) +
  stat_halfeye(fill=blend_colors(color_densities, alpha=0.4), color=color_densities, alpha=0.8) +
  geom_vline(xintercept=0, linetype=2) +
  mytheme() +
  theme(
    axis.title.y = element_blank(), # Remove y-axis label
    axis.text.y = element_blank(),  # Remove y-axis text (numbers)
    axis.title.x = element_blank(), # Remove x-axis label
    axis.ticks.y = element_blank(),  # Optionally remove y-axis ticks as well
    axis.ticks.x = element_blank(),  # Optionally remove y-axis ticks as well
    axis.text = element_text(size = 20),
    axis.line = element_blank()
  )
  
ggsave("figures/mediation/salience->gaze->accept.svg", width = 4, height = 2.5)


# Salience -> Accept
data.frame(post=post_samples$ADE) %>% 
  ggplot(aes(post)) +
  stat_halfeye(fill=blend_colors(color_densities, alpha=0.4), color=color_densities, alpha=0.8) +
  geom_vline(xintercept=0, linetype=2) +
  mytheme() +
  theme(
    axis.title.y = element_blank(), # Remove y-axis label
    axis.text.y = element_blank(),  # Remove y-axis text (numbers)
    axis.title.x = element_blank(), # Remove x-axis label
    axis.ticks.y = element_blank(),  # Optionally remove y-axis ticks as well
    axis.ticks.x = element_blank(),  # Optionally remove y-axis ticks as well
    axis.text = element_text(size = 20),
    axis.line = element_blank()
  )

ggsave("figures/mediation/salience->accept.svg", width = 4, height = 2.5)
