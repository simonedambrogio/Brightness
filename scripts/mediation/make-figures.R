source("src/utils.R")
source("scripts/mediation/fit.R")
library(ggdist); library(yaml)
config <- yaml::read_yaml(file = "config.yaml")

# 1. Extract posterior samples
m <- readRDS("output/mediation/m.rds")
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
cat("ACME =", round(median_ACME, 4), " 95% CI [", round(ci_ACME[1], 4), ",", round(ci_ACME[2], 4), "]\n")
cat("$\\mathrm{ACME} =", round(median_ACME, 3), "\\mathrm{SD}=", round(sd_ACME, 3), "95\\%\\mathrm{CrI}~[", round(ci_ACME[1], 3), ",", round(ci_ACME[2], 3), "]$")

cat("Conditional Direct Effects (ADE):\n")
cat(" ADE =", round(median_ADE, 4), " 95% CI [", round(ci_ADE[1], 4), ",", round(ci_ADE[2], 4), "]\n")
cat("$\\mathrm{ADE} =", round(median_ADE, 3), "\\mathrm{SD}=", round(sd_ADE, 3), "95\\%\\mathrm{CrI}~[", round(ci_ADE[1], 3), ",", round(ci_ADE[2], 3), "]$")

# 4. Calculate proportion of salience effect mediated by gaze


# Calculate proportion of salience effect mediated by gaze
# From the mediation analysis coefficients:
salience_to_gaze <- post_samples$b_gazegainZ_salience
gaze_to_choice <- post_samples$b_choice_gaze_gainZ
direct_effect <- post_samples$b_choice_salience

# Indirect effect (mediated by gaze)
indirect_effect <- post_samples$ACME

# Total effect
total_effect <- direct_effect + indirect_effect

# Proportion mediated by gaze
proportion_mediated <- indirect_effect / total_effect
percentage_mediated <- proportion_mediated * 100

mean(proportion_mediated)

print(paste("Proportion of salience effect mediated by gaze:", round(mean(proportion_mediated), 3)))
print(paste("Percentage of salience effect mediated by gaze:", round(mean(percentage_mediated), 1), "%"))


# 5. Make figures
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


# Subject level correlation
data_behavior <- read.csv(file = "data/processed/data_behavior.csv") %>% as_tibble()

data_bias <- data_behavior %>% 
  group_by(subject) %>% 
  summarise(
    gaze_bias =  mean(gaze_gain[!SalL]) - mean(gaze_gain[SalL]),
    choice_bias = mean(choice[!SalL]) - mean(choice[SalL])
  )

r <- cor.test(data_bias$gaze_bias, data_bias$choice_bias)

data_bias %>% 
  ggplot() +
  geom_point(aes(gaze_bias, choice_bias), size=6, alpha=0.5) +
  geom_smooth(aes(gaze_bias, choice_bias), se=F, method = "lm", color="darkorange", linetype=1) +
  geom_text(
    data = data.frame(x=0.6, y=1.3, label=str_c("r = ", round(r$estimate, 2), ", p = ", format(r$p.value, scientific=TRUE, digits=2))),
    aes(x,y,label = label), size=6, family = "Avenir Light"
  ) +
  labs(x="Gaze bias", y = "Choice Bias") +
  mytheme() +
  scale_x_continuous(guide = "prism_offset", limits = c(-0.4, 0.9), breaks = seq(-0.3, 0.6, 0.3)) +
  scale_y_continuous(guide = "prism_offset", limits = c(-0.8, 1.5), breaks = seq(-0.5, 1, 0.5))


ggsave("figures/mediation/corplot.png", width = 6, height = 6, dpi = 300)
ggsave("figures/mediation/corplot.svg", width = 6, height = 6)
