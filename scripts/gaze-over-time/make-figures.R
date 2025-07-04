library(lmerTest)
fix <- read.csv(file = "data/processed/gaze_allocation.csv")

# ------- Time Lock ------- #
get_event_related_gaze <- function(fix, Time, std = FALSE) {
  # Time <- seq(50, 6000, 200)
  if (std) {
    fix$std_time <- 0
    for (i in 2:length(Time)) {
      fix$std_time[fix$std_cum_time > Time[i - 1] & fix$std_cum_time <= Time[i]] <- Time[i]
    }
    
    fix <- fix %>%
      group_by(subject, trial) %>%
      mutate(cum_tim_diff = c(0, diff(cum_time)), 
             cum_time_gain = cumsum(cum_tim_diff * (fix_type1 == "gain")), 
             cum_time_loss = cumsum(cum_tim_diff * (fix_type1 == "loss")), 
             p_gain = cum_time_gain/max(cum_time),
             p_loss = cum_time_loss/max(cum_time)) %>%
      ungroup()
    
    return(fix)
  } else {
    fix$time_from_onset <- 0
    for (i in 2:length(Time)) {
      fix$time_from_onset[fix$cum_time > Time[i - 1] & fix$cum_time <= Time[i]] <- Time[i]
    }
    fix$time_from_onset[fix$cum_time < 50] <- 0
    
    fix <- fix %>%
      group_by(subject, trial) %>%
      mutate(time_to_choice = time_from_onset - max(time_from_onset), 
             cum_tim_diff = c(0, diff(cum_time)), 
             cum_time_gain = cumsum(cum_tim_diff * (fix_type1 == "gain")), 
             cum_time_loss = cumsum(cum_tim_diff * (fix_type1 == "loss")),
             p_gain = cum_time_gain/rt, p_loss = cum_time_loss/rt) %>%
      ungroup()
    
    return(fix)
  }
}


# ------- Time from Onset ------- #
df_time_from_onset <- function(fix, Time = seq(0, 6000, 150)) {
  
  fix1 <- get_event_related_gaze(fix, Time) %>% 
    select(subject, trial, p_gain, p_loss, SalL, time_from_onset) %>% 
    filter(time_from_onset<=2000)
  
  # P-value Gain
  p_value_gain <- c()
  print('Compute p-value gain')
  for( tfo in sort(unique(fix1$time_from_onset)) ){
    cat(tfo, '\r')
    
    # T-test
    # d <- fix1 %>%
    #   mutate(Sal=ifelse(SalL=='Loss is Salient', 'Not Salient', 'Salient')) %>%
    #   filter(time_from_onset==tfo)
    # tt <- t.test(p_gain~Sal, data=d, alternative = "greater")
    # p_value_gain <- c(p_value_gain, tt$p.value)
    
    # mixed-effect model
    m <- lmer(p_gain ~ Sal + (1|subject),
              data=fix1 %>%
                mutate(Sal=ifelse(SalL=='Loss is Salient', 'Not Salient', 'Salient')) %>%
                filter(time_from_onset==tfo)
    )
    fit <- as(m, "lmerModLmerTest")
    p_value_gain <- c(p_value_gain, anova(fit)$`Pr(>F)`)
  }
  
  # P-value Loss
  p_value_loss <- c()
  print('Compute p-value loss')
  for( tfo in sort(unique(fix1$time_from_onset)) ){
    cat(tfo, '\r')
    
    # T-test
    # d <- fix1 %>%
    #   mutate(Sal=ifelse(SalL=='Loss is Salient', 'Salient', 'Not Salient')) %>%
    #   filter(time_from_onset==tfo)
    # tt <- t.test(p_loss~Sal, data=d, alternative = "greater")
    # p_value_loss <- c(p_value_loss, tt$p.value)
    
    # mixed-effect model
    m <- lmer(p_loss ~ Sal + (1|subject),
              data=fix1 %>%
                mutate(Sal=ifelse(SalL=='Loss is Salient', 'Salient', 'Not Salient')) %>%
                filter(time_from_onset==tfo)
    )
    fit <- as(m, "lmerModLmerTest")
    p_value_loss <- c(p_value_loss, anova(fit)$`Pr(>F)`)
  }
  
  
  dat_from_onset <- rbind(Rmisc::summarySE(fix1, measurevar = "p_gain", groupvars = c("time_from_onset", "SalL")) %>%
                            mutate(attr = "Gain", p_value=rep(p_value_gain, each=2)) %>%
                            rename(prop = p_gain), Rmisc::summarySE(fix1, measurevar = "p_loss", groupvars = c("time_from_onset", "SalL")) %>%
                            mutate(attr = "Loss", p_value=rep(p_value_loss, each=2)) %>%
                            rename(prop = p_loss)) %>%
    mutate(SalL = ifelse(SalL == "Loss is Salient", SalL, "Gain is Salient"), 
           SalL = case_when(SalL =="Loss is Salient" & attr == "Loss" ~ "Salient", 
                            SalL == "Gain is Salient" & attr == "Gain" ~ "Salient", 
                            T ~ "Not Salient"))
  
  return(dat_from_onset)
}

df_tfo <- df_time_from_onset(fix)


dfplot <- df_tfo %>%
  mutate(is_sing=ifelse(p_value<0.05, 1, 0),
         y_text=ifelse(attr=='Gain', .22, .22),
         lab=ifelse(p_value<0.05, '*', ''),
         SalL=factor(SalL, levels = c('Salient', 'Not Salient'))) %>% 
  mutate(time_from_onset=time_from_onset/1000) %>% 
  mutate(
    sal = case_when(
      SalL=="Salient" & attr=="Gain" ~ "Gain Salient",
      SalL=="Salient" & attr=="Loss" ~ "Loss Salient",
      
      SalL=="Not Salient" & attr=="Gain" ~ "Loss Salient",
      SalL=="Not Salient" & attr=="Loss" ~ "Gain Salient"
    )
  )

# Gaze on Gain -----------------------------------------------------------------
gazegain <- dfplot %>%
  filter(attr =="Gain") %>% 
    ggplot() +
    geom_line(aes(time_from_onset, prop, color=sal), linewidth=2) +
    geom_text(aes(time_from_onset, y=y_text, label=lab), size=10) +
    geom_pointrange(aes(time_from_onset, prop, color=sal, ymin=prop-ci, ymax=prop+ci), size=1) +
    geom_text(
      data = data.frame(x=c(0.8, 1.5), y = c(0.18, 0.1), label=c("Gain brighter", "Loss brighter"), sal=c("Gain Salient", "Loss Salient") ), 
      aes(x, y, label = label, color=sal), size = 6, family = "Avenir Light"
    ) + 
    mytheme() + 
    labs(color='Condition', alpha='', y='Proportion Gaze on Gain', x='Time from Onset (s)', title='') +
    theme(legend.position='none', text = element_text(size=20)) +
    scale_color_manual(values = c(config$colors$`gain-salient`, config$colors$`loss-salient`)) +
    scale_x_continuous(guide = "prism_offset") +
    scale_y_continuous(
      guide = "prism_offset", breaks = seq(0, 0.2, 0.1), limits = c(-0.04, 0.24)
    ); print(gazegain_brightness)


# Save the plot
ggsave("figures/gaze-over-time/look_at_gain.png",
       plot = gazegain,
       width = 5, height = 5.5, dpi = 300)

ggsave("figures/gaze-over-time/look_at_gain.svg",
       plot = gazegain,
       width = 5, height = 5.5)

# Gaze on Loss -----------------------------------------------------------------
gazeloss <- dfplot %>%
  filter(attr =="Loss") %>% 
  ggplot() +
  geom_line(aes(time_from_onset, prop, color=sal), linewidth=2) +
  geom_text(aes(time_from_onset, y=y_text, label=lab), size=10) +
  geom_pointrange(aes(time_from_onset, prop, color=sal, ymin=prop-ci, ymax=prop+ci), size=1) +
  geom_text(
    data = data.frame(x=c(1.6, 0.8), y = c(0.1, 0.18), label=c("Gain brighter", "Loss brighter"), sal=c("Gain Salient", "Loss Salient") ), 
    aes(x, y, label = label, color=sal), size = 6, family = "Avenir Light"
  ) + 
  mytheme() + 
  labs(color='Attribute', alpha='', y='Proportion Gaze on Loss', x='Time from Onset (s)', title='') +
  theme(legend.position='none', text = element_text(size=20)) + 
  scale_color_manual(values = c(config$colors$`gain-salient`, config$colors$`loss-salient`)) +
  scale_x_continuous(guide = "prism_offset") +
  scale_y_continuous(
    guide = "prism_offset", breaks = seq(0, 0.2, 0.1), limits = c(-0.04, 0.24))


# Save the plot
ggsave("figures/gaze-over-time/look_at_loss.png",
       plot = gazeloss,
       width = 5, height = 5.5, dpi = 300)

ggsave("figures/gaze-over-time/look_at_loss.svg",
       plot = gazeloss,
       width = 5, height = 5.5)

