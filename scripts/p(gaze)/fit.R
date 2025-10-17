library(tidyverse); library(brms); library(yaml)
source("src/utils.R")
config <- read_yaml("config.yaml")


if (!file.exists("output/p(gaze)/m.rds")){
  
  # Load data
  fix <- read.csv(file = "data/processed/gaze_allocation.csv")
  data_fixations = fix %>% 
    filter(fix_type1!="elsewhere") %>% 
    mutate(
      look_at_gain = ifelse(fix_type1=="gain", 1, 0),
      evZ = z_score(gain * 0.5 - loss * 0.5),
      gain_is_salient = ifelse(SalL=="Loss is Salient", 0, 1)
    ) %>% 
    group_by(subject) %>% 
    filter(n()>100) %>% ungroup() %>% 
    select(subject, look_at_gain, evZ, gain_is_salient) %>% 
    mutate(subject = data.table::rleid(subject))
  
  # Fit model
  m <- brm(
    formula = look_at_gain ~ evZ + gain_is_salient + (evZ + gain_is_salient | subject),
    data = data_fixations,
    family = bernoulli(link = "logit"),
    prior = c(
      prior_string("normal(0,2.5)", class = "b"),
      prior_string("normal(0,2.5)", class = "Intercept")
    ),
    cores = 4, 
    refresh = 10,
    iter = 20000,
    seed = 123
  )
  path2save <- file.path(config$local$output, "p(gaze)")
  if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
  saveRDS(m, file.path(path2save, "m.rds"))
  save_bayes_md(m, file.path(path2save, "m.md"))
}


if (!file.exists("output/p(gaze)/m1.rds")){

    # Load data
    fix <- read.csv(file = "data/processed/gaze_allocation.csv")
    data_fixations = fix %>% 
    filter(fix_type1!="elsewhere") %>% 
    mutate(
        look_at_gain = ifelse(fix_type1=="gain", 1, 0),
        gainZ = z_score(gain),
        lossZ = z_score(loss),
        gain_is_salient = ifelse(SalL=="Loss is Salient", 0, 1)
    ) %>% 
    group_by(subject) %>% 
    filter(n()>100) %>% ungroup() %>% 
    select(subject, look_at_gain, gainZ, lossZ, gain_is_salient) %>% 
    mutate(subject = data.table::rleid(subject))

    # Fit model
    m <- brm(
        formula = look_at_gain ~ gainZ + lossZ + gain_is_salient + (gainZ + lossZ + gain_is_salient | subject),
        data = data_fixations,
        family = bernoulli(link = "logit"),
        prior = c(
            prior_string("normal(0,2.5)", class = "b"),
            prior_string("normal(0,2.5)", class = "Intercept")
        ),
        cores = 4, 
        refresh = 10,
        iter = 20000,
        seed = 123
    )
    path2save <- file.path(config$local$output, "p(gaze)")
    if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
    saveRDS(m, file.path(path2save, "m1.rds"))
    save_bayes_md(m, file.path(path2save, "m1.md"))
}

