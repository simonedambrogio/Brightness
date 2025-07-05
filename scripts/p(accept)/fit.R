library(stringr); library(yaml); library(brms); library(dplyr)
source("src/utils.R")
config <- read_yaml("config.yaml")

# p(accept) ~ gain + loss + gaze + salience + first_gaze
if (!file.exists("data/results/p(accept)/m.rds")){

    data_behavior <- read.csv("data/processed/data_behavior.csv") 

    data_model <- data_behavior %>% 
        mutate(
            gainZ = scale(gain) %>% as.vector(),
            lossZ = scale(loss) %>% as.vector(),
            gaze_gainZ = scale(gaze_gain) %>% as.vector(),
            first_is_gain = 1-first_is_loss,
            SalL = scale(SalL) %>% as.vector()
        )
        
    m <- brm(
        formula = choice ~ gainZ + lossZ + gaze_gainZ + first_is_gain + SalL + (
            gainZ + lossZ + gaze_gainZ + first_is_gain + SalL | subject
        ), 
        prior = c(
            prior_string("normal(0,2.5)", class = "b"),
            prior_string("normal(0,2.5)", class = "Intercept")
        ),
        iter = 20000,
        data = data_model, 
        cores = 4, refresh = 10, 
        family = bernoulli(link = "logit")
    )
    path2save <- file.path(config$local$data, "results/p(accept)")
    if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
    saveRDS(m, file.path(path2save, "m.rds"))
    save_bayes_md(m, file.path(path2save, "m.md"))
}

# p(accept) ~ gain + loss + salience
if (!file.exists("data/results/p(accept)/m.1.rds")){

    data_behavior <- read.csv("data/processed/data_behavior.csv") 

    data_model <- data_behavior %>% 
        mutate(
            gainZ = z_score(gain) %>% as.vector(),
            lossZ = z_score(loss) %>% as.vector(),
            gain_is_salient = 1-SalL
        )
        
    m <- brm(
        formula = choice ~ gainZ + lossZ + gain_is_salient + (
            gainZ + lossZ + gain_is_salient | subject
        ), 
        prior = c(
            prior_string("normal(0,2.5)", class = "b"),
            prior_string("normal(0,2.5)", class = "Intercept")
        ),
        iter = 20000,
        data = data_model, 
        cores = 4, refresh = 10, 
        family = bernoulli(link = "logit")
    )
    path2save <- file.path(config$local$data, "results/p(accept)")
    if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
    saveRDS(m, file.path(path2save, "m.1.rds"))
    save_bayes_md(m, file.path(path2save, "m.1.md"))
}

# p(accept) ~ ev + salience + gaze + first_gaze
if (!file.exists("data/results/p(accept)/m1.rds")){
  
  data_behavior <- read.csv("data/processed/data_behavior.csv") 
  
  data_model <- data_behavior %>% 
    mutate(
      evZ = z_score(gain * 0.5 - loss * 0.5),
      gaze_gainZ = z_score(gaze_gain),
      first_is_gain = 1-first_is_loss,
      gain_is_salient = 1-SalL
    )
  
  m <- brm(
    formula = choice ~ evZ + gain_is_salient + gaze_gainZ + first_is_gain + (
      evZ + gain_is_salient + gaze_gainZ + first_is_gain | subject
    ), 
    prior = c(
      prior_string("normal(0,2.5)", class = "b"),
      prior_string("normal(0,2.5)", class = "Intercept")
    ),
    iter = 20000,
    data = data_model, 
    cores = 4, refresh = 10, 
    family = bernoulli(link = "logit")
  )
  path2save <- file.path(config$local$data, "results/p(accept)")
  if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
  saveRDS(m, file.path(path2save, "m1.rds"))
  save_bayes_md(m, file.path(path2save, "m1.md"))
}

# p(accept) ~ ev + salience
if (!file.exists("data/results/p(accept)/m1.1.rds")){
  
  data_behavior <- read.csv("data/processed/data_behavior.csv") 
  
  data_model <- data_behavior %>% 
    mutate(
      evZ = z_score(gain * 0.5 - loss * 0.5),
      gaze_gainZ = z_score(gaze_gain),
      first_is_gain = 1-first_is_loss,
      gain_is_salient = 1-SalL
    )
  
  m <- brm(
    formula = choice ~ evZ + gain_is_salient + (
      evZ + gain_is_salient | subject
    ), 
    prior = c(
      prior_string("normal(0,2.5)", class = "b"),
      prior_string("normal(0,2.5)", class = "Intercept")
    ),
    iter = 20000,
    data = data_model, 
    cores = 4, refresh = 10, 
    family = bernoulli(link = "logit")
  )
  path2save <- file.path(config$local$data, "results/p(accept)")
  if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
  saveRDS(m, file.path(path2save, "m1.1.rds"))
  save_bayes_md(m, file.path(path2save, "m1.1.md"))
}
