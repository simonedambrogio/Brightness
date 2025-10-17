library(tidyverse); library(brms); library(yaml)
source("src/utils.R")
config <- read_yaml("config.yaml")

if (!file.exists("output/p(first-gaze)/m.rds")){

    # Load data
    data_behavior <- read.csv("data/processed/data_behavior.csv") 
    
    data_model <- data_behavior %>% 
    mutate(
      evZ = z_score(gain * 0.5 - loss * 0.5),
      first_is_gain = 1-first_is_loss,
      gain_is_salient = 1-SalL
    )

    # Fit model
    m <- brm(
        formula = first_is_gain ~ evZ * gain_is_salient + (evZ * gain_is_salient | subject), 
        data = data_model,
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
    path2save <- file.path(config$local$output, "p(first-gaze)")
    if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
    saveRDS(m, file.path(path2save, "m.rds"))
    save_bayes_md(m, file.path(path2save, "m.md"))
}



if (!file.exists("output/p(first-gaze)/m1.rds")){

    # Load data
    data_behavior <- read.csv("data/processed/data_behavior.csv") 
    
    data_model <- data_behavior %>% 
    mutate(
      gainZ = z_score(gain ),
      lossZ = z_score(loss),
      first_is_gain = 1-first_is_loss,
      gain_is_salient = 1-SalL
    )

    # Fit model
    m <- brm(
        formula = first_is_gain ~ gainZ + lossZ + gain_is_salient + (gainZ + lossZ + gain_is_salient | subject), 
        data = data_model,
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
    path2save <- file.path(config$local$output, "p(first-gaze)")
    if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
    saveRDS(m, file.path(path2save, "m1.rds"))
    save_bayes_md(m, file.path(path2save, "m1.md"))
}
