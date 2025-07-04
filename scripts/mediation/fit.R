source("src/utils.R")
library(stringr); library(lme4); library(mediation)
library(tidyverse); library(brms); library(yaml)
config <- read_yaml("config.yaml")

if (!file.exists("data/results/mediation/m.rds")){
    print("Fitting model...")

    # Mediation analysis ===========================================================
    data_mediation <- read.csv(file = "data/processed/data_behavior.csv") %>% as_tibble() %>% 
    mutate(gainZ = z_score(gain), lossZ = z_score(loss), gaze_lossZ = z_score(gaze_loss), gaze_gainZ=z_score(gaze_gain)) %>% 
    mutate(salience = 1-SalL) %>% 
    dplyr::select(choice, gainZ, lossZ, gaze_gainZ, subject, salience)

    # Bayesian mediation analysis ---

    # Mediator model formula
    bf_mediator <- bf(gaze_gainZ ~ salience + gainZ + lossZ + (salience + gainZ + lossZ | subject))

    # Outcome model formula
    bf_outcome <- bf(
        choice ~ salience + gaze_gainZ + gainZ + lossZ + (salience + gaze_gainZ + gainZ + lossZ | subject),
        family = bernoulli(link = "logit"))

    # Fit the model (adjust iter, chains, cores, priors, control as needed)
    # Using more iterations (e.g., 4000 total, 2000 warmup) is often good practice
    brms_mediation_fit <- brm(
        bf_mediator + bf_outcome + set_rescor(FALSE),
        data = data_mediation,
        iter = 20000, warmup = 10000, chains = 4, cores = 4,
        control = list(adapt_delta = 0.95), # Increase if divergences occur
        seed = 123 # For reproducibility
    )

    path2save <- file.path(config$local$data, "results/mediation")
    if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
    saveRDS(brms_mediation_fit, file = file.path(path2save, "m.rds"))
} else {
    print("Loading model...")
    m <- readRDS("data/results/mediation/m.rds")
}

# gazegainZ_Intercept: $\beta = -0.062, \text{SD} = 0.029, 95\%~\text{CrI} [-0.119, -0.005]$ 
# choice_Intercept: $\beta = -0.895, \text{SD} = 0.166, 95\%~\text{CrI} [-1.222, -0.573]$ 
# gazegainZ_salience: $\beta = 0.162, \text{SD} = 0.047, 95\%~\text{CrI} [0.070, 0.254]$ 
# gazegainZ_gainZ: $\beta = 0.040, \text{SD} = 0.015, 95\%~\text{CrI} [0.010, 0.069]$ 
# gazegainZ_lossZ: $\beta = -0.054, \text{SD} = 0.012, 95\%~\text{CrI} [-0.078, -0.029]$ 
# choice_salience: $\beta = 0.784, \text{SD} = 0.147, 95\%~\text{CrI} [0.498, 1.076]$ 
# choice_gaze_gainZ: $\beta = 0.274, \text{SD} = 0.044, 95\%~\text{CrI} [0.187, 0.362]$ 
# choice_gainZ: $\beta = 2.237, \text{SD} = 0.166, 95\%~\text{CrI} [1.924, 2.570]$ 
# choice_lossZ: $\beta = -2.340, \text{SD} = 0.165, 95\%~\text{CrI} [-2.669, -2.027]$
