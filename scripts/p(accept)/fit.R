library(stringr); library(yaml); library(brms); library(dplyr)
config <- read_yaml("config.yaml")


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
} else {
  print("Loading model...")
  m <- readRDS("data/results/p(accept)/m.rds")
}
