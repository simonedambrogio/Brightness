library(tidyverse); library(brms); library(yaml)
config <- read_yaml("config.yaml")

if (!file.exists("data/results/p(gaze)/m.rds")){

    # Load data
    fix <- read.csv(file = "data/processed/gaze_allocation.csv")
    data_fixations = fix %>% 
    filter(fix_type1!="elsewhere") %>% 
    mutate(
        look_at_gain = ifelse(fix_type1=="gain", 1, 0),
        gainZ = scale(gain) %>% as.vector(),
        lossZ = scale(loss) %>% as.vector(),
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
    path2save <- file.path(config$local$data, "results/p(gaze)")
    if (!dir.exists(path2save)) dir.create(path2save, recursive = TRUE)
    saveRDS(m, file.path(path2save, "m.rds"))
} else {  
    print("Loading pre-computed model")
    m = readRDS("data/results/p(gaze)/m.rds")
}

