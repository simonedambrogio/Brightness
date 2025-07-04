This readme file was generated on 2024-01-15 by Simone D'Ambrogio

# GENERAL INFORMATION

* Title of Dataset: Visual Salience Effects on Economic Choice: Eye-tracking and Computational Modeling Data

## Author/Principal Investigator Information
Name: Simone D'Ambrogio
ORCID: [TO BE FILLED]
Institution: University of Oxford
Address: Department of Experimental Psychology, University of Oxford, Oxford, UK
Email: simone.dambrogio@psy.ox.ac.uk

## Author/Associate or Co-investigator Information
Name: Michael Platt
ORCID: [TO BE FILLED]
Institution: University of Pennsylvania
Address: University of Pennsylvania, Philadelphia, PA, USA
Email: [TO BE FILLED]

## Author/Associate or Co-investigator Information
Name: Feng Sheng
ORCID: [TO BE FILLED]
Institution: Zhejiang University
Address: Zhejiang University, Hangzhou, China
Email: [TO BE FILLED]

* Date of data collection: [TO BE FILLED - approximate date range]
* Geographic location of data collection: Online (participants recruited via Prolific)
* Information about funding sources that supported the collection of the data: [TO BE FILLED]

# SHARING/ACCESS INFORMATION

* Licenses/restrictions placed on the data: [TO BE FILLED]
* Links to publications that cite or use the data: [TO BE FILLED WHEN PUBLISHED]
* Links to other publicly accessible locations of the data: Data will be made available on OSF upon publication
* Links/relationships to ancillary data sets: None
* Was data derived from another source? No
* Recommended citation for this dataset: [TO BE FILLED WHEN PUBLISHED]

# CODE & ANALYSIS REPOSITORY OVERVIEW

**Note: This GitHub repository contains only the analysis code and figure generation scripts. The raw data will be made available separately on OSF.**

## Repository Structure

* **scripts/**: Analysis and figure generation code
  * `gaze-over-time/`: Scripts for analyzing gaze patterns over time
  * `mediation/`: Code for causal mediation analysis
  * `p(accept)/`: Scripts for choice probability analysis
  * `p(gaze)/`: Scripts for gaze probability analysis  
  * `response-time/`: Response time analysis code
  * `vam/`: Visual Accumulator Model (VAM) implementation and analysis
    * `train.py`: Model training script
    * `predict_choices_rts.py`: Generate model predictions
    * `analyze_network_weights.jl`: Network weight analysis (Julia)
    * `save_network_weights.py`: Extract and save network weights

* **src/**: Core utilities and model definitions
  * `vam/`: Visual Accumulator Model implementation in JAX
    * `models.py`: Neural network architectures
    * `training.py`: Training procedures
    * `task_data.py`: Data loading and preprocessing
    * `lba.py`: Linear Ballistic Accumulator implementation
    * `metrics.py`: Model evaluation metrics
    * `utils.py`: General utilities
  * `utils.R`: R utility functions
  * `mytheme.R`: Custom ggplot themes

* **figures/**: Generated figures (created by scripts)
  * `gaze-over-time/`: Gaze dynamics figures
  * `mediation/`: Mediation analysis visualizations
  * `p(accept)/`: Choice probability figures
  * `p(gaze)/`: Gaze probability figures
  * `response-time/`: Response time analysis figures
  * `vam/`: VAM model analysis figures

* **config.yaml**: Configuration file with data paths and styling parameters

## File Relationships
* Scripts in `scripts/` generate figures saved in `figures/`
* Scripts utilize functions and utilities from `src/`
* All paths and parameters are configured via `config.yaml`
* VAM model scripts work together: `train.py` → `save_network_weights.py` → `analyze_network_weights.jl` → `predict_choices_rts.py`

## Additional Information
* The `docs/` folder contains manuscript files but will not be included in the public repository
* Raw data files will be hosted separately on OSF due to size and privacy considerations
* All analysis code is designed to work with the data structure as it will be provided on OSF

# METHODOLOGICAL INFORMATION

## Description of methods used for collection/generation of data
Participants completed an online eye-tracking experiment where they made risky gambling decisions. In each trial, participants decided whether to accept or reject gambles with equal probability (50/50) of winning or losing money. Visual salience was manipulated by varying the brightness or font-size of potential gains vs. losses. Eye movements were recorded using WebGazer.js, a JavaScript library that uses webcams to infer eye-gaze locations in real time. The experiment was implemented using jsPsych and participants were recruited via Prolific.

## Methods for processing the data
Raw gaze data was preprocessed with multiple quality control steps including response time filtering (>300ms), sampling rate requirements (>8Hz), handling missing data (NA values), and region-of-interest analysis. Gaze positions were denoised using a 300ms median filter. Choice data was analyzed using Bayesian multilevel regression models. Causal mediation analysis was performed to separate direct and indirect effects of salience on choice. Computational modeling employed a Visual Accumulator Model (VAM) combining convolutional neural networks with Linear Ballistic Accumulator models.

## Instrument- or software-specific information needed to interpret the data
* **R version 4.x+** with packages: brms, ggplot2, dplyr, tidyr, and others
* **Python 3.8+** with packages: JAX, PyTorch, NumPy, Pandas, Matplotlib
* **Julia 1.6+** for network weight analysis
* **WebGazer.js** for eye-tracking data collection
* **jsPsych** for experiment presentation
* **VAM (Visual Accumulator Model)** custom implementation in JAX

## Standards and calibration information
* Eye-tracking calibration: 13-point calibration-validation scheme with 70% accuracy requirement
* Response time filtering: minimum 300ms response time
* Gaze data quality: minimum 8Hz sampling rate, maximum 50% missing data per trial
* Model validation: Bayesian model fitting with multiple chains and convergence diagnostics

## Environmental/experimental conditions
* Online experiment conducted on participants' personal computers
* Webcam-based eye-tracking using standard consumer hardware
* Screen resolution and viewing distance varied across participants
* Participants required normal or corrected-to-normal vision

## Quality-assurance procedures performed on the data
* Multi-stage data exclusion criteria for response times, sampling rates, and gaze data quality
* Bayesian model diagnostics including R-hat values and effective sample sizes
* Cross-validation procedures for computational models
* Robustness checks across different model specifications

## People involved with sample collection, processing, analysis and/or submission
* Simone D'Ambrogio: Study design, data collection, analysis, manuscript preparation
* Michael Platt: Study design, supervision, manuscript review
* Feng Sheng: Study design, analysis consultation, manuscript review

# COMPUTATIONAL MODEL INFORMATION

## Model Architecture
The Visual Accumulator Model (VAM) consists of:
* **Visual Processing Module**: 6-layer Convolutional Neural Network (CNN) with 64, 64, 128, 128, 128, and 256 features, followed by fully-connected layer with 1024 units
* **Decision Module**: Linear Ballistic Accumulator (LBA) with gaze-dependent evidence accumulation
* **Gaze Integration**: Attention-weighted evidence accumulation with fitted gaze-bias parameter λ

## Model Parameters
* Non-decision time (t0), threshold component (c), starting point variability (a)
* Gaze-bias parameter (λ) for attention-weighted evidence accumulation
* CNN weights initialized from VGG16 pre-trained on ImageNet

## Training Procedure
* Data augmentation with random displacement and rotation (±15°)
* Joint optimization of CNN parameters and LBA posterior distributions
* Variational inference using Evidence Lower Bound (ELBO) maximization
* Bayesian framework for parameter uncertainty quantification
