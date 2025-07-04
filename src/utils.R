library(dplyr)
library(brms)
source("src/mytheme.R")
z_score <- function(x) as.vector(scale(x))

blend_colors <- function(foreground, background = "#FFFFFF", alpha = 0.3) {
  fg_rgb <- col2rgb(foreground) / 255
  bg_rgb <- col2rgb(background) / 255
  blended_rgb <- alpha * fg_rgb + (1 - alpha) * bg_rgb
  rgb(blended_rgb[1], blended_rgb[2], blended_rgb[3])
}

report_bayes_latex <- function(model, digits = 3, ci_level = 0.95) {

  # Calculate quantiles for credible interval
  lower_q <- (1 - ci_level) / 2
  upper_q <- 1 - lower_q
  
  # Get posterior summary with explicit quantiles
  post <- as.data.frame(posterior_summary(model, probs = c(lower_q, upper_q)))
  post <- post[grepl("^b_", rownames(post)), , drop = FALSE] # fixed effects only

  # Find the quantile columns
  q_cols <- colnames(post)[grepl("^Q", colnames(post))]
  if (length(q_cols) < 2) stop("Could not find both lower and upper CI columns.")

  # Sort CI columns by value (lower first)
  ci_cols <- post[, q_cols]
  ci_low_col <- names(sort(colMeans(ci_cols)))[1]
  ci_high_col <- names(sort(colMeans(ci_cols)))[2]

  ci_label <- paste0(round(ci_level * 100), "\\%~\\text{CrI}")

  output <- apply(post, 1, function(row) {
    beta     <- row["Estimate"]
    sd       <- row["Est.Error"]
    ci_low   <- row[ci_low_col]
    ci_high  <- row[ci_high_col]

    beta_str    <- sprintf(paste0("%.", digits, "f"), beta)
    sd_str      <- sprintf(paste0("%.", digits, "f"), sd)
    ci_low_str  <- sprintf(paste0("%.", digits, "f"), ci_low)
    ci_high_str <- sprintf(paste0("%.", digits, "f"), ci_high)

    latex_str <- paste0(
      "$\\beta = ", beta_str,
      ", \\text{SD} = ", sd_str,
      ", ", ci_label,
      " [", ci_low_str, ", ", ci_high_str, "]$"
    )

    param_name <- gsub("^b_", "", rownames(post)[which(apply(post, 1, identical, row))])
    paste0(param_name, ": ", latex_str)
  })

  names(output) <- NULL
  return(output)
}

# Helper function to display LaTeX output properly (with single backslashes)
print_bayes_latex <- function(model, digits = 3, ci_level = 0.95) {
  result <- report_bayes_latex(model, digits, ci_level)
  for (i in seq_along(result)) {
    cat(result[i], "\n")
  }
}