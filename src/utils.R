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

  ci_label <- paste0(round(ci_level * 100), "\\%~\\mathrm{CrI}")

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
      ", \\mathrm{SD} = ", sd_str,
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

print_vector_latex <- function(posterior_vector, param_name = "Parameter", digits = 3, ci_level = 0.95) {
  # Calculate summary statistics
  mean_val <- mean(posterior_vector)
  sd_val <- sd(posterior_vector)
  
  # Calculate credible interval quantiles
  lower_q <- (1 - ci_level) / 2
  upper_q <- 1 - lower_q
  ci_low <- quantile(posterior_vector, lower_q)
  ci_high <- quantile(posterior_vector, upper_q)
  
  # Format values
  mean_str <- sprintf(paste0("%.", digits, "f"), mean_val)
  sd_str <- sprintf(paste0("%.", digits, "f"), sd_val)
  ci_low_str <- sprintf(paste0("%.", digits, "f"), ci_low)
  ci_high_str <- sprintf(paste0("%.", digits, "f"), ci_high)
  
  # Create CI label
  ci_label <- paste0(round(ci_level * 100), "\\%~\\mathrm{CrI}")
  
  # Format LaTeX string
  latex_str <- paste0(
    "$\\beta = ", mean_str,
    ", \\mathrm{SD} = ", sd_str,
    ", ", ci_label,
    " [", ci_low_str, ", ", ci_high_str, "]$"
  )
  
  # Print the result
  result <- paste0(param_name, ": ", latex_str)
  cat(result, "\n")
}

print_latex <- function(inp, digits = 3, ci_level = 0.95, param_name = ""){
  # Check if input is a brms model
  if (inherits(inp, "brmsfit")) {
    # Use print_bayes_latex for brms models
    print_bayes_latex(inp, digits = digits, ci_level = ci_level)
  } else if (is.numeric(inp) && is.vector(inp)) {
    # Use print_vector_latex for numeric vectors
    print_vector_latex(inp, param_name = param_name, digits = digits, ci_level = ci_level)
  } else {
    stop("Input must be either a brmsfit object or a numeric vector")
  }
}

save_bayes_md <- function(model, path, digits = 3, ci_level = 0.95) {
  result <- report_bayes_latex(model, digits, ci_level)
  writeLines(result, path)
}
