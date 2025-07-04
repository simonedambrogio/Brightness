library(ggprism)

mytheme <- function(palette = "black_and_white", 
                    base_size = 14, 
                    base_family = "Avenir Light",
                    base_fontface = "plain", 
                    base_line_size = 0.3, 
                    base_rect_size = base_size/14,
                    axis_text_angle = 0, 
                    border = FALSE) {
  
  # Helper function to check boolean values
  is_bool <- function(x) {
    rlang::is_logical(x, n = 1) && !is.na(x)
  }
  
  # Validate axis text angle
  angle <- axis_text_angle[1]
  if (!angle %in% c(0, 45, 90, 270)) {
    stop(sprintf("'axis_text_angle' must be one of [%s]", 
                 paste(c(0, 45, 90, 270), collapse = ", ")), 
         ".\nFor other angles, use the guide_axis() function in ggplot2 instead",
         call. = FALSE)
  }
  
  # Validate palette
  if (!palette %in% names(ggprism::ggprism_data$themes)) {
    stop("The palette ", paste(palette), " does not exist.\n", 
         "See names(ggprism_data$themes) for valid palette names")
  }
  
  # Get color scheme
  colours <- tibble::deframe(ggprism::ggprism_data$themes[[palette]])
  
  # Set border and axis line properties
  if (!is_bool(border)) {
    stop("border must be either: TRUE or FALSE")
  } else {
    if (border) {
      panel.border <- element_rect(fill = NA)
      axis.line <- element_blank()
    } else if (!border) {
      panel.border <- element_blank()
      axis.line <- element_line(linewidth = base_line_size)
    }
  }
  
  # Create the theme
  t <- theme(
    # Base elements
    line = element_line(
      colour = colours["axisColor"], 
      linewidth = base_line_size,
      linetype = 1, 
      lineend = "square"
    ),
    
    rect = element_rect(
      fill = "white", 
      colour = colours["axisColor"],
      linewidth = base_rect_size, 
      linetype = 1
    ),
    
    text = element_text(
      family = base_family,
      face = base_fontface, 
      colour = colours["graphTitleColor"], 
      size = base_size,
      lineheight = 0.9, 
      hjust = 0.5, 
      vjust = 0.5, 
      angle = 0, 
      margin = margin(),
      debug = FALSE
    ),
    
    # Prism-specific
    prism.ticks.length = unit(base_size/50, "pt"),
    
    # Axis lines
    axis.line = axis.line,
    axis.line.x = NULL,
    axis.line.y = NULL,
    
    # Axis text
    axis.text = element_text(
      size = 20,
      colour = colours["axisLabelColor"]
    ),
    
    axis.text.x = element_text(
      margin = margin(t = 0.8 * base_size/4),
      angle = axis_text_angle,
      hjust = ifelse(axis_text_angle %in% c(45, 90, 270), 1, 0.5),
      vjust = ifelse(axis_text_angle %in% c(0, 90, 270), 0.5, 1)
    ),
    
    axis.text.x.top = element_text(
      margin = margin(b = 0.8 * base_size/4), 
      vjust = 0
    ),
    
    axis.text.y = element_text(
      margin = margin(r = 0.5 * base_size/4), 
      hjust = 1
    ),
    
    axis.text.y.right = element_text(
      margin = margin(l = 0.5 * base_size/4), 
      hjust = 0
    ),
    
    # Axis ticks
    axis.ticks = element_line(linewidth = base_line_size),
    axis.ticks.length = unit(3, "points"),
    axis.ticks.length.x = NULL,
    axis.ticks.length.x.top = NULL,
    axis.ticks.length.x.bottom = NULL,
    axis.ticks.length.y = NULL,
    axis.ticks.length.y.left = NULL,
    axis.ticks.length.y.right = NULL,
    
    # Axis titles
    axis.title = element_text(
      colour = colours["axisTitleColor"],
      size = 25
    ),
    
    axis.title.x = element_text(
      margin = margin(t = base_size * 0.6), 
      vjust = 1
    ),
    
    axis.title.x.top = element_text(
      margin = margin(b = base_size * 0.6), 
      vjust = 0
    ),
    
    axis.title.y = element_text(
      angle = 90, 
      margin = margin(r = base_size * 0.6),
      vjust = 1
    ),
    
    axis.title.y.right = element_text(
      angle = -90, 
      margin = margin(l = base_size * 0.6), 
      vjust = 0
    ),
    
    # Legend
    legend.background = element_blank(),
    legend.spacing = unit(base_size, "pt"),
    legend.spacing.x = NULL,
    legend.spacing.y = NULL,
    legend.margin = margin(base_size/2, base_size/2, base_size/2, base_size/2),
    legend.key = element_blank(),
    legend.key.size = unit(1.2, "lines"),
    legend.key.height = NULL,
    legend.key.width = unit(base_size * 1.8, "pt"),
    legend.text = element_text(size = 17, face = "plain"),
    legend.text.align = NULL,
    legend.title = element_text(size = 25),
    legend.title.align = NULL,
    legend.position = "right",
    legend.direction = NULL,
    legend.justification = "center",
    legend.box = NULL,
    legend.box.margin = margin(0, 0, 0, 0, "cm"),
    legend.box.background = element_blank(),
    legend.box.spacing = unit(base_size, "pt"),
    
    # Panel
    panel.background = element_rect(
      fill = ifelse(palette == "office", colours["plottingAreaColor"], NA), 
      colour = NA
    ),
    panel.border = panel.border,
    panel.grid = element_blank(),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(base_size/2, "pt"),
    panel.spacing.x = NULL,
    panel.spacing.y = NULL,
    panel.ontop = FALSE,
    
    # Strip (facet labels)
    strip.background = element_blank(),
    strip.text = element_text(
      colour = colours["axisTitleColor"],
      size = 25, 
      margin = margin(base_size/2.5, base_size/2.5, base_size/2.5, base_size/2.5)
    ),
    strip.text.x = element_text(margin = margin(b = base_size/3)),
    strip.text.y = element_text(angle = -90, margin = margin(l = base_size/3)),
    strip.text.y.left = element_text(angle = 90),
    strip.placement = "inside",
    strip.placement.x = NULL,
    strip.placement.y = NULL,
    strip.switch.pad.grid = unit(base_size/4, "pt"),
    strip.switch.pad.wrap = unit(base_size/4, "pt"),
    
    # Plot elements
    plot.background = element_rect(
      fill = colours["pageBackgroundColor"],
      colour = NA
    ),
    
    plot.title = element_text(
      size = rel(1.2), 
      hjust = 0.5,
      vjust = 1, 
      margin = margin(b = base_size)
    ),
    plot.title.position = "panel",
    
    plot.subtitle = element_text(
      hjust = 0.5, 
      vjust = 1, 
      margin = margin(b = base_size/2)
    ),
    
    plot.caption = element_text(
      size = rel(0.8), 
      hjust = 1, 
      vjust = 1, 
      margin = margin(t = base_size/2)
    ),
    plot.caption.position = "panel",
    
    plot.tag = element_text(
      size = rel(1.2),
      hjust = 0.5, 
      vjust = 0.5
    ),
    plot.tag.position = "topleft",
    
    plot.margin = margin(base_size/2, base_size/2, base_size/2, base_size/2),
    
    complete = TRUE
  )
  
  # Return the theme
  ggprism::ggprism_data$themes[["all_null"]] %+replace% t
}