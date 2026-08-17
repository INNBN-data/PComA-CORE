# Decision-curve analysis for a generated temporal prediction file.
# Requires rmda and ggplot2.

library(rmda)
library(ggplot2)

path <- "models/pcoma_c/temporal_predictions.csv"
if (!file.exists(path)) stop("Run temporal evaluation first: ", path)

df <- read.csv(path)
if (!all(c("observed", "probability") %in% names(df))) {
  stop("Temporal predictions must contain observed and probability columns.")
}

analysis <- data.frame(outcome = df$observed, probability = df$probability)
dca <- decision_curve(
  outcome ~ probability,
  data = analysis,
  thresholds = seq(0.10, 0.30, by = 0.01),
  confidence.intervals = FALSE,
  study.design = "cohort"
)

dir.create("figures/generated", recursive = TRUE, showWarnings = FALSE)
png("figures/generated/pcoma_c_decision_curve.png", width = 1800, height = 1200, res = 200)
plot_decision_curve(
  dca,
  curve.names = "PComA-C",
  xlab = "Threshold probability",
  ylab = "Net benefit",
  standardize = FALSE
)
dev.off()
