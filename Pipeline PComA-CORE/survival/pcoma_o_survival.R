library(survival)

path <- "data/features_development.csv"
if (!file.exists(path)) stop("Run preprocessing first: ", path)

df <- read.csv(path)
df_o <- subset(df, preop_cn3_palsy == 1 & !is.na(pcoma_o_event) & !is.na(pcoma_o_time_days))
if (nrow(df_o) == 0) stop("No eligible PComA-O observations.")

fit <- coxph(
  Surv(pcoma_o_time_days, pcoma_o_event) ~
    preop_cn3_duration_days + cn3_grooving_present +
    preop_cn3_complete_or_partial + neck_width_mm + age_years +
    pcoa_diameter_mm + cn3_compression_grade + preop_cn3_pupil_involvement,
  data = df_o,
  x = TRUE
)

print(summary(fit))
dir.create("results/generated", recursive = TRUE, showWarnings = FALSE)
saveRDS(fit, "results/generated/pcoma_o_cox_reference.rds")
