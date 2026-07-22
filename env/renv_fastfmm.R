# R packages for the FLMM (functional linear mixed modeling) fits — Fig 5 / S7.
# Run once:  Rscript env/renv_fastfmm.R
install.packages(c(
  "fastFMM",
  "refund",
  "lme4",
  "mgcv",
  "MASS",
  "arrow",
  "dplyr",
  "yaml"
), repos = "https://cloud.r-project.org")
