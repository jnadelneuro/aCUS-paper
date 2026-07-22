# R packages for CeA->SNL clustering (Fig 3A-D) and the Bayesian Dirichlet
# regression of mixture weights (Fig 1F).
# Run once:  Rscript env/renv_stats.R
install.packages(c(
  "brms",        # Bayesian Dirichlet regression (bayesian_dirichlet_for_muPI.R)
  "lme4",        # stress-LMM residuals prior to clustering (ClusterDeeZ.R)
  "FactoMineR",  # PCA
  "factoextra",
  "cluster",
  "dplyr",
  "yaml"
), repos = "https://cloud.r-project.org")
# brms requires a working C++ toolchain and rstan/cmdstanr; see https://mc-stan.org.
