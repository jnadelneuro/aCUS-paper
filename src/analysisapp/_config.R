# Central path configuration for the aCUS-paper repo (R side).
# Usage in a script:
#   source("_config.R")   # or the path to it
#   d <- arrow::read_feather(file.path(RI60_DIR, "photoDF_R_with_weights.feather"))
#
# Reads config/config.yaml (copy from config/config.example.yaml). A copy of this
# file is placed in each src/ pipeline folder so source("_config.R") resolves locally.

.find_config <- function() {
  d <- getwd()
  for (i in 1:8) {
    p <- file.path(d, "config", "config.yaml")
    if (file.exists(p)) return(p)
    d <- dirname(d)
  }
  stop("config/config.yaml not found - copy config/config.example.yaml to config/config.yaml")
}

.cfg <- yaml::read_yaml(.find_config())

DATA_ROOT      <- .cfg$paths$data_root
AVOIDANCE_ROOT <- .cfg$paths$avoidance_root
FIG_OUT        <- .cfg$paths$fig_out

RI60_DIR      <- file.path(DATA_ROOT, .cfg$subpaths$ri60)
INTRINSIC_DIR <- file.path(DATA_ROOT, .cfg$subpaths$intrinsic)
MODEL_DIR     <- file.path(DATA_ROOT, .cfg$subpaths$model)
RNASEQ_DIR    <- file.path(DATA_ROOT, .cfg$subpaths$rnaseq)
