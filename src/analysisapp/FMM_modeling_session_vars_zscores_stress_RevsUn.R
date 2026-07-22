library(fastFMM)

patch_fastFMM_parallel <- function() {
  message("Applying hotfix to fastFMM:::var_analytic...")
  
  va_fun <- fastFMM:::var_analytic
  va_src <- deparse(body(va_fun), width.cutoff = 500)
  
  target <- '    if (!randintercept) {'
  hit <- which(va_src == target)
  
  z_block <- which(va_src == '        Z <- data_cov$Z_orig')
  
  if (length(z_block) == 0) {
    warning("Patch Failed: Could not find Z assignment block. Version might differ.")
    return()
  }
  
  insert_after <- hit[which.min(abs(hit - z_block))] - 1
  
  va_src_patched <- c(
    va_src[1:insert_after],
    '    Z <- NULL',
    va_src[(insert_after + 1):length(va_src)]
  )
  
  body(va_fun) <- parse(text = paste(va_src_patched, collapse = "\n"))[[1]]
  assignInNamespace("var_analytic", va_fun, ns = "fastFMM")
  
  message("Success: Z <- NULL inserted before randintercept branch. parLapply can now evaluate Z = Z safely.")
}

patch_fastFMM_parallel()
# ======================================

library(lme4)
library(parallel)
library(cAIC4)
library(magrittr)
library(dplyr)
library(mgcv)
library(MASS)
library(lsei)
library(refund)
library(stringr)
library(Matrix)
library(mvtnorm)
library(progress)
library(ggplot2)
library(gridExtra)
library(Rfast)
library(arrow)
library(feather)
library(data.table)
library(tidyr)
library(doParallel)
library(foreach)

rm(list=ls())
gc()

# ==============================================================================
# CONFIGURATION & FUNCTIONS
# ==============================================================================

source("_config.R")
base_dir   <- RI60_DIR
output_dir <- file.path(RI60_DIR, "FMM_models")
setwd(base_dir)

# Naive-vs-stress analysis of the within-event (ReNP vs UnNP) contrast.
# eventType: 1 = ReNP, 0 = UnNP.  group: 0 = naive, 1 = stress.
# The eventType*predictor interaction is fit SEPARATELY for each group so the two
# group-specific surfaces are directly comparable and pair up in plot_FMMs.

clean_via_csv <- function(df, filename_base) {
  fname <- paste0(filename_base, ".csv")
  write.csv(df, file = fname, row.names = FALSE)
  read_df <- read.csv(fname)
  if ("timeSince_ReNP" %in% names(read_df)) {
    read_df <- read_df[!is.na(read_df$timeSince_ReNP), ]
  }
  return(read_df)
}

# Session-based ROUT outlier removal (Median/MAD + BH-FDR), removes whole sessions
remove_rout_outliers <- function(data, col_name, Q = 0.001) {
  if (!col_name %in% names(data)) return(data)
  if (!all(c("mouse", "dayOnType") %in% names(data))) {
    warning("Cannot perform session-based outlier removal: 'mouse' or 'dayOnType' columns missing.")
    return(data)
  }
  
  session_summary <- data %>%
    group_by(mouse, dayOnType) %>%
    summarise(mean_val = mean(.data[[col_name]], na.rm = TRUE), .groups = "drop") %>%
    filter(!is.na(mean_val))
  
  vals    <- session_summary$mean_val
  med_val <- median(vals, na.rm = TRUE)
  mad_val <- mad(vals, constant = 1.4826, na.rm = TRUE)
  
  if (mad_val == 0) {
    message(paste("MAD is 0 for", col_name, "- skipping outlier removal."))
    return(data)
  }
  
  z_scores <- abs(vals - med_val) / mad_val
  p_vals   <- 2 * (1 - pnorm(z_scores))
  
  is_outlier <- rep(FALSE, length(vals))
  valid_idx  <- !is.na(p_vals)
  if (sum(valid_idx) > 0) {
    p_adj <- p.adjust(p_vals[valid_idx], method = "BH")
    is_outlier[valid_idx] <- p_adj < Q
  }
  
  outlier_sessions <- session_summary[is_outlier, ]
  n_sessions_removed <- nrow(outlier_sessions)
  
  if (n_sessions_removed > 0) {
    message(paste("ROUT (Q=", Q*100, "%): Removed", n_sessions_removed, "outlier session(s) for", col_name))
    data <- data %>% mutate(session_id = paste(mouse, dayOnType, sep = "_"))
    outlier_ids <- paste(outlier_sessions$mouse, outlier_sessions$dayOnType, sep = "_")
    data_clean <- data %>% filter(!session_id %in% outlier_ids) %>% dplyr::select(-session_id)
    return(data_clean)
  }
  return(data)
}

run_fui_analysis <- function(data, fixed_formula, output_filename) {
  if (file.exists(output_filename)) {
    message(paste("File exists, skipping:", output_filename))
    return(NULL)
  }
  message(paste("Running FUI for:", output_filename))
  slurm_cores <- Sys.getenv("SLURM_CPUS_PER_TASK")
  n_cores <- if (slurm_cores == "") 1L else as.numeric(slurm_cores)
  
  trace_cols <- grep("^trimTrace", names(data), value = TRUE)
  rhs_vars   <- all.vars(as.formula(fixed_formula))
  rhs_vars   <- setdiff(rhs_vars, "trimTrace")
  keep_cols  <- unique(c(trace_cols, rhs_vars, "mouse"))
  keep_cols  <- intersect(keep_cols, names(data))
  data <- as.data.frame(na.omit(data[, keep_cols, drop = FALSE]))
  
  full_formula <- as.formula(paste(fixed_formula, "+ (1 | mouse)"))
  
  mod <- fui(formula = full_formula,
             data = data,
             analytic = TRUE,
             parallel = TRUE,
             n_cores = n_cores)
  
  plot_obj <- plot_fui(mod, x_rescale = 51, align_x = 2, xlab = "Time (s)", return = TRUE)
  saveRDS(plot_obj, file = output_filename)
}

# Interaction FMM on the within-event contrast (eventType: ReNP vs UnNP), fit
# SEPARATELY for naive and stress so the two group-specific eventType*predictor
# surfaces are directly comparable and pair up in plot_FMMs.
# ROUT per group x event-type cell, then pool the z-score across BOTH groups (and
# both events) so the naive and stress beta(t) curves land on one common scale.
run_event_interaction_fmm <- function(df_combined, formula_str, predictors_to_zscore,
                                      region_label, tag, Q_rout = 0.001) {
  # 4 cells: group (0 = naive, 1 = stress) x eventType (0 = UnNP, 1 = ReNP)
  cells <- list(
    naive_un  = df_combined[df_combined$group == 0 & df_combined$eventType == 0, ],
    naive_re  = df_combined[df_combined$group == 0 & df_combined$eventType == 1, ],
    stress_un = df_combined[df_combined$group == 1 & df_combined$eventType == 0, ],
    stress_re = df_combined[df_combined$group == 1 & df_combined$eventType == 1, ]
  )
  for (cn in predictors_to_zscore) {
    for (k in names(cells)) {
      if (cn == 'reward_rate') {
        cells[[k]] <- cells[[k]][!is.nan(cells[[k]]$reward_rate), ]
      }
      cells[[k]] <- remove_rout_outliers(cells[[k]], cn, Q = Q_rout)
    }
  }
  df_naive  <- rbind(cells$naive_un,  cells$naive_re)
  df_stress <- rbind(cells$stress_un, cells$stress_re)

  # Pooled z-params across BOTH groups (and both events) -> one common scale
  for (cn in predictors_to_zscore) {
    v <- c(df_naive[[cn]], df_stress[[cn]]); v <- v[is.finite(v)]
    m <- mean(v, na.rm = TRUE); s <- sd(v, na.rm = TRUE)
    if (is.na(s) || s == 0) {
      warning(paste("Pooled SD is 0/NA for", cn, "- skipping."))
      return(invisible(NULL))
    }
    message(sprintf("  [%s %s] pooled mean=%.4f sd=%.4f", region_label, cn, m, s))
    df_naive[[cn]]  <- (df_naive[[cn]]  - m) / s
    df_stress[[cn]] <- (df_stress[[cn]] - m) / s
  }

  out_naive  <- paste0("ReVsUn_FMM_interaction_naive_",  region_label, "_", tag, "_zscored.rds")
  out_stress <- paste0("ReVsUn_FMM_interaction_stress_", region_label, "_", tag, "_zscored.rds")
  run_fui_analysis(df_naive,  formula_str, out_naive)
  run_fui_analysis(df_stress, formula_str, out_stress)
}

# Per-event group-comparable single-predictor fits (ReNP vs UnNP get matched z-scoring).
# Renamed from the old stress/naive pairing; here df_a = ReNP, df_b = UnNP.
run_zscored_event_pair <- function(df_re, df_un, col_name, region_label, Q_rout = 0.001) {
  if (col_name == 'reward_rate') {
    df_re <- df_re[!is.nan(df_re$reward_rate), ]
    df_un <- df_un[!is.nan(df_un$reward_rate), ]
  }
  df_re_clean <- remove_rout_outliers(df_re, col_name, Q = Q_rout)
  df_un_clean <- remove_rout_outliers(df_un, col_name, Q = Q_rout)
  
  pooled_vals <- c(df_re_clean[[col_name]], df_un_clean[[col_name]])
  pooled_vals <- pooled_vals[is.finite(pooled_vals)]
  pooled_mean <- mean(pooled_vals, na.rm = TRUE)
  pooled_sd   <- sd(pooled_vals, na.rm = TRUE)
  if (is.na(pooled_sd) || pooled_sd == 0) {
    warning(paste("Pooled SD is 0 or NA for", col_name, "in", region_label, "- skipping."))
    return(invisible(NULL))
  }
  message(sprintf("  [%s %s] pooled mean=%.4f sd=%.4f", region_label, col_name, pooled_mean, pooled_sd))
  df_re_clean[[col_name]] <- (df_re_clean[[col_name]] - pooled_mean) / pooled_sd
  df_un_clean[[col_name]] <- (df_un_clean[[col_name]] - pooled_mean) / pooled_sd
  
  formula_str <- paste0("trimTrace ~ ", col_name)
  out_re <- paste0("ReNP_FMM_", region_label, "_", col_name, "_zscored.rds")
  out_un <- paste0("UnNP_FMM_", region_label, "_", col_name, "_zscored.rds")
  run_fui_analysis(df_re_clean, formula_str, out_re)
  run_fui_analysis(df_un_clean, formula_str, out_un)
}

# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================

photoDF <- arrow::read_feather('photoDF_R_with_weights.feather')
photoDF$log_timeTo_RePE <- log(photoDF$timeTo_RePE)

# Both groups, RI60, ReNP + UnNP. Make eventType + numeric group BEFORE any split/drop.
# group: 0 = naive, 1 = stress (matches the _Re.R / _Un.R coding).
photoDF <- photoDF %>%
  filter(sesType == 'RI60', event %in% c('ReNP', 'UnNP')) %>%
  mutate(
    eventType = ifelse(event == 'ReNP', 1, 0),
    group = case_when(group == "naive" ~ 0, group == "stress" ~ 1, TRUE ~ as.numeric(group)),
    sex = case_when(sex == "M" ~ 0, sex == "F" ~ 1, TRUE ~ as.numeric(sex))
  )

# Split by recording location
TS_photoDF  <- photoDF[photoDF$recordingLoc == 'TS', ]
DMS_photoDF <- photoDF[photoDF$recordingLoc == 'DMS', ]

# Downsample trace (keep every 2nd point)
TS_photoDF$trimTrace  <- lapply(TS_photoDF$trimTrace,  function(x) x[seq(1, length(x), by = 2)])
DMS_photoDF$trimTrace <- lapply(DMS_photoDF$trimTrace, function(x) x[seq(1, length(x), by = 2)])

# Drop unneeded cols (keep eventType, which we already created)
drop_cols <- c('recordingLoc', 'event', 'sesType', 'trimTime')
TS_photoDF  <- TS_photoDF[,  !(names(TS_photoDF)  %in% drop_cols)]
DMS_photoDF <- DMS_photoDF[, !(names(DMS_photoDF) %in% drop_cols)]

# Explode trace columns
TS_exploded  <- TS_photoDF  %>% unnest_wider(trimTrace, names_sep = "_")
DMS_exploded <- DMS_photoDF %>% unnest_wider(trimTrace, names_sep = "_")

# Time filter [0, 200] on timeSince_ReNP
filter_time <- function(df) {
  df[!is.na(df$timeSince_ReNP) & df$timeSince_ReNP >= 0 & df$timeSince_ReNP <= 200, ]
}

# Per-event subsets for the single-predictor Re-vs-Un fits. These stay WITHIN-STRESS
# (group == 1) so their ReNP_FMM_* / UnNP_FMM_* outputs are unchanged; only the
# eventType-interaction models above are split by naive vs stress.
df_TS_re  <- filter_time(TS_exploded[TS_exploded$eventType == 1  & TS_exploded$group == 1, ])
df_TS_un  <- filter_time(TS_exploded[TS_exploded$eventType == 0  & TS_exploded$group == 1, ])
df_DMS_re <- filter_time(DMS_exploded[DMS_exploded$eventType == 1 & DMS_exploded$group == 1, ])
df_DMS_un <- filter_time(DMS_exploded[DMS_exploded$eventType == 0 & DMS_exploded$group == 1, ])

# ==============================================================================
# CSV WRITE/READ CYCLE (CRITICAL STEP)
# ==============================================================================

df_TS  <- clean_via_csv(TS_exploded,  "TS_photoDF")    # combined ReNP+UnNP, for interaction
df_DMS <- clean_via_csv(DMS_exploded, "DMS_photoDF")

df_TS_re  <- clean_via_csv(df_TS_re,  "df_TS_re")
df_TS_un  <- clean_via_csv(df_TS_un,  "df_TS_un")
df_DMS_re <- clean_via_csv(df_DMS_re, "df_DMS_re")
df_DMS_un <- clean_via_csv(df_DMS_un, "df_DMS_un")

# ==============================================================================
# MODELING
# ==============================================================================
setwd(output_dir)

drop_na_traces <- function(df) {
  tc <- grep("^trimTrace", names(df), value = TRUE)
  df[complete.cases(df[, tc]), ]
}
df_TS     <- drop_na_traces(df_TS)
df_DMS    <- drop_na_traces(df_DMS)
df_TS_re  <- drop_na_traces(df_TS_re)
df_TS_un  <- drop_na_traces(df_TS_un)
df_DMS_re <- drop_na_traces(df_DMS_re)
df_DMS_un <- drop_na_traces(df_DMS_un)

# Predictors. The eventType:cn beta(t) is the test: reward-gated coupling if significant.
colnames <- c('cumulative_poke', 'reward_rate', 'instant_poke_rate')

for (cn in colnames) {
  f_int <- paste0("trimTrace ~ eventType * ", cn)
  
  run_event_interaction_fmm(df_DMS, formula_str = f_int, predictors_to_zscore = cn,
                            region_label = "DMS", tag = paste0(cn, "_eventInteraction"))
  run_event_interaction_fmm(df_TS,  formula_str = f_int, predictors_to_zscore = cn,
                            region_label = "TS",  tag = paste0(cn, "_eventInteraction"))
  
  message(paste("=== Predictor:", cn, "==="))
  run_zscored_event_pair(df_TS_re,  df_TS_un,  cn, "TS")
  run_zscored_event_pair(df_DMS_re, df_DMS_un, cn, "DMS")
}

# Combined model: eventType crossed with both poke + reward terms
run_event_interaction_fmm(df_DMS,
                          formula_str = "trimTrace ~ eventType * cumulative_poke + eventType * reward_rate",
                          predictors_to_zscore = c("cumulative_poke", "reward_rate"),
                          region_label = "DMS",
                          tag = "cumPoke_rewardRate_eventInteraction")

run_event_interaction_fmm(df_TS,
                          formula_str = "trimTrace ~ eventType * cumulative_poke + eventType * reward_rate",
                          predictors_to_zscore = c("cumulative_poke", "reward_rate"),
                          region_label = "TS",
                          tag = "cumPoke_rewardRate_eventInteraction")

# ==============================================================================
# EXPORT RDS RESULTS TO CSV
# ==============================================================================
# Convert every fastFMM RDS in the models dir into the per-component CSVs that
# plot_FMMs reads. Guarded by !file.exists, so it only writes new ones.
rds_directory <- file.path(base_dir, "FMM_Models")

if(dir.exists(rds_directory)) {
  rds_files <- list.files(path = rds_directory, pattern = "\\.rds$", full.names = TRUE)

  for (file in rds_files) {
    data <- readRDS(file)
    file_name <- tools::file_path_sans_ext(basename(file))

    # Iterate through objects in the RDS list
    for (name in names(data)) {
      df <- data[[name]]
      safe_name <- gsub("[^[:alnum:]_]", "_", name)
      csv_file_name <- file.path(rds_directory, paste0(file_name, "_", safe_name, ".csv"))

      # Check if object is data frame/matrix AND file doesn't exist
      if ((is.data.frame(df) || is.matrix(df)) && !file.exists(csv_file_name)) {
        write.csv(df, file = csv_file_name, row.names = FALSE)
      }
    }
  }
} else {
  warning(paste("Directory not found for final CSV export:", rds_directory))
}