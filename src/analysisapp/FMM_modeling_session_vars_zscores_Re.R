library(fastFMM)

patch_fastFMM_parallel <- function() {
  message("Applying hotfix to fastFMM:::var_analytic...")
  
  va_fun <- fastFMM:::var_analytic
  va_src <- deparse(body(va_fun), width.cutoff = 500)
  
  # Find the line with the randintercept Z assignment and insert Z <- NULL before it
  target <- '    if (!randintercept) {'
  hit <- which(va_src == target)
  
  # There are two such blocks; we want the first one (around line 53 of body)
  # specifically the one followed by Z <- data_cov$Z_orig
  z_block <- which(va_src == '        Z <- data_cov$Z_orig')
  
  if (length(z_block) == 0) {
    warning("Patch Failed: Could not find Z assignment block. Version might differ.")
    return()
  }
  
  # The if (!randintercept) immediately before the Z assignment
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

# Proceed with your analysis
# ...
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
#library(arrangements) 
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

# Set Base Directory
source("_config.R")
base_dir <- RI60_DIR
output_dir <- file.path(RI60_DIR, "FMM_models")
setwd(base_dir)



# Function: Clean Data via CSV (Required hack for FUI)
clean_via_csv <- function(df, filename_base) {
  fname <- paste0(filename_base, ".csv")
  write.csv(df, file = fname, row.names = FALSE)
  
  read_df <- read.csv(fname)
  # Filter NAs as per original script logic
  if("timeSince_ReNP" %in% names(read_df)){
    read_df <- read_df[!is.na(read_df$timeSince_ReNP), ]
  }
  return(read_df)
}


# Function: ROUT Outlier Removal (Session-Based)
# 1. Calculates mean per session (Mouse x Day)
# 2. Identifies outlier sessions using ROUT (Median/MAD + FDR)
# 3. Removes ALL trials for those sessions
remove_rout_outliers <- function(data, col_name, Q = 0.001) {
  if (!col_name %in% names(data)) return(data)
  
  # Ensure we have session identifiers
  if(!all(c("mouse", "dayOnType") %in% names(data))) {
    warning("Cannot perform session-based outlier removal: 'mouse' or 'dayOnType' columns missing.")
    return(data)
  }
  
  # 1. Summarize by Session (Mouse + Day)
  # We use dplyr::select to avoid conflict with MASS::select
  session_summary <- data %>%
    group_by(mouse, dayOnType) %>%
    summarise(mean_val = mean(.data[[col_name]], na.rm = TRUE), .groups = "drop") %>%
    filter(!is.na(mean_val))
  
  vals <- session_summary$mean_val
  
  # 2. Calculate Robust Statistics (Median and MAD)
  med_val <- median(vals, na.rm = TRUE)
  mad_val <- mad(vals, constant = 1.4826, na.rm = TRUE)
  
  if (mad_val == 0) {
    message(paste("MAD is 0 for", col_name, "- skipping outlier removal."))
    return(data)
  }
  
  # Calculate Robust Z-scores
  z_scores <- abs(vals - med_val) / mad_val
  
  # Calculate P-values (Two-tailed)
  p_vals <- 2 * (1 - pnorm(z_scores))
  
  # Apply FDR (Benjamini-Hochberg)
  is_outlier <- rep(FALSE, length(vals))
  valid_idx <- !is.na(p_vals)
  
  if (sum(valid_idx) > 0) {
    p_adj <- p.adjust(p_vals[valid_idx], method = "BH")
    is_outlier[valid_idx] <- p_adj < Q
  }
  
  # 3. Identify Outlier Sessions
  outlier_sessions <- session_summary[is_outlier, ]
  
  # Report and Filter
  n_sessions_removed <- nrow(outlier_sessions)
  
  if (n_sessions_removed > 0) {
    message(paste("ROUT (Q=", Q*100, "%): Removed", n_sessions_removed, "outlier session(s) for", col_name))
    
    # Create a composite key for filtering
    data <- data %>%
      mutate(session_id = paste(mouse, dayOnType, sep = "_"))
    
    outlier_ids <- paste(outlier_sessions$mouse, outlier_sessions$dayOnType, sep = "_")
    
    # Filter out all trials belonging to outlier sessions
    data_clean <- data %>% filter(!session_id %in% outlier_ids) %>% dplyr::select(-session_id)
    return(data_clean)
  }
  
  return(data)
}

run_zscored_pair_formula <- function(df_stress, df_naive, formula_str, region_label, target_event,
                                     Q_rout = 0.001, tag = NULL) {
  # Extract predictor names from RHS of formula
  f <- as.formula(formula_str)
  col_names <- all.vars(f[[3]])
  
  df_stress_clean <- df_stress
  df_naive_clean  <- df_naive
  for (cn in col_names) {
    if (cn == 'reward_rate') {
      df_stress_clean <- df_stress_clean[!is.nan(df_stress_clean$reward_rate), ]
      df_naive_clean  <- df_naive_clean[!is.nan(df_naive_clean$reward_rate), ]
    }
    df_stress_clean <- remove_rout_outliers(df_stress_clean, cn, Q = Q_rout)
    df_naive_clean  <- remove_rout_outliers(df_naive_clean,  cn, Q = Q_rout)
  }
  
  for (cn in col_names) {
    pooled_vals <- c(df_stress_clean[[cn]], df_naive_clean[[cn]])
    pooled_vals <- pooled_vals[is.finite(pooled_vals)]
    pooled_mean <- mean(pooled_vals, na.rm = TRUE)
    pooled_sd   <- sd(pooled_vals, na.rm = TRUE)
    if (is.na(pooled_sd) || pooled_sd == 0) {
      warning(paste("Pooled SD is 0 or NA for", cn, "in", region_label, "- skipping whole call."))
      return(invisible(NULL))
    }
    message(sprintf("  [%s %s] pooled mean=%.4f sd=%.4f", region_label, cn, pooled_mean, pooled_sd))
    df_stress_clean[[cn]] <- (df_stress_clean[[cn]] - pooled_mean) / pooled_sd
    df_naive_clean[[cn]]  <- (df_naive_clean[[cn]]  - pooled_mean) / pooled_sd
  }
  
  tag_str <- if (is.null(tag)) paste(col_names, collapse = "_AND_") else tag
  out_stress <- paste0(target_event, "_FMM_stress_", region_label, "_", tag_str, "_zscored.rds")
  out_naive  <- paste0(target_event, "_FMM_naive_",  region_label, "_", tag_str, "_zscored.rds")
  run_fui_analysis(df_stress_clean, formula_str, out_stress)
  run_fui_analysis(df_naive_clean,  formula_str, out_naive)
}

run_interaction_fmm <- function(df_combined, formula_str, predictors_to_zscore,
                                region_label, target_event, Q_rout = 0.001, tag) {
  # Split for ROUT (per-group), then concatenate
  df_s <- df_combined[df_combined$group == 1, ]
  df_n <- df_combined[df_combined$group == 0, ]
  for (cn in predictors_to_zscore) {
    if (cn == 'reward_rate') {
      df_s <- df_s[!is.nan(df_s$reward_rate), ]
      df_n <- df_n[!is.nan(df_n$reward_rate), ]
    }
    df_s <- remove_rout_outliers(df_s, cn, Q = Q_rout)
    df_n <- remove_rout_outliers(df_n, cn, Q = Q_rout)
  }
  df_all <- rbind(df_s, df_n)
  
  # Pooled z-score on the combined post-ROUT data
  for (cn in predictors_to_zscore) {
    v <- df_all[[cn]]
    v <- v[is.finite(v)]
    m <- mean(v, na.rm = TRUE); s <- sd(v, na.rm = TRUE)
    if (is.na(s) || s == 0) {
      warning(paste("Pooled SD is 0/NA for", cn, "- skipping."))
      return(invisible(NULL))
    }
    message(sprintf("  [%s %s] pooled mean=%.4f sd=%.4f", region_label, cn, m, s))
    df_all[[cn]] <- (df_all[[cn]] - m) / s
  }
  
  out_file <- paste0(target_event, "_FMM_interaction_", region_label, "_", tag, "_zscored.rds")
  run_fui_analysis(df_all, formula_str, out_file)
}

run_fui_analysis <- function(data, fixed_formula, output_filename) {
  if (file.exists(output_filename)) {
    message(paste("File exists, skipping:", output_filename))
    return(NULL)
  }
  
  message(paste("Running FUI for:", output_filename))
  slurm_cores <- Sys.getenv("SLURM_CPUS_PER_TASK")
  
  if (slurm_cores == "") {
    n_cores <- 1L
  } else {
    n_cores <- as.numeric(slurm_cores)
  }
  
  n_cores <- 5
  # IMPORTANT: Tibble fix for fastFMM compatibility
  trace_cols <- grep("^trimTrace", names(data), value = TRUE)
  rhs_vars   <- all.vars(as.formula(fixed_formula))
  rhs_vars   <- setdiff(rhs_vars, "trimTrace")
  keep_cols  <- unique(c(trace_cols, rhs_vars, "mouse"))
  keep_cols  <- intersect(keep_cols, names(data))   # guard against any other stragglers
  data <- as.data.frame(na.omit(data[, keep_cols, drop = FALSE]))  
  
  full_formula <- as.formula(paste(fixed_formula, "+ (1 | mouse)"))
  
  n_cores <-
  
  # Run Model
  mod <- fui(formula = full_formula,
             data = data,
             analytic = TRUE, # Change to FALSE if you want bootstrap trust
             parallel = TRUE,
             n_cores = n_cores) # Keep FALSE to avoid "Object Z" bug with analytic=TRUE
  
  # Generate Plot Data
  plot_obj <- plot_fui(mod,
                       x_rescale = 51,
                       align_x = 2,
                       xlab = "Time (s)",
                       return = TRUE)
  
  # Save RDS
  saveRDS(plot_obj, file = output_filename)
}

# Function: run stress+naive FUI models with pooled z-scored predictor
# Does ROUT per group, computes pooled mean/SD across both post-ROUT groups,
# applies that same z-score to both groups, then fits each.
run_zscored_pair <- function(df_stress, df_naive, col_name, region_label, target_event, Q_rout = 0.001) {
  if (col_name == 'reward_rate') {
    df_stress <- df_stress[!is.nan(df_stress$reward_rate), ]
    df_naive  <- df_naive[!is.nan(df_naive$reward_rate), ]
  }
  # ROUT Outlier Removal (Session-Based, Q=0.1%)
  df_stress_clean <- remove_rout_outliers(df_stress, col_name, Q = Q_rout)
  df_naive_clean  <- remove_rout_outliers(df_naive,  col_name, Q = Q_rout)
  
  # Pooled z-params from post-ROUT data, applied identically to both groups
  pooled_vals <- c(df_stress_clean[[col_name]], df_naive_clean[[col_name]])
  pooled_vals <- pooled_vals[is.finite(pooled_vals)]
  pooled_mean <- mean(pooled_vals, na.rm = TRUE)
  pooled_sd   <- sd(pooled_vals, na.rm = TRUE)
  if (is.na(pooled_sd) || pooled_sd == 0) {
    warning(paste("Pooled SD is 0 or NA for", col_name, "in", region_label, "- skipping."))
    return(invisible(NULL))
  }
  message(sprintf("  [%s %s] pooled mean=%.4f sd=%.4f", region_label, col_name, pooled_mean, pooled_sd))
  df_stress_clean[[col_name]] <- (df_stress_clean[[col_name]] - pooled_mean) / pooled_sd
  df_naive_clean[[col_name]]  <- (df_naive_clean[[col_name]]  - pooled_mean) / pooled_sd
  
  formula_str <- paste0("trimTrace ~ ", col_name)
  out_stress <- paste0(target_event, "_FMM_stress_", region_label, "_", col_name, "_zscored.rds")
  out_naive  <- paste0(target_event, "_FMM_naive_",  region_label, "_", col_name, "_zscored.rds")
  run_fui_analysis(df_stress_clean, formula_str, out_stress)
  run_fui_analysis(df_naive_clean,  formula_str, out_naive)
}
# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================

# photoDF <- arrow::read_feather('photoDF_R.feather')
# Use the timeToRePE feather: it is photoDF_R_with_weights.feather + the
# timeTo_RePE column (same 45,412 rows; numShocks/pi_*/bout cols all present).
# The plain weights file has NO timeTo_RePE, so log(timeTo_RePE) below errors.
photoDF <- arrow::read_feather('photoDF_R_with_weights_timeToRePE.feather')


target_event <- 'UnNP'
photoDF$log_timeTo_RePE <- log(photoDF$timeTo_RePE)

# Basic cleaning and filtering
photoDF <- photoDF %>%
  filter(event == target_event, sesType == 'RI60') %>%
  mutate(
    group = case_when(
      group == "naive" ~ 0,
      group == "stress" ~ 1,
      TRUE ~ as.numeric(group)
    ),
    sex = case_when(
      sex == "M" ~ 0,
      sex == "F" ~ 1,
      TRUE ~ as.numeric(sex)
    )
  )

# Split by recording location
TS_photoDF <- photoDF[photoDF$recordingLoc == 'TS', ]
DMS_photoDF <- photoDF[photoDF$recordingLoc == 'DMS', ]

# Downsample trace (keep every 2nd point)
TS_photoDF$trimTrace <- lapply(TS_photoDF$trimTrace, function(x) x[seq(1, length(x), by = 2)])
DMS_photoDF$trimTrace <- lapply(DMS_photoDF$trimTrace, function(x) x[seq(1, length(x), by = 2)])

# Remove unnecessary columns to save memory
drop_cols <- c('recordingLoc', 'event', 'sesType', 'trimTime')
TS_photoDF <- TS_photoDF[, !(names(TS_photoDF) %in% drop_cols)]
DMS_photoDF <- DMS_photoDF[, !(names(DMS_photoDF) %in% drop_cols)]

# Explode trace columns
TS_exploded <- TS_photoDF %>% unnest_wider(trimTrace, names_sep = "_")
DMS_exploded <- DMS_photoDF %>% unnest_wider(trimTrace, names_sep = "_")

# Create Subsets (Stress vs Naive) and Filter Time
# Helper to filter time range [0, 200]
filter_time <- function(df) {
  df[!is.na(df$timeSince_ReNP) & df$timeSince_ReNP >= 0 & df$timeSince_ReNP <= 200, ]
}



df_TS_stress <- filter_time(TS_exploded[TS_exploded$group == 1, ])
df_TS_naive  <- filter_time(TS_exploded[TS_exploded$group == 0, ])
df_DMS_stress <- filter_time(DMS_exploded[DMS_exploded$group == 1, ])
df_DMS_naive  <- filter_time(DMS_exploded[DMS_exploded$group == 0, ])

# Calculate Diff from 60
df_TS_stress$time_diff_from_60 <- abs(df_TS_stress$timeSince_ReNP - 60)
df_TS_naive$time_diff_from_60  <- abs(df_TS_naive$timeSince_ReNP - 60)
# Calculate Diff from 60
df_DMS_stress$time_diff_from_60 <- abs(df_DMS_stress$timeSince_ReNP - 60)
df_DMS_naive$time_diff_from_60  <- abs(df_DMS_naive$timeSince_ReNP - 60)

# ==============================================================================
# CSV WRITE/READ CYCLE (CRITICAL STEP)
# ==============================================================================

df_TS         <- clean_via_csv(TS_exploded, "TS_photoDF")
df_DMS        <- clean_via_csv(DMS_exploded, "DMS_photoDF")

df_TS_stress  <- clean_via_csv(df_TS_stress, "df_TS_stress")
df_TS_naive   <- clean_via_csv(df_TS_naive, "df_TS_naive")

df_DMS_stress <- clean_via_csv(df_DMS_stress, "df_DMS_stress")
df_DMS_naive  <- clean_via_csv(df_DMS_naive, "df_DMS_naive")

# ==============================================================================
# MODELING
# ==============================================================================
setwd(output_dir)

# Drop trials with any NA in their trace
drop_na_traces <- function(df) {
  tc <- grep("^trimTrace", names(df), value = TRUE)
  df[complete.cases(df[, tc]), ]
}
df_TS         <- drop_na_traces(df_TS)
df_DMS        <- drop_na_traces(df_DMS)
df_TS_stress  <- drop_na_traces(df_TS_stress)
df_TS_naive   <- drop_na_traces(df_TS_naive)
df_DMS_stress <- drop_na_traces(df_DMS_stress)
df_DMS_naive  <- drop_na_traces(df_DMS_naive)
# Loop through predictors, pooled-z-scoring each across stress + naive so that
# the β(t) curves from the two group-specific models are directly comparable.
# colnames <- c('numShocks','pokeRate', 'entryRate', 'reward_rate', 'cumulative_poke', 'pi_0', 'pi_1', 'pi_2', 'pi_3', 'pi_4')
# colnames <- c('poke_component', 'poke_within_bout','bout_status')
# colnames <- c('numShocks')
# colnames <- c('log_timeTo_RePE')
colnames <- c('reward_rate', 'cumulative_poke','instant_poke_rate')

for (cn in colnames) {
  # print()
  if (cn == 'timeTo_RePE' || cn == 'log_timeTo_RePE') {
    df_TS_stress1  <- df_TS_stress[!is.na(df_TS_stress$timeTo_RePE)   & df_TS_stress$timeTo_RePE   <= 60, ]
    # df_DMS_stress1 <- df_DMS_stress[!is.na(df_DMS_stress$timeTo_RePE) & df_DMS_stress$timeTo_RePE <= 60, ]
    df_TS_naive1   <- df_TS_naive[!is.na(df_TS_naive$timeTo_RePE)     & df_TS_naive$timeTo_RePE   <= 60, ]
    # df_DMS_naive1  <- df_DMS_naive[!is.na(df_DMS_naive$timeTo_RePE)   & df_DMS_naive$timeTo_RePE <= 60, ]
  } else {
    df_TS_stress1  <- df_TS_stress
    # df_DMS_stress1 <- df_DMS_stress
    df_TS_naive1   <- df_TS_naive
    # df_DMS_naive1  <- df_DMS_naive
    }
    
  f_int <- paste0("trimTrace ~ group * ", cn)
  
  # run_interaction_fmm(df_DMS, formula_str = f_int, predictors_to_zscore = cn,
                      # region_label = "DMS", target_event = target_event,
                      # tag = paste0(cn, "_groupInteraction"))
  run_interaction_fmm(df_TS,  formula_str = f_int, predictors_to_zscore = cn,
                      region_label = "TS",  target_event = target_event,
                      tag = paste0(cn, "_groupInteraction"))
  
  message(paste("=== Predictor:", cn, "==="))
  run_zscored_pair(df_TS_stress1,  df_TS_naive1,  cn, "TS",  target_event)
  # run_zscored_pair(df_DMS_stress1, df_DMS_naive1, cn, "DMS", target_event)
  }

# run_interaction_fmm(df_DMS,
#                     formula_str = "trimTrace ~ group * cumulative_poke + group * reward_rate",
#                     predictors_to_zscore = c("cumulative_poke", "reward_rate"),
#                     region_label = "DMS",
#                     target_event = target_event,
#                     tag = "cumPoke_rewardRate_interaction")
# 
# run_interaction_fmm(df_TS,
#                     formula_str = "trimTrace ~ group * cumulative_poke + group * reward_rate",
#                     predictors_to_zscore = c("cumulative_poke", "reward_rate"),
#                     region_label = "TS",
#                     target_event = target_event,
#                     tag = "cumPoke_rewardRate_interaction")
# 


  # ==============================================================================
# EXPORT RDS RESULTS TO CSV
# ==============================================================================
# Define directory explicitly (or reuse base_dir if that was the intent)
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
